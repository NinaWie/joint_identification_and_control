import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

from neural_control.dataset import QuadDataset
from train_base import TrainBase
from neural_control.drone_loss import (
    drone_loss_function, simply_last_loss, reference_loss, mse_loss,
    weighted_loss
)
from neural_control.trajectory.generate_trajectory import load_prepare_trajectory
from neural_control.dynamics.quad_dynamics_simple import SimpleDynamics
from neural_control.dynamics.quad_dynamics_flightmare import (
    FlightmareDynamics
)
from neural_control.dynamics.quad_dynamics_trained import LearntDynamics
from neural_control.controllers.network_wrapper import NetworkWrapper
from neural_control.environments.drone_env import QuadRotorEnvBase
from evaluate_drone import QuadEvaluator
from neural_control.models.hutter_model import Net
try:
    from neural_control.flightmare import FlightmareWrapper
except ModuleNotFoundError:
    pass


class TrainDrone(TrainBase):
    """
    Train a controller for a quadrotor
    """

    def __init__(self, train_dynamics, eval_dynamics, config):
        """
        param sample_in: one of "train_env", "eval_env", "real_flightmare"
        """
        self.config = config
        super().__init__(train_dynamics, eval_dynamics, **config)
        self.in_state_size = config["in_state_size"]

        # Create environment for evaluation
        if self.sample_in == "real_flightmare":
            self.eval_env = FlightmareWrapper(self.delta_t)
        elif self.sample_in == "eval_env":
            self.eval_env = QuadRotorEnvBase(self.eval_dynamics, self.delta_t)
        elif self.sample_in == "train_env":
            self.eval_env = QuadRotorEnvBase(self.train_dynamics, self.delta_t)
        else:
            raise ValueError(
                "sample in must be one of eval_env, train_env, real_flightmare"
            )

    def initialize_model(
        self,
        base_model=None,
        modified_params={},
        base_model_name="model_quad"
    ):
        # Load model or initialize model
        if base_model is not None:
            self.net = torch.load(os.path.join(base_model, base_model_name))
            # load std or other parameters from json
            config_path = os.path.join(base_model, "config.json")
            if not os.path.exists(config_path):
                print("Load old config..")
                config_path = os.path.join(base_model, "param_dict.json")
            with open(config_path, "r") as outfile:
                previous_parameters = json.load(outfile)
        else:
            self.net = Net(
                self.in_state_size,
                self.nr_actions,
                self.ref_dim,
                self.action_dim * self.nr_actions_rnn,
                conv=(self.nr_actions >= 5)
            )

        # update the used parameters:
        self.config["horizon"] = self.nr_actions
        self.config["ref_length"] = self.nr_actions
        self.config["thresh_div"] = self.thresh_div_start
        self.config["dt"] = self.delta_t
        self.config["take_every_x"] = self.self_play_every_x
        self.config["thresh_stable"] = self.thresh_stable_start
        for k, v in modified_params.items():
            if type(v) == np.ndarray:
                modified_params[k] = v.tolist()
        self.config["modified_params"] = modified_params

        with open(os.path.join(self.save_path, "config.json"), "w") as outfile:
            json.dump(self.config, outfile)

        # init dataset
        self.state_data = QuadDataset(**self.config)
        self.init_optimizer()

    def train_controller_model(
        self, current_state, action_seq, in_ref_states, ref_states
    ):
        # zero the parameter gradients
        self.optimizer_controller.zero_grad()
        # save the reached states
        intermediate_states = torch.zeros(
            current_state.size()[0], self.nr_actions, self.state_size
        )
        for k in range(self.nr_actions):
            # extract action
            action = action_seq[:, k]
            current_state = self.train_dynamics(
                current_state, action, dt=self.delta_t
            )
            intermediate_states[:, k] = current_state

        loss = simply_last_loss(
            intermediate_states, ref_states[:, -1], action_seq, printout=0
        )

        # Backprop
        loss.backward()
        self.optimizer_controller.step()
        return loss

    def run_epoch(self, train="controller"):
        # load only one trajectory
        # trajectory = torch.tensor(
        #     [
        #         load_prepare_trajectory(
        #             "data/traj_data_1",
        #             self.delta_t,
        #             self.speed_factor,
        #             test=0,
        #             path=None
        #         )
        #     ]
        # ).float()
        # # np.set_printoptions(precision=3, suppress=1)
        # # print(trajectory[0, :, :6].numpy())
        # trajectory = trajectory.repeat(8, 1, 1)
        # print(trajectory.size())
        eval_every = 10
        self.evaluate_model(0)
        # for i in range(1000):
        for i, trajectory in enumerate(self.trainloader, 0):
            traj_len = trajectory.size()[1]
            b_s = trajectory.size()[0]
            ind = np.random.choice(np.arange(self.nr_actions))

            traj_loss = 0

            while ind + 2 * self.nr_actions + 1 < traj_len:
                # for now: reset state to trajectory each time
                current_state = torch.cat(
                    (trajectory[:, ind], torch.zeros(b_s, 3)), dim=1
                )

                self.optimizer_controller.zero_grad()
                # target and placeholder for reached states
                intermediate_states = torch.zeros(
                    b_s, self.nr_actions, self.state_size
                )
                action_seq = torch.zeros(b_s, self.nr_actions, self.action_dim)
                target_states = trajectory[:,
                                           ind + 1:ind + self.nr_actions + 1]
                # if i % 10 == 0:
                #     print("---------------------------")
                #     np.set_printoptions(precision=2, suppress=1)
                #     print("current state", current_state[0].detach().numpy())
                #     print("target states", target_states[0].detach().numpy())
                # ------------- version 1 -----------------
                # in_state, in_ref_state = self.state_data.preprocess_data(
                #     current_state, target_states
                # )
                # action = self.net(in_state, in_ref_state)
                # action = torch.sigmoid(action)
                # action_seq = torch.reshape(
                #     action, (-1, self.nr_actions, self.action_dim)
                # )

                for j in range(self.nr_actions):
                    ref_traj = trajectory[:, ind + 1:ind + self.nr_actions + 1]
                    in_state, in_ref_state = self.state_data.preprocess_data(
                        current_state, ref_traj
                    )

                    # predict action
                    # action = action_seq[:, j] # version 1
                    action = self.net(in_state, in_ref_state)
                    action = torch.sigmoid(action)
                    # print(current_state.size(), action.size())
                    # print("action_seq", action_seq.size())
                    current_state = self.train_dynamics(
                        current_state, action, dt=self.delta_t
                    )
                    intermediate_states[:, j] = current_state
                    action_seq[:, j] = action  # del for version 1

                    ind += 1

                # if i % 10 == 0:
                #     print(
                #         "reached states",
                #         intermediate_states[0].detach().numpy()
                #     )

                loss = drone_loss_function(
                    intermediate_states, target_states, action_seq, printout=0
                )
                # Backprop
                loss.backward()

                for name, param in self.net.named_parameters():
                    self.writer.add_histogram(name + ".grad", param)
                traj_loss += loss.item()

                self.optimizer_controller.step()
                self.optimizer_steps += 1

            self.writer.add_scalar("Loss/train", traj_loss, i)
            if (i + 1) % eval_every == 0:
                print()
                print(
                    "finished set of trajectories", round(traj_loss, 2),
                    "after ", self.optimizer_steps
                )
                self.results_dict["loss"].append(traj_loss)
                self.results_dict["used_traj_d1"].append(i * self.batch_size)
                self.results_dict["samples_in_d1"].append(self.optimizer_steps)
                suc_mean, suc_std = self.evaluate_model((i + 1) // eval_every)
                self.writer.add_scalar("Success_mean", suc_mean, i)
                self.writer.add_scalar("Success_std", suc_std, i)
                for name, param in self.net.named_parameters():
                    self.writer.add_histogram(name, param)
                self.writer.flush()

    def evaluate_model(self, epoch):
        print("EPOCH", epoch)
        # EVALUATE
        controller = NetworkWrapper(self.net, self.state_data, **self.config)

        evaluator = QuadEvaluator(
            controller, self.eval_env, test_time=1, **self.config
        )
        # run with mpc to collect data
        # eval_env.run_mpc_ref("rand", nr_test=5, max_steps=500)
        # run without mpc for evaluation
        with torch.no_grad():
            suc_mean, suc_std = evaluator.run_eval(
                "rand",
                nr_test=config["nr_eval_runs"],
                thresh_stable=config["thresh_stable_eval"],
                thresh_div=config["thresh_div_eval"]
            )

        # self.sample_new_data(epoch)

        # # increase threshold
        # if epoch % 5 == 0 and self.config["thresh_div"] < self.thresh_div_end:
        #     self.config["thresh_div"] += .05
        #     print("increased thresh div", round(self.config["thresh_div"], 2))

        # save best model
        self.save_model(epoch, suc_mean)

        self.results_dict["mean_success"].append(suc_mean)
        self.results_dict["std_success"].append(suc_std)
        self.results_dict["thresh_div"].append(self.config["thresh_div"])
        return suc_mean, suc_std


def train_control(base_model, config):
    """
    Train a controller from scratch or with an initial model
    """
    modified_params = config["modified_params"]
    # TODO: might be problematic
    print(modified_params)
    train_dynamics = FlightmareDynamics(modified_params=modified_params)
    eval_dynamics = FlightmareDynamics(modified_params=modified_params)

    # make sure that also the self play samples are collected in same env
    config["sample_in"] = "train_env"

    trainer = TrainDrone(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    trainer.run_control(config)


def train_dynamics(base_model, config):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
    config["sample_in"] = "train_env"

    # train environment is learnt
    train_dynamics = LearntDynamics()
    eval_dynamics = FlightmareDynamics(modified_params)

    trainer = TrainDrone(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    # RUN
    trainer.run_dynamics(config)


def train_sampling_finetune(base_model, config):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
    config["sample_in"] = "eval_env"

    # train environment is learnt
    train_dynamics = FlightmareDynamics()
    eval_dynamics = FlightmareDynamics(modified_params=modified_params)

    trainer = TrainDrone(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    # RUN
    trainer.run_control(config, sampling_based_finetune=True)


if __name__ == "__main__":
    # LOAD CONFIG
    with open("configs/quad_config.json", "r") as infile:
        config = json.load(infile)

    # mod_params = {"mass": 1}
    # # {'translational_drag': np.array([0.7, 0.7, 0.7])}
    # config["modified_params"] = mod_params

    baseline_model = "trained_models/quad/first_try_all_traj"
    config["thresh_div_start"] = 1
    config["thresh_stable_start"] = 1.5

    config["save_name"] = "train_all_traj_further"

    # config["nr_epochs"] = 20

    # TRAIN
    train_control(baseline_model, config)
    # train_dynamics(baseline_model, config)
    # train_sampling_finetune(baseline_model, config)
    # FINE TUNING parameters:
    # self.thresh_div_start = 1
    # self.self_play = 1.5
    # self.epoch_size = 500
    # self.max_steps = 1000
    # self.self_play_every_x = 5
    # self.learning_rate = 0.0001
