import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import argparse

from neural_control.dataset import QuadDataset
from train_base import TrainBase
from neural_control.drone_loss import quad_mpc_loss
from neural_control.dynamics.quad_dynamics_simple import SimpleDynamics
from neural_control.dynamics.quad_dynamics_flightmare import (
    FlightmareDynamics
)
from neural_control.dynamics.quad_dynamics_trained import LearntQuadDynamics
from neural_control.dynamics.learnt_dynamics import LearntDynamics
from neural_control.dynamics.learnt_dynamics import LearntDynamicsMPC
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
            if previous_parameters["dt"] != self.delta_t:
                raise RuntimeWarning(
                    f"dt difference: {previous_parameters['dt']} in loaded\
                         config but {self.delta_t} now"
                )
        else:
            self.state_data = QuadDataset(
                self.epoch_size,
                self.self_play,
                reset_strength=self.reset_strength,
                max_drone_dist=self.max_drone_dist,
                ref_length=self.nr_actions,
                dt=self.delta_t
            )
            in_state_size = self.state_data.normed_states.size()[1]
            # +9 because adding 12 things but deleting position (3)
            self.net = Net(
                in_state_size,
                self.nr_actions,
                self.ref_dim,
                self.action_dim * self.nr_actions,
                conv=1
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
        self.state_data = QuadDataset(self.epoch_size, **self.config)
        if self.config.get("nr_test_data", 0) > 0:
            print("Initialize test dataset")
            self.test_data = QuadDataset(
                num_states=self.config["nr_test_data"],
                test_time=1,
                **self.config
            )
            self.testloader = torch.utils.data.DataLoader(
                self.test_data, batch_size=1, shuffle=False, num_workers=0
            )
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

        loss = quad_mpc_loss(
            intermediate_states, ref_states, action_seq, printout=0
        )

        # Backprop
        loss.backward()
        # for name, param in self.net.named_parameters():
        #     if param.grad is not None:
        #         self.writer.add_histogram(name + ".grad", param.grad)
        self.optimizer_controller.step()
        return loss

    def evaluate_model(self, epoch):
        # EVALUATE
        controller = NetworkWrapper(self.net, self.state_data, **self.config)

        self.state_data.num_self_play = 0
        print("--------- eval in trained simulator (D1 modified) --------")
        eval_dyn = (
            self.train_dynamics
            if isinstance(self.train_dynamics, LearntDynamics)
            or isinstance(self.train_dynamics, LearntDynamicsMPC) else None
        )
        evaluator = QuadEvaluator(
            controller, self.eval_env, eval_dyn=eval_dyn, **self.config
        )
        with torch.no_grad():
            res_eval = evaluator.run_eval("rand", nr_test=10, **self.config)

        # logging
        if self.config["return_div"]:
            suc_mean, suc_std = (res_eval["mean_div"], res_eval["std_div"])
        else:
            suc_mean, suc_std = (
                res_eval["mean_stable"], res_eval["std_stable"]
            )
        for key, val in res_eval.items():
            self.results_dict[key].append(val)
        self.results_dict["eval_in_d1_trained_mean"].append(suc_mean)
        self.results_dict["eval_in_d1_trained_std"].append(suc_std)

        # if epoch == 0 and self.config["train_dyn_for_epochs"] >= 0:
        #     self.tmp_num_selfplay = self.state_data.num_self_play
        #     self.state_data.num_self_play = 0
        #     print("stop self play")
        # if epoch == self.config["train_dyn_for_epochs"]:
        #     self.state_data.num_self_play = self.tmp_num_selfplay
        #     print("start self play to", self.tmp_num_selfplay)

        ### code to evaluate also in D1 and D2
        ### need to ensure that eval_env is with train_dynamics
        # # set self play to zero so no sampled data is added
        # tmp_self_play = self.state_data.num_self_play
        # self.state_data.num_self_play = 0
        # print("--------- eval in real (D2) -------------")
        # d2_env = QuadRotorEnvBase(self.eval_dynamics, self.delta_t)
        # evaluator = QuadEvaluator(
        #     controller, d2_env, test_time=1, **self.config
        # )
        # with torch.no_grad():
        #     suc_mean, suc_std = evaluator.run_eval(
        #         "rand", nr_test=10, **self.config
        #     )
        # self.results_dict["eval_in_d2_mean"].append(suc_mean)
        # self.results_dict["eval_in_d2_std"].append(suc_std)

        # print("--------- eval in base simulator (D1) -------------")
        # base_env = QuadRotorEnvBase(FlightmareDynamics(), self.delta_t)
        # evaluator = QuadEvaluator(controller, base_env, **self.config)
        # with torch.no_grad():
        #     suc_mean, suc_std = evaluator.run_eval(
        #         "rand", nr_test=10, **self.config
        #     )
        # self.results_dict["eval_in_d1_mean"].append(suc_mean)
        # self.results_dict["eval_in_d1_std"].append(suc_std)
        # self.state_data.num_self_play = tmp_self_play

        # self.sample_new_data(epoch)

        # increase threshold
        if epoch % 5 == 0 and self.config["thresh_div"] < self.thresh_div_end:
            self.config["thresh_div"] += .05
            print("increased thresh div", round(self.config["thresh_div"], 2))

        # save best model
        self.save_model(epoch, suc_mean, suc_std)

        self.results_dict["thresh_div"].append(self.config["thresh_div"])
        return suc_mean, suc_std

    def collect_data(self, allocate=False, random=False, use_mpc=False):
        print("COLLECT DATA")
        self.state_data.num_self_play = self.tmp_num_selfplay
        if allocate and self.current_epoch > 0:
            self.state_data.allocate_self_play(self.tmp_num_selfplay)

        controller = NetworkWrapper(self.net, self.state_data, **self.config)
        evaluator = QuadEvaluator(controller, self.eval_env, **self.config)

        if random:
            print("USE RANDOM")
            evaluator.use_random_actions = True
        if use_mpc:
            print("USE MPC")
            from neural_control.controllers.mpc import MPC
            evaluator.use_mpc = MPC(horizon=10, dt=.1, dynamics="flightmare")
        prev_eval_counter = self.state_data.eval_counter
        with torch.no_grad():
            while self.state_data.eval_counter < self.config[
                "self_play"] + prev_eval_counter:
                evaluator.run_eval("rand", nr_test=5, **self.config)


def train_control(base_model, config):
    """
    Train a controller from scratch or with an initial model
    """
    modified_params = config["modified_params"]
    print(modified_params)
    train_dynamics = FlightmareDynamics(modified_params=modified_params)
    eval_dynamics = FlightmareDynamics(modified_params=modified_params)

    # make sure that also the self play samples are collected in same env
    config["sample_in"] = "train_env"
    config["train_dyn_for_epochs"] = -1

    trainer = TrainDrone(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    trainer.run_control(config)


def train_dynamics(base_model, config, trainable_params=1):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
    config["sample_in"] = "train_env"
    # config["sample_in"] = "eval_env"

    config["thresh_div_start"] = 1
    config["thresh_div_end"] = 3
    config["thresh_stable_start"] = 2
    config["l2_lambda"] = 0
    # return the divergence, not the stable steps
    config["return_div"] = 1
    config["suc_up_down"] = -1

    config["epoch_size"] = 500
    config["train_dyn_for_epochs"] = 10
    config["nr_test_data"] = 200
    # make sure not to resample during dynamics training
    config["resample_every"] = config["train_dyn_for_epochs"] + 1

    # config["epoch_size"] = 1
    # config["self_play"] = 200
    # config["min_epochs"] = 5
    # config["eval_var_dyn"] = "mean_trained_delta"
    # config["eval_var_con"] = "mean_div"
    # config["min_epochs"] = 5
    # config["suc_up_down"] = -1
    # config["return_div"] = 1

    # train environment is learnt
    train_dynamics = LearntQuadDynamics(trainable_params=trainable_params)
    eval_dynamics = FlightmareDynamics(modified_params)

    trainer = TrainDrone(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    # RUN
    trainer.run_dynamics(config)
    # trainer.run_iterative(config)


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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--todo",
        type=str,
        default="pretrain",
        help="what to do - pretrain, adapt or finetune"
    )
    parser.add_argument(
        "-m",
        "--model_load",
        type=str,
        default="trained_models/quad/current_model",
        help="Model to start with (default: None - from scratch)"
    )
    parser.add_argument(
        "-s",
        "--save_name",
        type=str,
        default="test",
        help="Name under which the trained model shall be saved"
    )
    parser.add_argument(
        "-p",
        "--params_trainable",
        type=bool,
        default=False,
        help="Train the parameters of \hat{f} (1) or only residual (0)"
    )
    args = parser.parse_args()

    baseline_model = args.model_load
    trainable_params = args.params_trainable
    todo = args.todo
    config["save_name"] = args.save_name

    if todo == "pretrain":
        # No baseline model used
        baseline_model = None
        train_control(baseline_model, config)
    elif todo == "adapt":
        # For finetune dynamics
        mod_params = {'translational_drag': np.array([0.3, 0.3, 0.3])}
        config["modified_params"] = mod_params
        # Define whether the parameters are trainable
        trainable_params = args.params_trainable
        print(
            f"start from pretrained model {args.model_load}, consider scenario\
                {mod_params}, train also parameters - {trainable_params}\
                save adapted dynamics and controller at {args.save_name}"
        )
        # Run
        train_dynamics(baseline_model, config, trainable_params)
    elif todo == "finetune":
        config["thresh_div_start"] = 1
        config["thresh_stable_start"] = 1.5
        train_control(baseline_model, config)
