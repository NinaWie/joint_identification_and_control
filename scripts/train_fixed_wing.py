import os
import json
import time
import numpy as np
import argparse
import torch.optim as optim
import torch
import torch.nn.functional as F

from neural_control.dataset import WingDataset
from neural_control.drone_loss import fixed_wing_last_loss, fixed_wing_mpc_loss
from neural_control.dynamics.fixed_wing_dynamics import (
    FixedWingDynamics, LearntFixedWingDynamics
)
from neural_control.dynamics.learnt_dynamics import (
    LearntDynamics, LearntDynamicsMPC
)
from neural_control.environments.wing_env import SimpleWingEnv
from neural_control.models.hutter_model import Net
from evaluate_fixed_wing import FixedWingEvaluator
from neural_control.controllers.network_wrapper import FixedWingNetWrapper
from train_base import TrainBase

never_trainable = ["rho", "g", "residual_factor"]


class TrainFixedWing(TrainBase):
    """
    Train a controller for a quadrotor
    """

    def __init__(self, train_dynamics, eval_dynamics, config):
        """
        param sample_in: one of "train_env", "eval_env"
        """
        self.config = config
        super().__init__(train_dynamics, eval_dynamics, **config)

        # specify  self.sample_in to collect more data (exploration)
        if self.sample_in == "eval_env":
            self.eval_env = SimpleWingEnv(self.eval_dynamics, self.delta_t)
        elif self.sample_in == "train_env":
            self.eval_env = SimpleWingEnv(self.train_dynamics, self.delta_t)
        else:
            raise ValueError("sample in must be one of eval_env, train_env")

    def initialize_model(self, base_model=None, base_model_name="model_wing"):
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
                self.config["mean"] = previous_parameters["mean"]
                self.config["std"] = previous_parameters["std"]
        else:
            # +9 because adding 12 things but deleting position (3)
            self.net = Net(
                self.state_size - self.ref_dim,
                1,
                self.ref_dim,
                self.action_dim * self.nr_actions,
                conv=False
            )

        # init dataset
        self.state_data = WingDataset(self.epoch_size, **self.config)
        # update mean and std:
        self.config = self.state_data.get_means_stds(self.config)
        # add other parameters
        self.config["horizon"] = self.nr_actions
        self.config["ref_length"] = self.nr_actions
        self.config["thresh_div"] = self.thresh_div_start
        self.config["dt"] = self.delta_t
        self.config["take_every_x"] = self.self_play_every_x
        self.config["thresh_stable"] = self.thresh_stable_start

        with open(os.path.join(self.save_path, "config.json"), "w") as outfile:
            json.dump(self.config, outfile)

        self.init_optimizer()

    def train_controller_model(
        self, current_state, action_seq, in_ref_state, ref_states
    ):
        # zero the parameter gradients
        self.optimizer_controller.zero_grad()
        intermediate_states = torch.zeros(
            current_state.size()[0], self.nr_actions_rnn, self.state_size
        )
        for k in range(self.nr_actions_rnn):
            # extract action
            action = action_seq[:, k]
            current_state = self.train_dynamics(
                current_state, action, dt=self.delta_t_train
            )
            intermediate_states[:, k] = current_state

        loss = fixed_wing_mpc_loss(
            intermediate_states, ref_states, action_seq, printout=0
        )

        # Backprop
        loss.backward()
        # for name, param in self.net.named_parameters():
        #     if param.grad is not None:
        #         self.writer.add_histogram(name + ".grad", param.grad)
        self.optimizer_controller.step()
        return loss

    def train_controller_recurrent(
        self, current_state, action_seq, in_ref_state, ref_states
    ):
        target_pos = self._compute_target_pos(current_state, in_ref_state)
        # ------------ VERSION 2: recurrent -------------------
        for k in range(self.nr_actions_rnn):
            in_state, _, in_ref_state, _ = self.state_data.prepare_data(
                current_state, ref_states
            )
            # print(k, "current state", current_state[0, :3])
            # print(k, "in_state", in_state[0])
            # print(k, "in ref", in_ref_state[0])
            action = torch.sigmoid(self.net(in_state, in_ref_state))
            current_state = self.train_dynamics.simulate_fixed_wing(
                current_state, action, dt=self.delta_t
            )
        loss = fixed_wing_last_loss(
            current_state, target_pos, None, printout=0
        )
        # Backprop
        loss.backward()
        self.optimizer_controller.step()
        return loss

    def evaluate_model(self, epoch):
        # EVALUATE
        controller = FixedWingNetWrapper(
            self.net, self.state_data, **self.config
        )

        eval_dyn = self.train_dynamics if (
            isinstance(self.train_dynamics, LearntDynamics)
            or isinstance(self.train_dynamics, LearntDynamicsMPC)
        ) else None  # and self.config.get("is_seq", 0)  TODO
        evaluator = FixedWingEvaluator(
            controller, self.eval_env, eval_dyn=eval_dyn, **self.config
        )

        # previous version with resampling
        # if epoch % self.config["resample_every"] == 0:
        #     print("START COLLECT DATA")
        #     self.state_data.num_self_play = self.tmp_num_selfplay
        # if epoch == 0 or self.results_dict["train_dyn_con"][epoch
        #                                                     ] == "controller":
        #     self.state_data.num_self_play = self.tmp_num_selfplay
        #     self.results_dict["collected_data"].append(
        #         self.state_data.num_self_play
        #     )
        # else:
        #     self.results_dict["collected_data"].append(0)

        # switch off self play
        self.state_data.num_self_play = 0

        # run without mpc for evaluation
        # print("--------- eval in simulator (D1) -------------")
        # print(
        #     "check: self play?", self.state_data.num_self_play, "eval counter",
        #     self.state_data.eval_counter
        # )
        # run eval
        with torch.no_grad():
            res_eval = evaluator.run_eval(nr_test=20)
        for key, val in res_eval.items():
            self.results_dict[key].append(val)
        suc_mean = res_eval["mean_div_linear"]
        suc_std = res_eval["std_div_linear"]

        # FOR DYN TRAINING
        # if epoch == 0 and self.config["train_dyn_for_epochs"] >= 0:
        #     self.tmp_num_selfplay = self.state_data.num_self_play
        #     self.state_data.num_self_play = 0
        #     print("stop self play")
        # if epoch == self.config["train_dyn_for_epochs"]:
        #     self.state_data.num_self_play = self.tmp_num_selfplay
        #     print("start self play to", self.tmp_num_selfplay)

        # if training dynamics: evaluate in target dynamics
        # if self.config["train_dyn_for_epochs"] >= 0:
        #     tmp_self_play = self.state_data.num_self_play
        #     self.state_data.num_self_play = 0
        #     print("--------- eval in real (D2) -------------")
        #     d2_env = SimpleWingEnv(self.eval_dynamics, self.delta_t)
        #     evaluator = FixedWingEvaluator(controller, d2_env, **self.config)
        #     with torch.no_grad():
        #         suc_mean, suc_std = evaluator.run_eval(nr_test=10)
        #     self.results_dict["eval_in_d2_mean"].append(suc_mean)
        #     self.results_dict["eval_in_d2_std"].append(suc_std)
        #     self.state_data.num_self_play = tmp_self_play
        #     print("eval samples counter", self.state_data.eval_counter)

        # self.sample_new_data(epoch)

        # increase thresholds
        # if self.config["thresh_div"] < self.thresh_div_end:
        #     self.config["thresh_div"] += .5
        #     print("increased thresh div", self.config["thresh_div"])

        # if self.config["thresh_stable"] < self.thresh_stable_end:
        #     self.config["thresh_stable"] += .05
        #     print("increased thresh stable", self.config["thresh_stable"])

        # save best model
        self.save_model(epoch, suc_mean, suc_std)

        self.results_dict["thresh_div"].append(self.config["thresh_div"])
        return suc_mean, suc_std

    def collect_data(self, random=False, allocate=False, mpc=False):
        print("COLLECT DATA")
        # switch on self play
        self.state_data.num_self_play = self.tmp_num_selfplay

        if allocate and self.current_epoch > 0:
            self.state_data.allocate_self_play(self.tmp_num_selfplay)
        else:
            self.state_data.num_sampled_states = 0

        controller = FixedWingNetWrapper(
            self.net, self.state_data, **self.config
        )
        evaluator = FixedWingEvaluator(
            controller, self.eval_env, **self.config
        )
        if random:
            print("USE RANDOM")
            evaluator.use_random_actions = True
        if mpc:
            print("USE MPC")
            from neural_control.controllers.mpc import MPC
            evaluator.use_mpc = MPC(
                horizon=20, dt=0.1, dynamics="fixed_wing_3D"
            )
        prev_eval_counter = self.state_data.eval_counter
        while self.state_data.eval_counter < self.config["self_play"
                                                         ] + prev_eval_counter:
            with torch.no_grad():
                _ = evaluator.run_eval(nr_test=5)
        print(
            len(self.state_data.states), self.state_data.get_eval_index(),
            self.state_data.eval_counter, self.state_data.num_sampled_states
        )


def train_control(base_model, config):
    """
    Train a controller from scratch or with an initial model
    """
    config["train_dyn_for_epochs"] = -1
    modified_params = config["modified_params"]
    train_dynamics = FixedWingDynamics(modified_params)
    eval_dynamics = FixedWingDynamics(modified_params)

    # make sure that also the self play samples are collected in same env
    config["sample_in"] = "train_env"

    trainer = TrainFixedWing(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model)

    trainer.run_control(config, curriculum=0)


def train_sampling_finetune(base_model, config):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
    config["sample_in"] = "eval_env"

    # train environment is learnt
    train_dynamics = FixedWingDynamics()
    eval_dynamics = FixedWingDynamics(modified_params=modified_params)

    trainer = TrainFixedWing(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model)

    # RUN
    trainer.run_control(config, sampling_based_finetune=True)


def train_dynamics(base_model, config, not_trainable, base_dyn=None):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
    config["sample_in"] = "train_env"

    # DYNAMIC TRAINING
    # set high thresholds because not training from scratch
    config["thresh_div_start"] = 20
    config["thresh_stable_start"] = 1.5
    # set self play to zero to avoid bad actions
    # config["self_play"] = 1000
    # config["epoch_size"] = 1
    config["train_dyn_for_epochs"] = 20
    config["resample_every"] = config["train_dyn_for_epochs"] + 1
    # config["learning_rate_controller"] = 1e-4
    # lambda: how much delta network is penalized
    config["l2_lambda"] = 0
    config["waypoint_metric"] = True

    # train environment is learnt
    train_dynamics = LearntFixedWingDynamics(not_trainable=not_trainable)
    if base_dyn is not None:
        train_dynamics.load_state_dict(torch.load(base_dyn))
        print("loaded dynamics from", base_dyn)
    eval_dynamics = FixedWingDynamics(modified_params=modified_params)

    trainer = TrainFixedWing(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model)

    # RUN
    trainer.run_dynamics(config)


if __name__ == "__main__":
    # LOAD CONFIG
    with open("configs/wing_config.json", "r") as infile:
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
        default="trained_models/wing/current_model",
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
        train_control(None, config)
    elif todo == "adapt":
        # For finetune dynamics
        # set high thresholds because not training from scratch
        # config["thresh_div_start"] = 20
        # config["thresh_stable_start"] = 1.5
        mod_params = {"vel_drag_factor": 0.3}
        config["modified_params"] = mod_params
        trainable_params = never_trainable + [
            "vel_drag_factor"
        ] if args.params_trainable else "all"
        print(
            f"start from pretrained model {args.model_load}, consider scenario\
                {mod_params}, train also parameters - {trainable_params}\
                save adapted dynamics and controller at {args.save_name}"
        )
        # Run
        train_dynamics(baseline_model, config, not_trainable=trainable_params)
    elif todo == "finetune":
        config["thresh_div_start"] = 20
        config["thresh_stable_start"] = 1.5
        train_control(baseline_model, config)
