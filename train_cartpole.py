import os
import numpy as np
import json
import torch
import torch.optim as optim

from train_base import TrainBase
from neural_control.dataset import CartpoleDataset
from neural_control.drone_loss import (
    cartpole_loss_balance, cartpole_loss_swingup, cartpole_loss_mpc
)
from evaluate_cartpole import Evaluator
from neural_control.models.resnet_like_model import Net as SimpleResNet
from neural_control.models.simple_model import Net as SimpleNet
from neural_control.plotting import plot_loss, plot_success
from neural_control.environments.cartpole_env import (
    construct_states, CartPoleEnv
)
from neural_control.controllers.network_wrapper import CartpoleWrapper
from neural_control.dynamics.cartpole_dynamics import (
    CartpoleDynamics, LearntCartpoleDynamics
)


class TrainCartpole(TrainBase):
    """
    Train a controller for a quadrotor
    """

    def __init__(self, train_dynamics, eval_dynamics, config, swingup=0):
        """
        param sample_in: one of "train_env", "eval_env"
        """
        self.swingup = swingup
        part_cfg = config["swingup"] if swingup else config["balance"]
        self.config = {**config["general"], **part_cfg}
        super().__init__(train_dynamics, eval_dynamics, **self.config)
        if self.sample_in == "eval_env":
            self.eval_env = CartPoleEnv(self.eval_dynamics, self.delta_t)
        elif self.sample_in == "train_env":
            self.eval_env = CartPoleEnv(self.train_dynamics, self.delta_t)
        else:
            raise ValueError("sample in must be one of eval_env, train_env")

    def initialize_model(
        self, base_model=None, base_model_name="model_cartpole"
    ):
        if base_model is not None:
            self.net = torch.load(os.path.join(base_model, base_model_name))
        else:
            if self.swingup:
                self.net = SimpleResNet(
                    self.state_size, self.nr_actions * self.action_dim
                )
            else:
                self.net = SimpleNet(
                    self.state_size, self.nr_actions * self.action_dim
                )
        self.state_data = CartpoleDataset(
            num_states=self.config["sample_data"]
        )
        with open(os.path.join(self.save_path, "config.json"), "w") as outfile:
            json.dump(self.config, outfile)

        self.init_optimizer()
        self.config["thresh_div"] = self.config["thresh_div_start"]

    def make_reference(self, current_state):
        ref_states = torch.zeros(
            current_state.size()[0], self.nr_actions, self.state_size
        )
        for k in range(self.nr_actions - 1):
            ref_states[:, k] = (
                current_state * (1 - 1 / self.nr_actions * (k + 1))
            )
        return ref_states

    def train_controller_model(self, current_state, action):
        # zero the parameter gradients
        self.optimizer_controller.zero_grad()
        ref_states = self.make_reference(current_state)

        intermediate_states = torch.zeros(
            current_state.size()[0], self.nr_actions, self.state_size
        )
        for k in range(action.size()[1]):
            current_state = self.train_dynamics(
                current_state, action[:, k], dt=self.delta_t
            )
            intermediate_states[:, k] = current_state
        # Loss
        if self.swingup:
            loss = cartpole_loss_swingup(current_state)
        else:
            loss = cartpole_loss_mpc(intermediate_states, ref_states)

        loss.backward()
        self.optimizer_controller.step()
        return loss

    def run_epoch(self, train="controller"):
        # tic_epoch = time.time()
        running_loss = 0
        for i, data in enumerate(self.trainloader, 0):
            # inputs are normalized states, current state is unnormalized in
            # order to correctly apply the action
            in_state, current_state = data

            actions = self.net(in_state)
            action_seq = torch.reshape(
                actions, (-1, self.nr_actions, self.action_dim)
            )

            if train == "controller":
                loss = self.train_controller_model(current_state, action_seq)
            else:
                # should work for both recurrent and normal
                loss = self.train_dynamics_model(current_state, action_seq)
                self.count_finetune_data += len(current_state)

            running_loss += loss.item()
        # time_epoch = time.time() - tic
        epoch_loss = running_loss / i
        self.results_dict["loss"].append(epoch_loss)
        self.results_dict["trained"].append(train)
        print(f"Loss ({train}): {round(epoch_loss, 2)}")
        self.writer.add_scalar("Loss/train", epoch_loss)
        return epoch_loss

    def evaluate_model(self, epoch):

        if self.swingup:
            new_data = self.evaluate_swingup(epoch)
        else:
            new_data = self.evaluate_balance(epoch)

        # Renew dataset dynamically
        if epoch % self.resample_every == 0:
            self.state_data.resample_data(
                num_states=self.config["sample_data"],
                thresh_div=self.config["thresh_div"]
            )
            if self.config["use_new_data"] > 0 and epoch > 0:
                # add the data generated during evaluation
                rand_inds_include = np.random.permutation(
                    len(new_data)
                )[:self.config["use_new_data"]]
                self.state_data.add_data(np.array(new_data)[rand_inds_include])
            # self.trainloader = torch.utils.data.DataLoader(
            #     self.state_data, batch_size=8, shuffle=True, num_workers=0
            # )
            print(
                f"\nsampled new data {len(self.state_data)},\
                    thresh: {round(self.config['thresh_div'], 2)}"
            )

        # increase thresholds
        if epoch % 3 == 0 and self.config["thresh_div"] < self.thresh_div_end:
            self.config["thresh_div"] += self.config["thresh_div_step"]

    def evaluate_balance(self, epoch):
        controller = CartpoleWrapper(self.net, **self.config)
        # EVALUATION:
        self.eval_env.thresh_div = self.config["thresh_div"]
        evaluator = Evaluator(controller, self.eval_env)
        # Start in upright position and see how long it is balaned
        success_mean, success_std, data = evaluator.evaluate_in_environment(
            nr_iters=10
        )
        self.save_model(epoch, success_mean, success_std)
        return data

    def evaluate_swingup(self, epoch):
        evaluator = Evaluator(self.eval_env)
        success_mean, success_std, _ = evaluator.evaluate_in_environment(
            self.net, nr_iters=10
        )
        swing_up_mean, swing_up_std, new_data = evaluator.make_swingup(
            self.net, nr_iters=10
        )
        print(
            "Average episode length: ", success_mean, "std:", success_std,
            "swing up:", swing_up_mean, "std:", swing_up_std
        )
        if swing_up_mean[0] < .5 and swing_up_mean[2] < .5 and np.sum(
            swing_up_mean
        ) < 3 and np.sum(swing_up_std) < 1 and success_mean > 180:
            print("early stopping")
        # TODO: save model when swingup performance metric is sufficient
        performance_swingup = swing_up_mean[0] + swing_up_mean[
            2] + (251 - success_mean) * 0.01

        self.save_model(epoch, swing_up_mean, swing_up_std)
        return new_data


def train_control(base_model, config, swingup=0):
    """
    Train a controller from scratch or with an initial model
    """
    modified_params = config["general"]["modified_params"]
    train_dynamics = CartpoleDynamics(modified_params)
    eval_dynamics = CartpoleDynamics(modified_params, test_time=1)

    trainer = TrainCartpole(
        train_dynamics, eval_dynamics, config, swingup=swingup
    )
    trainer.initialize_model(base_model)
    try:
        for epoch in range(trainer.config["nr_epochs"]):
            trainer.evaluate_model(epoch)
            print()
            print("Epoch", epoch)
            trainer.run_epoch(train="controller")
    except KeyboardInterrupt:
        pass
    trainer.finalize()


def train_dynamics(base_model, config, not_trainable):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["general"]["modified_params"]
    config["sample_in"] = "train_env"
    config["train_dyn_for_epochs"] = 4
    config["train_dyn_every"] = 1

    # train environment is learnt
    train_dyn = LearntCartpoleDynamics(not_trainable=not_trainable)
    eval_dyn = CartpoleDynamics(modified_params=modified_params)
    trainer = TrainCartpole(train_dyn, eval_dyn, config)

    trainer.initialize_model(base_model)

    # RUN
    trainer.run_dynamics(config)


if __name__ == "__main__":
    # LOAD CONFIG - select balance or swigup
    with open("configs/cartpole_config.json", "r") as infile:
        config = json.load(infile)

    baseline_model = None  # "trained_models/cartpole/current_model"
    config["general"]["save_name"] = "train_from_scratch"

    # mod_params = {"wind": .5}
    # config["general"]["modified_params"] = mod_params

    # TRAIN
    # config["nr_epochs"] = 20
    train_control(baseline_model, config)
    # train_dynamics(baseline_model, config, not_trainable="all")
