import os
import numpy as np
import json
import torch
import torch.optim as optim

from train_base import TrainBase
from neural_control.dataset import CartpoleDataset, CartpoleImageDataset
from neural_control.drone_loss import (
    cartpole_loss_balance, cartpole_loss_swingup, cartpole_loss_mpc
)
from evaluate_cartpole import Evaluator
from neural_control.models.simple_model import (
    Net, ImageControllerNet, ImageControllerNetDQN
)
from neural_control.plotting import plot_loss, plot_success
from neural_control.environments.cartpole_env import (
    construct_states, CartPoleEnv
)
from neural_control.controllers.network_wrapper import (
    CartpoleWrapper, CartpoleImageWrapper
)
from neural_control.dynamics.cartpole_dynamics import (
    CartpoleDynamics, LearntCartpoleDynamics, ImageCartpoleDynamics
)


class TrainCartpole(TrainBase):
    """
    Train a controller for a quadrotor
    """

    def __init__(
        self,
        train_dynamics,
        eval_dynamics,
        config,
        train_image_dyn=0,
        swingup=0
    ):
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

        # for image processing
        self.train_image_dyn = train_image_dyn

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
                self.net = Net(
                    self.state_size, self.nr_actions * self.action_dim
                )
        if self.train_image_dyn:
            if base_model is None:
                self.net = ImageControllerNetDQN(
                    100,
                    60,
                    out_size=self.nr_actions * self.action_dim,
                    nr_img=self.config["nr_img"]
                )
            self.state_data = CartpoleImageDataset(
                load_data_path="data/cartpole_img_12.npz", **self.config
            )
        else:
            self.state_data = CartpoleDataset(
                num_states=self.config["sample_data"], **self.config
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
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(name + ".grad", param.grad)
                self.writer.add_histogram(name, param)
        self.optimizer_controller.step()
        return loss

    def run_image_epoch(self, train="dynamics"):
        """
        Overwrite function in order to include images
        """
        running_loss = 0
        for i, data in enumerate(self.trainloader, 0):
            # get state and action and correspodning image sequence
            current_state, actions, image_seq, next_state_d2 = data
            # the first image is the next ground truth!
            images = image_seq[:, 1:]
            gt_next_img = images[:, 0]
            mask = gt_next_img.greater(0)
            mask_inverse = gt_next_img.le(0.01)

            # zero the parameter gradients
            if train == "dynamics":
                self.optimizer_dynamics.zero_grad()
                next_state_d1, pred_next_img = self.train_dynamics(
                    current_state, images, actions, dt=self.delta_t
                )
                if i == 0:
                    print("\nExample dynamics")
                    next_state_eval_dyn = self.eval_dynamics(
                        current_state, actions, self.delta_t
                    )
                    print("start at ", current_state[0])
                    print("pred", next_state_d1[0].detach())
                    print("gt", next_state_d2[0])
                    print("next without modify", next_state_eval_dyn[0])
                    print()
                    print(np.max(pred_next_img.detach().numpy()))
                    print(np.min(pred_next_img.detach().numpy()))
                    # import matplotlib.pyplot as plt
                    # pred_example = pred_next_img[0].detach().numpy()
                    # gt_example = gt_next_img[0].detach().numpy()
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(gt_example)
                    # plt.title("GT")
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(pred_example)
                    # plt.title("Pred")
                    # plt.colorbar()
                    # plt.show()
                    print()

                loss_state = torch.sum((next_state_d1 - next_state_d2)**2)
                loss_img = torch.sum(
                    torch.masked_select(gt_next_img, mask) -
                    torch.masked_select(pred_next_img, mask)
                )**2
                loss_zeros = torch.sum(
                    torch.masked_select(gt_next_img, mask_inverse) -
                    torch.masked_select(pred_next_img, mask_inverse)
                )**2
                loss = loss_state + loss_img * 1e-6 + loss_zeros * 1e-7
                loss.backward()
                for name, param in self.net.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(name + ".grad", param.grad)
                        self.writer.add_histogram(name, param)
                self.optimizer_dynamics.step()
                self.results_dict["loss_dyn_per_step"].append(loss.item())

            elif train == "controller":
                self.optimizer_controller.zero_grad()
                actions = self.net(images.float())
                action_seq = torch.reshape(
                    actions, (-1, self.nr_actions, self.action_dim)
                )
                ref_states = self.make_reference(current_state)

                intermediate_states = torch.zeros(
                    current_state.size()[0], self.nr_actions, self.state_size
                )
                for k in range(action_seq.size()[1]):
                    current_state = self.train_dynamics(
                        current_state,
                        # TODO: output next image as well and change sequence
                        # images,
                        action_seq[:, k],
                        dt=self.delta_t
                    )
                    intermediate_states[:, k] = current_state
                # loss = self.train_controller_model(current_state, action_seq)
                loss = cartpole_loss_mpc(intermediate_states, ref_states)
                loss.backward()
                self.optimizer_controller.step()

            running_loss += loss.item()

        epoch_loss = (running_loss / i)
        self.loss_logging(epoch_loss, train="image " + train)
        return epoch_loss

    def loss_logging(self, epoch_loss, train="controller"):
        self.results_dict["loss"].append(epoch_loss)
        print(f"Loss ({train}): {round(epoch_loss, 2)}")
        self.writer.add_scalar("Loss/train", epoch_loss)

    def run_epoch(self, train="controller"):
        self.results_dict["trained"].append(train)
        # training image dynamics
        if self.train_image_dyn:
            return self.run_image_epoch(train=train)

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
        self.loss_logging(epoch_loss, train=train)
        return epoch_loss

    def evaluate_model(self, epoch):

        if self.swingup:
            new_data = self.evaluate_swingup(epoch)
        else:
            new_data = self.evaluate_balance(epoch)

        # Renew dataset dynamically
        if (epoch + 1) % self.resample_every == 0:
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
        if isinstance(self.net, Net):
            controller_model = CartpoleWrapper(self.net, **self.config)
        else:
            controller_model = CartpoleImageWrapper(self.net, **self.config)
        # EVALUATION:
        # self.eval_env.thresh_div = self.config["thresh_div"]
        evaluator = Evaluator(controller_model, self.eval_env)
        # Start in upright position and see how long it is balaned
        success_mean, success_std, data = evaluator.evaluate_in_environment(
            nr_iters=1, render=self.train_image_dyn
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


def train_img_dynamics(
    base_model, config, not_trainable="all", base_image_dyn=None
):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["general"]["modified_params"]
    config["sample_in"] = "eval_env"
    config["general"]["resample_every"] = 1000
    config["train_dyn_for_epochs"] = 200
    config["train_dyn_every"] = 1

    # train environment is learnt
    train_dyn = ImageCartpoleDynamics(
        100, 60, nr_img=config["general"]["nr_img"], state_size=4
    )
    #  CartpoleDynamics()
    if base_image_dyn is not None:
        print("loading base dyn model from", base_image_dyn)
        train_dyn.load_state_dict(
            torch.load(os.path.join(base_image_dyn, "dynamics_model"))
        )
    eval_dyn = CartpoleDynamics()  # modified_params=modified_params)
    trainer = TrainCartpole(train_dyn, eval_dyn, config, train_image_dyn=1)
    trainer.initialize_model(base_model)
    trainer.run_dynamics(config)


def train_norm_dynamics(base_model, config, not_trainable="all"):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["general"]["modified_params"]
    config["sample_in"] = "train_env"
    config["train_dyn_for_epochs"] = 2
    config["thresh_div_start"] = 0.2
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
    baseline_dyn = None  # "trained_models/cartpole/train_dyn_img"
    config["general"]["save_name"] = "train_img_pred"

    mod_params = {"wind": .5}
    config["general"]["modified_params"] = mod_params

    # TRAIN
    # config["nr_epochs"] = 20
    # train_control(baseline_model, config)
    # train_norm_dynamics(baseline_model, config, not_trainable="all")
    train_img_dynamics(
        baseline_model,
        config,
        not_trainable="all",
        base_image_dyn=baseline_dyn
    )
