import os
import numpy as np
import json
import torch
import torch.optim as optim

from train_base import TrainBase
from neural_control.dataset import (CartpoleImageDataset)
from neural_control.drone_loss import cartpole_loss_mpc
from evaluate_cartpole import Evaluator
from neural_control.models.simple_model import (
    ImageControllerNetDQN, StateToImg
)
from neural_control.environments.cartpole_env import (CartPoleEnv)
from neural_control.controllers.network_wrapper import (CartpoleImageWrapper)
from neural_control.dynamics.cartpole_dynamics import (
    CartpoleDynamics, ImageCartpoleDynamics
)


class TrainImageCartpole(TrainBase):
    """
    Train a controller for a quadrotor
    """

    def __init__(
        self,
        train_dynamics,
        eval_dynamics,
        config,
        train_image_dyn=0,
        train_seq_dyn=0,
        swingup=0
    ):
        """
        param sample_in: one of "train_env", "eval_env"
        """
        self.swingup = swingup
        # part_cfg = config["swingup"] if swingup else config["balance"]
        self.config = config  # {**config, **config}
        super().__init__(train_dynamics, eval_dynamics, **self.config)
        if self.sample_in == "eval_env":
            self.eval_env = CartPoleEnv(self.eval_dynamics, self.delta_t)
        elif self.sample_in == "train_env":
            self.eval_env = CartPoleEnv(self.train_dynamics, self.delta_t)
        else:
            raise ValueError("sample in must be one of eval_env, train_env")

        # state to image transformer:
        self.state_to_img_net = torch.load(
            "trained_models/cartpole/state_img_net"
        )
        self.state_to_img_optim = optim.SGD(
            self.state_to_img_net.parameters(), lr=0.0001, momentum=0.9
        )

    def initialize_model(
        self,
        base_model=None,
        base_model_name="model_cartpole",
        load_dataset="data/cartpole_img_20.npz",
        load_state_to_img=None
    ):
        if base_model is not None:
            self.net = torch.load(os.path.join(base_model, base_model_name))
        else:
            # image dynamics
            self.net = ImageControllerNetDQN(
                100,
                120,
                out_size=self.nr_actions * self.action_dim,
                nr_img=self.config["nr_img"]
            )
            # # for sanity check
            # self.net = Net(
            #     self.state_size, self.nr_actions * self.action_dim
            # )
        self.state_data = CartpoleImageDataset(
            load_data_path=load_dataset, **self.config
        )
        self.model_wrapped = CartpoleImageWrapper(
            self.net, self.state_data, **self.config
        )

        with open(os.path.join(self.save_path, "config.json"), "w") as outfile:
            json.dump(self.config, outfile)

        # load state to img netwrok if it was finetuned
        if load_state_to_img is not None:
            self.state_to_img_net.load_state_dict(
                torch.load(os.path.join(load_state_to_img, "state_to_img"))
            )

        self.init_optimizer()
        self.config["thresh_div"] = self.config["thresh_div_start"]

    def make_reference(self, current_state):
        ref_states = torch.zeros(
            current_state.size()[0], self.nr_actions, self.state_size
        )
        for k in range(self.nr_actions - 1):
            ref_states[:, k] = (
                current_state * (1 - 1 / (self.nr_actions - 1) * k)
            )
        return ref_states

    def run_epoch(self, train="dynamics"):
        """
        Overwrite function in order to include images
        """
        running_loss = 0
        for i, data in enumerate(self.trainloader, 0):
            # get state and action and correspodning image sequence
            initial_state_buffer, actions, image_seq = data

            # state buffer has most next state in pos 0
            next_state_d2 = initial_state_buffer[:, 0]
            initial_state = initial_state_buffer[:, 1]
            # TODO: state sequence would be [:, 1:]
            current_state = initial_state.clone()
            # the first image is the next ground truth!
            images = image_seq[:, 1:]

            # zero the parameter gradients
            if train == "dynamics":
                # Render the states to images
                state_seq = initial_state_buffer[:, 1:]
                # extract x and theta
                render_inp = torch.reshape(state_seq[:, :, [0, 2]], (-1, 2))

                rendered_img = self.state_to_img_net(render_inp.float())
                images = torch.reshape(
                    rendered_img, (-1, state_seq.size()[1], 100, 120)
                )

                # Finetune the state-to-img network
                # self.state_to_img_optim.zero_grad()
                # inp_img_loss = torch.sum((images - image_seq[:, 1:])**2)
                # inp_img_loss.backward()
                # self.state_to_img_optim.step()

                use_inp_images = images.detach()

                # train the dynamics network
                self.optimizer_dynamics.zero_grad()
                next_state_pred = self.train_dynamics(
                    current_state,
                    use_inp_images,
                    torch.unsqueeze(actions[:, 0].float(), 1),
                    dt=self.delta_t
                )
                if i == 0:
                    print("\nExample dynamics")
                    next_state_eval_dyn = self.eval_dynamics(
                        current_state,
                        torch.unsqueeze(actions[:, 0].float(), 1), self.delta_t
                    )
                    print("start at ", current_state[0])
                    print("pred", next_state_pred[0].detach())
                    print("gt", next_state_d2[0])
                    print("next without modify", next_state_eval_dyn[0])
                    print()

                loss = torch.sum((next_state_pred - next_state_d2)**2)
                loss.backward()

                loss *= 1000
                # for name, param in self.net.named_parameters():
                #     if param.grad is not None:
                #         self.writer.add_histogram(name + ".grad", param.grad)
                #         self.writer.add_histogram(name, param)
                self.optimizer_dynamics.step()
                self.results_dict["loss_dyn_per_step"].append(loss.item())

            elif train == "controller":
                self.optimizer_controller.zero_grad()
                actions = self.net(images.float())  # current_state.float())
                action_seq = torch.reshape(
                    actions, (-1, self.nr_actions, self.action_dim)
                )
                ref_states = self.make_reference(current_state)

                intermediate_states = torch.zeros(
                    current_state.size()[0], self.nr_actions, self.state_size
                )
                # eval_dyn_state = current_state.clone()
                for k in range(action_seq.size()[1]):
                    # # Check how much our image residual diverges from desired
                    # eval_dyn_state = self.eval_dynamics(
                    #     eval_dyn_state, action_seq[:, k], dt=self.delta_t
                    # )
                    # image dynamics
                    current_state = self.train_dynamics(
                        current_state,
                        images.float(),
                        torch.unsqueeze(actions[:, k].float(), 1),
                        dt=self.delta_t
                    )
                    intermediate_states[:, k] = current_state
                    # render img
                    x_diff = torch.unsqueeze(
                        current_state[:, 0] - initial_state[:, 0], 1
                    )
                    theta = torch.unsqueeze(current_state[:, 2], 1)
                    render_input = torch.cat((x_diff, theta), dim=1)
                    render_current_state = torch.unsqueeze(
                        self.state_to_img_net(render_input.float()), dim=1
                    )
                    # add to image sequence
                    images = torch.cat(
                        (render_current_state, images[:, :-1]), dim=1
                    )
                # if i == 0:
                #     print("compare img dyn to gt dyn")
                #     print(intermediate_states[0])
                #     print(eval_dyn_state[0])

                # LOSS
                loss = cartpole_loss_mpc(
                    intermediate_states, ref_states, action_seq
                )
                loss.backward()
                self.optimizer_controller.step()

            running_loss += loss.item()

        epoch_loss = (running_loss / i)
        self.loss_logging(epoch_loss, train=train)
        return epoch_loss

    def loss_logging(self, epoch_loss, train="controller"):
        self.results_dict["loss_" + train].append(epoch_loss)
        print(f"Loss ({train}): {round(epoch_loss, 2)}")
        # self.writer.add_scalar("Loss/train", epoch_loss)

    def evaluate_model(self, epoch):

        _ = self.evaluate_balance(epoch)

    def evaluate_balance(self, epoch):
        # EVALUATION:
        # self.eval_env.thresh_div = self.config["thresh_div"]
        evaluator = Evaluator(self.model_wrapped, self.eval_env)
        # Start in upright position and see how long it is balaned
        res_eval = evaluator.evaluate_in_environment(nr_iters=10, render=1)
        success_mean = res_eval["mean_vel"]
        success_std = res_eval["std_vel"]
        for key, val in res_eval.items():
            self.results_dict[key].append(val)
        self.save_model(epoch, success_mean, success_std)
        return None

    def finalize(self):
        torch.save(
            self.state_to_img_net.state_dict(),
            os.path.join(self.save_path, "state_img_net")
        )
        super().finalize(plot_loss="loss_dynamics")


def train_img_dynamics(
    base_model, config, not_trainable="all", base_image_dyn=None
):
    modified_params = config["modified_params"]
    config["sample_in"] = "eval_env"
    config["resample_every"] = 1000
    config["train_dyn_for_epochs"] = 200
    config["train_dyn_every"] = 1
    # No self play!
    config["self_play"] = 0

    # train environment is learnt
    # train_dyn = CartpoleDynamics(modified_params=modified_params)
    train_dyn = ImageCartpoleDynamics(
        100, 120, nr_img=config["nr_img"], state_size=4
    )
    # load pretrained dynamics
    if base_image_dyn is not None:
        print("loading base dyn model from", base_image_dyn)
        train_dyn.load_state_dict(
            torch.load(os.path.join(base_image_dyn, "dynamics_model"))
        )
    eval_dyn = CartpoleDynamics(modified_params=modified_params)
    trainer = TrainImageCartpole(
        train_dyn, eval_dyn, config, train_image_dyn=1
    )
    trainer.initialize_model(
        base_model, load_dataset="data/cartpole_img_1000_notcenter.npz"
    )
    trainer.run_dynamics(config)


def train_img_controller(
    base_model, config, not_trainable="all", base_image_dyn=None
):
    """
    Train controller with image dynamics
    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = {"contact": 1}
    # modified_params = config["modified_params"]
    # Only collect experience in the trained dynamics, not the ground truth
    config["sample_in"] = "eval_env"
    config["resample_every"] = 1000
    config["train_dyn_for_epochs"] = -1
    config["train_dyn_every"] = 1
    config["suc_up_down"] = -1

    # train environment is learnt (next line only for sanity check)
    # train_dyn = CartpoleDynamics(modified_params=modified_params)
    train_dyn = ImageCartpoleDynamics(
        100, 120, nr_img=config["nr_img"], state_size=4
    )
    # Load finetuned image dynamics
    if base_image_dyn is not None:
        print("loading base dyn model from", base_image_dyn)
        train_dyn.load_state_dict(
            torch.load(os.path.join(base_image_dyn, "dynamics_model"))
        )
    eval_dyn = CartpoleDynamics(modified_params=modified_params)
    trainer = TrainImageCartpole(
        train_dyn, eval_dyn, config, train_image_dyn=1
    )
    trainer.initialize_model(
        base_model,
        load_dataset="data/cartpole_img_1000_center.npz",
        load_state_to_img=None
    )
    trainer.run_dynamics(config)


if __name__ == "__main__":
    # LOAD CONFIG - select balance or swigup
    with open("configs/cartpole_config.json", "r") as infile:
        config = json.load(infile)

    # FINETUNE DYNAMICS (TRAIN RESIDUAL)
    # baseline_model = None
    # baseline_dyn = None
    # config["save_name"] = "dyn_img_contact"
    # train_img_dynamics(
    #     None, config, not_trainable="all", base_image_dyn=baseline_dyn
    # )

    # TRAIN CONTROLLER AFTER DYNAMICS ARE FINETUNED
    baseline_model = None
    baseline_dyn = "trained_models/cartpole/dyn_img_contact"
    config["save_name"] = "con_img_contact"

    train_img_controller(
        baseline_model,
        config,
        not_trainable="all",
        base_image_dyn=baseline_dyn
    )
