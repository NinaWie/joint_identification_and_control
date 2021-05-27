import os
import numpy as np
import json
import torch
import torch.optim as optim

from train_base import TrainBase
from neural_control.dataset import (
    CartpoleDataset, CartpoleImageDataset, CartpoleSequenceDataset
)
from neural_control.drone_loss import (
    cartpole_loss_balance, cartpole_loss_swingup, cartpole_loss_mpc
)
from evaluate_cartpole import Evaluator
from neural_control.models.simple_model import (
    Net, ImageControllerNet, ImageControllerNetDQN, StateToImg
)
from neural_control.plotting import plot_loss, plot_success
from neural_control.environments.cartpole_env import (
    construct_states, CartPoleEnv
)
from neural_control.controllers.network_wrapper import (
    CartpoleWrapper, CartpoleImageWrapper, SequenceCartpoleWrapper
)
from neural_control.dynamics.cartpole_dynamics import (
    CartpoleDynamics, LearntCartpoleDynamics, ImageCartpoleDynamics,
    SequenceCartpoleDynamics
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
        train_seq_dyn=0,
        swingup=0
    ):
        """
        param sample_in: one of "train_env", "eval_env"
        """
        self.swingup = swingup
        self.config = config
        super().__init__(train_dynamics, eval_dynamics, **self.config)
        if self.sample_in == "eval_env":
            self.eval_env = CartPoleEnv(self.eval_dynamics, self.delta_t)
        elif self.sample_in == "train_env":
            self.eval_env = CartPoleEnv(self.train_dynamics, self.delta_t)
        else:
            raise ValueError("sample in must be one of eval_env, train_env")

        # for image processing
        self.train_image_dyn = train_image_dyn
        self.train_seq_dyn = train_seq_dyn

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

        # image dynamics
        if self.train_image_dyn:
            if base_model is None:
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
        # sequnce of states
        elif self.train_seq_dyn:
            if base_model is None:
                self.net = Net(
                    (self.state_size + self.action_dim) * 3,
                    self.nr_actions * self.action_dim
                )
            self.state_data = CartpoleSequenceDataset(
                load_data_path=load_dataset, **self.config
            )
            self.model_wrapped = SequenceCartpoleWrapper(
                self.net, self.state_data, **self.config
            )
        # normal situation
        else:
            if self.swingup and base_model is None:
                self.net = SimpleResNet(
                    self.state_size, self.nr_actions * self.action_dim
                )
            elif base_model is None:
                self.net = Net(
                    self.state_size, self.nr_actions * self.action_dim
                )
            self.state_data = CartpoleDataset(
                num_states=self.config["sample_data"], **self.config
            )
            self.model_wrapped = CartpoleWrapper(self.net, **self.config)

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

    def run_image_epoch(self, train="dynamics"):
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
                self.state_to_img_optim.zero_grad()
                inp_img_loss = torch.sum((images - image_seq[:, 1:])**2)
                inp_img_loss.backward()
                self.state_to_img_optim.step()

                use_inp_images = images.detach()

                # train the dynamics network
                self.optimizer_dynamics.zero_grad()
                next_state_pred = self.train_dynamics(
                    current_state,
                    use_inp_images,
                    torch.unsqueeze(actions, 1),
                    dt=self.delta_t
                )
                if i == 0:
                    print("\nExample dynamics")
                    next_state_eval_dyn = self.eval_dynamics(
                        current_state, actions, self.delta_t
                    )
                    print("start at ", current_state[0])
                    print("pred", next_state_pred[0].detach())
                    print("gt", next_state_d2[0])
                    print("next without modify", next_state_eval_dyn[0])
                    print()

                loss = torch.sum((next_state_pred - next_state_d2)**2)
                loss.backward()

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
                        action_seq[:, k],
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
        self.loss_logging(epoch_loss, train="image " + train)
        return epoch_loss

    def loss_logging(self, epoch_loss, train="controller"):
        self.results_dict["loss_" + train].append(epoch_loss)
        print(f"Loss ({train}): {round(epoch_loss, 2)}")
        # self.writer.add_scalar("Loss/train", epoch_loss)

    def run_sequence_epoch(self, train="controller"):
        self.results_dict["trained"].append(train)
        # training image dynamics
        if self.train_image_dyn:
            return self.run_image_epoch(train=train)

        # tic_epoch = time.time()
        running_loss = 0
        for i, data in enumerate(self.trainloader, 0):
            # don't use images, only state and action buffers
            state_buffer, action_buffer = data
            state_action_history = torch.cat(
                (state_buffer[:, 1:], action_buffer[:, 1:]), dim=2
            )

            network_input = torch.reshape(
                state_action_history, (
                    -1, state_action_history.size()[1] *
                    state_action_history.size()[2]
                )
            )
            current_state = state_buffer[:, 1]

            if train == "controller":
                self.optimizer_controller.zero_grad()
                ref_states = self.make_reference(current_state)
                actions = self.net(network_input.float())
                action_seq = torch.reshape(
                    actions, (-1, self.nr_actions, self.action_dim)
                )
                intermediate_states = torch.zeros(
                    current_state.size()[0], self.nr_actions, self.state_size
                )

                # eval_dyn_state = current_state.clone()
                for k in range(action_seq.size()[1]):
                    # eval_dyn_state = self.eval_dynamics(
                    #     eval_dyn_state, action_seq[:, k], dt=self.delta_t
                    # )
                    network_input = torch.reshape(
                        state_action_history, (
                            -1, state_action_history.size()[1] *
                            state_action_history.size()[2]
                        )
                    ).float()
                    current_state = self.train_dynamics(
                        current_state,
                        network_input.float(),
                        action_seq[:, k],
                        dt=self.delta_t
                    )
                    intermediate_states[:, k] = current_state
                    # roll history
                    state_action_cat = torch.unsqueeze(
                        torch.cat((current_state, action_seq[:, k]), dim=1),
                        dim=1
                    )
                    state_action_history = torch.cat(
                        (state_action_cat, state_action_history[:, :-1]),
                        dim=1
                    )
                # if i == 0:
                #     print("compare img dyn to gt dyn")
                #     print(intermediate_states[0])
                #     print(eval_dyn_state[0])
                # Loss
                loss = cartpole_loss_mpc(
                    intermediate_states, ref_states, action_seq
                )
                loss.backward()
                self.optimizer_controller.step()
            elif train == "dynamics":
                actions = action_buffer[:, 0].float()
                next_state_d2 = state_buffer[:, 0]
                # should work for both recurrent and normal
                self.optimizer_dynamics.zero_grad()
                next_state_pred = self.train_dynamics(
                    current_state.float(),
                    network_input.float(),
                    actions,
                    dt=self.delta_t
                )
                # Since we have periodic force here, this is not useful
                # if i == 0:
                # print("\nExample dynamics")
                # self.eval_dynamics.timestamp = 0.1
                # next_state_eval_contact = self.eval_dynamics(
                #     current_state, actions, self.delta_t
                # )
                # self.eval_dynamics.timestamp = -0.1
                # next_state_eval_noc = self.eval_dynamics(
                #     current_state, actions, self.delta_t
                # )
                # print("start at ", current_state[0])
                # print("pred", next_state_pred[0].detach())
                # print("gt", next_state_d2[0])
                #     print("next with contact", next_state_eval_contact[0])
                #     print("next ohne contact", next_state_eval_noc[0])
                #     print()

                loss = torch.sum((next_state_pred - next_state_d2)**2)
                loss.backward()
                self.optimizer_dynamics.step()
                # self.results_dict["loss_dyn_per_step"].append(loss.item())
                loss *= 1000

            running_loss += loss.item()
        # time_epoch = time.time() - tic
        epoch_loss = running_loss / i
        self.loss_logging(epoch_loss * 1000, train=train)
        return epoch_loss

    def run_normal_epoch(self, train="controller"):
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
                # zero the parameter gradients
                self.optimizer_controller.zero_grad()
                ref_states = self.make_reference(current_state)

                intermediate_states = torch.zeros(
                    current_state.size()[0], self.nr_actions, self.state_size
                )
                for k in range(action_seq.size()[1]):
                    current_state = self.train_dynamics(
                        current_state, action_seq[:, k], dt=self.delta_t
                    )
                    intermediate_states[:, k] = current_state
                # Loss
                if self.swingup:
                    loss = cartpole_loss_swingup(current_state)
                else:
                    loss = cartpole_loss_mpc(intermediate_states, ref_states)

                loss.backward()
                # for name, param in self.net.named_parameters():
                #     if param.grad is not None:
                #         self.writer.add_histogram(name + ".grad", param.grad)
                #         self.writer.add_histogram(name, param)
                self.optimizer_controller.step()
            else:
                # should work for both recurrent and normal
                loss = self.train_dynamics_model(current_state, action_seq)
                self.count_finetune_data += len(current_state)

            running_loss += loss.item()
        # time_epoch = time.time() - tic
        epoch_loss = running_loss / i
        self.loss_logging(epoch_loss, train=train)
        return epoch_loss

    def run_epoch(self, train="controller"):
        self.results_dict["trained"].append(train)
        # training image dynamics
        if self.train_image_dyn:
            return self.run_image_epoch(train=train)
        elif self.train_seq_dyn:
            return self.run_sequence_epoch(train=train)
        else:
            return self.run_normal_epoch(train=train)

    def evaluate_model(self, epoch):

        eval_dyn = self.train_dynamics if isinstance(
            self.train_dynamics, SequenceCartpoleDynamics
        ) else None
        evaluator = Evaluator(
            self.model_wrapped, self.eval_env, eval_dyn=eval_dyn
        )
        # Start in upright position and see how long it is balaned
        res_eval = evaluator.evaluate_in_environment(
            nr_iters=10, render=self.train_image_dyn
        )
        success_mean = res_eval["mean_vel"]
        success_std = res_eval["std_vel"]
        for key, val in res_eval.items():
            self.results_dict[key].append(val)
        self.results_dict["evaluate_at"].append(epoch)
        self.save_model(epoch, success_mean, success_std)

        # increase thresholds
        if epoch % 3 == 0 and self.config["thresh_div"] < self.thresh_div_end:
            self.config["thresh_div"] += self.config["thresh_div_step"]

    def finalize(self):
        torch.save(
            self.state_to_img_net.state_dict(),
            os.path.join(self.save_path, "state_img_net")
        )
        super().finalize(plot_loss="loss_dynamics")


def train_control(base_model, config, swingup=0):
    """
    Train a controller from scratch or with an initial model
    """
    modified_params = config["modified_params"]
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
    trainer = TrainCartpole(train_dyn, eval_dyn, config, train_image_dyn=1)
    trainer.initialize_model(
        base_model, load_dataset="data/cartpole_img_23_wind_notcentered.npz"
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
    modified_params = config["modified_params"]
    # Only collect experience in the trained dynamics, not the ground truth
    config["sample_in"] = "train_env"
    config["resample_every"] = 1000
    config["train_dyn_for_epochs"] = -1
    config["train_dyn_every"] = 1

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
    trainer = TrainCartpole(train_dyn, eval_dyn, config, train_image_dyn=1)
    trainer.initialize_model(
        base_model,
        load_dataset="data/cartpole_img_24_wind_centered.npz",
        load_state_to_img=None
    )
    trainer.run_dynamics(config)


def train_norm_dynamics(base_model, config, not_trainable="all"):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
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

    # # NORMAL TRAINING OF CONTROLLER FROM SCRATCH AND WITH FINETUNING
    # baseline_model = None  #  "trained_models/cartpole/current_model"
    # baseline_dyn = None
    # config["save_name"] = "cartpole_controller"
    # mod_params = {"wind": .5}
    # config["modified_params"] = mod_params
    # train_control(baseline_model, config)
    # train_norm_dynamics(baseline_model, config, not_trainable="all")

    # # FINETUNE DYNAMICS (TRAIN RESIDUAL)
    # baseline_model = None
    # baseline_dyn = None
    # config["save_name"] = "dyn_img_wrenderer_trained"
    # train_img_dynamics(
    #     None, config, not_trainable="all", base_image_dyn=baseline_dyn
    # )

    # TRAIN CONTROLLER AFTER DYNAMICS ARE FINETUNED
    # baseline_model = None
    # baseline_dyn = "trained_models/cartpole/dyn_img_wrenderer_trained"
    # config["save_name"] = "con_img_corrected"

    # train_img_controller(
    #     baseline_model,
    #     config,
    #     not_trainable="all",
    #     base_image_dyn=baseline_dyn
    # )

    # TRAIN DYNAMICS WITH SEQUENCE
    # base_model = "trained_models/cartpole/final_baseline_nocontact"
    # baseline_dyn = None
    # config["save_name"] = "dyn_seq_1000_newdata"
    # config["sample_in"] = "train_env"
    # config["resample_every"] = 1000
    # config["train_dyn_for_epochs"] = 200
    # config["thresh_div_start"] = .21
    # config["train_dyn_every"] = 1
    # config["min_epochs"] = 100
    # config["eval_var_dyn"] = "mean_dyn_trained"
    # config["eval_var_con"] = "mean_vel"
    # config["learning_rate_dynamics"] = 0.01

    # # train environment is learnt
    # train_dyn = SequenceCartpoleDynamics()
    # eval_dyn = CartpoleDynamics({"contact": 1})
    # trainer = TrainCartpole(train_dyn, eval_dyn, config, train_seq_dyn=1)
    # trainer.initialize_model(
    #     base_model, load_dataset="data/cartpole_seq_1000.npz"
    # )
    # # RUN
    # trainer.run_dynamics(config)

    # TRAIN CONTROLLER WITH SEQUENCE
    num_samples = 500
    base_model = "trained_models/cartpole/final_baseline_nocontact"
    baseline_dyn = None  # "trained_models/cartpole/dyn_seq_1000_newdata"
    config["save_name"] = f"con_seq_{num_samples}"

    config["sample_in"] = "eval_env"
    config["resample_every"] = 1000
    config["train_dyn_for_epochs"] = 50
    config["thresh_div_start"] = 0.2
    # no self play possible for contact dynamics!
    config["self_play"] = 0
    config["learning_rate_controller"] = 1e-6
    config["learning_rate_dynamics"] = 0.1
    config["min_epochs"] = 100
    config["eval_var_dyn"] = "mean_dyn_trained"
    config["eval_var_con"] = "mean_vel"
    config["suc_up_down"] = -1
    config["use_samples"] = num_samples

    # train environment is learnt
    train_dyn = SequenceCartpoleDynamics()
    if baseline_dyn is not None:
        train_dyn.load_state_dict(
            torch.load(os.path.join(baseline_dyn, "dynamics_model"))
        )
    eval_dyn = CartpoleDynamics({"contact": 1})
    trainer = TrainCartpole(train_dyn, eval_dyn, config, train_seq_dyn=1)
    trainer.initialize_model(
        base_model, load_dataset="data/cartpole_seq_1000.npz"
    )
    # RUN
    trainer.run_dynamics(config)

    ## USED TO PRETRAIN CONTROLLER: (set random init in evaluate!)
    # base_model = None
    # baseline_dyn = None
    # config["save_name"] = "final_baseline_nocontact"
    # config["sample_in"] = "train_env"
    # config["resample_every"] = 1000
    # config["train_dyn_for_epochs"] = -1
    # config["thresh_div_start"] = 0.2
    # config["balance"]["learning_rate_controller"] = 1e-6

    # # train environment is learnt
    # train_dyn = CartpoleDynamics()
    # eval_dyn = CartpoleDynamics()
    # trainer = TrainCartpole(train_dyn, eval_dyn, config, train_seq_dyn=1)
    # trainer.initialize_model(
    #     base_model, load_dataset="data/cartpole_img_28_nocontact.npz"
    # )
    # trainer.run_dynamics(config)
