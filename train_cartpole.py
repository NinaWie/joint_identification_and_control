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
    Net, ImageControllerNet, ImageControllerNetDQN, StateToImg
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
        else:
            self.state_data = CartpoleDataset(
                num_states=self.config["sample_data"], **self.config
            )
        with open(os.path.join(self.save_path, "config.json"), "w") as outfile:
            json.dump(self.config, outfile)

        # load state to img netwrok if it was finetuned
        if load_state_to_img is not None:
            self.state_to_img_net.load_state_dict(
                torch.load(os.path.join(base_image_dyn, "state_to_img"))
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

                for name, param in self.net.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(name + ".grad", param.grad)
                        self.writer.add_histogram(name, param)
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

        print(
            "self play:", self.state_data.eval_counter,
            self.state_data.get_eval_index()
        )

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
            controller_model = CartpoleImageWrapper(
                self.net, self.state_data, **self.config
            )
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
    modified_params = config["general"]["modified_params"]
    config["general"]["sample_in"] = "eval_env"
    config["general"]["resample_every"] = 1000
    config["train_dyn_for_epochs"] = 200
    config["train_dyn_every"] = 1
    # No self play!
    config["self_play"] = 0

    # train environment is learnt
    # train_dyn = CartpoleDynamics(modified_params=modified_params)
    train_dyn = ImageCartpoleDynamics(
        100, 120, nr_img=config["general"]["nr_img"], state_size=4
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
    modified_params = config["general"]["modified_params"]
    # Only collect experience in the trained dynamics, not the ground truth
    config["general"]["sample_in"] = "train_env"
    config["general"]["resample_every"] = 1000
    config["train_dyn_for_epochs"] = -1
    config["train_dyn_every"] = 1

    # train environment is learnt (next line only for sanity check)
    # train_dyn = CartpoleDynamics(modified_params=modified_params)
    train_dyn = ImageCartpoleDynamics(
        100, 120, nr_img=config["general"]["nr_img"], state_size=4
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
        load_state_to_img=base_image_dyn
    )
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

    # # NORMAL TRAINING OF CONTROLLER FROM SCRATCH AND WITH FINETUNING
    # baseline_model = None  #  "trained_models/cartpole/current_model"
    # baseline_dyn = None
    # config["general"]["save_name"] = "cartpole_controller"
    # mod_params = {"wind": .5}
    # config["general"]["modified_params"] = mod_params
    # train_control(baseline_model, config)
    # train_norm_dynamics(baseline_model, config, not_trainable="all")

    # # FINETUNE DYNAMICS (TRAIN RESIDUAL)
    # baseline_model = None
    # baseline_dyn = None
    # config["general"]["save_name"] = "dyn_img_wrenderer_trained"
    # train_img_dynamics(
    #     None, config, not_trainable="all", base_image_dyn=baseline_dyn
    # )

    # TRAIN CONTROLLER AFTER DYNAMICS ARE FINETUNED
    baseline_model = None
    baseline_dyn = "trained_models/cartpole/dyn_img_wrenderer_trained"
    config["general"]["save_name"] = "con_img_corrected"

    train_img_controller(
        baseline_model,
        config,
        not_trainable="all",
        base_image_dyn=baseline_dyn
    )
