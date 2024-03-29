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


class TrainSequenceCartpole(TrainBase):
    """
    Train a controller for a quadrotor
    """

    def __init__(self, train_dynamics, eval_dynamics, config, swingup=0):
        """
        param sample_in: one of "train_env", "eval_env"
        """
        self.config = config
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

    def loss_logging(self, epoch_loss, train="controller"):
        self.results_dict["loss_" + train].append(epoch_loss)
        print(f"Loss ({train}): {round(epoch_loss, 2)}")
        # self.writer.add_scalar("Loss/train", epoch_loss)

    def run_epoch(self, train="controller"):
        self.results_dict["trained"].append(train)
        # training image dynamics

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
                # if i == 0:
                #     print("start at ", current_state[0])
                #     print("pred", next_state_pred[0].detach())
                #     print("gt", next_state_d2[0])
                #     print()

                loss = torch.sum((next_state_pred - next_state_d2)**2)
                loss.backward()
                self.optimizer_dynamics.step()
                self.results_dict["loss_dyn_per_step"].append(loss.item())
                loss += 1000

            running_loss += loss.item()
        # time_epoch = time.time() - tic
        epoch_loss = running_loss / i
        self.loss_logging(epoch_loss * 1000, train=train)
        return epoch_loss

    def evaluate_model(self, epoch):

        new_data = self.evaluate_balance(epoch)

        print(
            "self play:", self.state_data.eval_counter,
            self.state_data.get_eval_index()
        )

    def evaluate_balance(self, epoch):
        # EVALUATION:
        eval_dyn = self.train_dynamics if isinstance(
            self.train_dynamics, SequenceCartpoleDynamics
        ) else None

        self.model_wrapped.self_play = self.config["self_play"]
        evaluator = Evaluator(
            self.model_wrapped, self.eval_env, eval_dyn=eval_dyn
        )
        # Start in upright position and see how long it is balaned
        res_eval = evaluator.evaluate_in_environment(nr_iters=10, render=False)
        success_mean = res_eval["mean_vel"]
        success_std = res_eval["std_vel"]
        for key, val in res_eval.items():
            self.results_dict[key].append(val)
        self.results_dict["evaluate_at"].append(epoch)
        self.save_model(epoch, success_mean, success_std)

        # increase thresholds
        if epoch % 3 == 0 and self.config["thresh_div"] < self.thresh_div_end:
            self.config["thresh_div"] += self.config["thresh_div_step"]


if __name__ == "__main__":
    # LOAD CONFIG - select balance or swigup
    with open("configs/cartpole_config.json", "r") as infile:
        config = json.load(infile)

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

    # FINETUNE DYNAMICS AND CONTROLLER TOGETHER
    # for num_samples in np.arange(200, 2000, 200):
    #     print()
    #     print(f"----------- number of samples {num_samples} -----------")
    #     # num_samples = 1000
    #     trainer = None
    #     # reload config:
    #     with open("configs/cartpole_config.json", "r") as infile:
    #         config = json.load(infile)

    #     mode = "_random"
    #     base_model = "trained_models/cartpole/final_baseline_nocontact"
    #     baseline_dyn = None  # "trained_models/cartpole/dyn_seq_1000_newdata"
    #     config["save_name"] = f"con_seq_{num_samples}" + mode

    #     config["sample_in"] = "eval_env"
    #     config["resample_every"] = 1000
    #     config["nr_epochs"] = 110
    #     config["train_dyn_for_epochs"] = 20
    #     config["thresh_div_start"] = 0.2
    #     # no self play possible for contact dynamics!
    #     config["self_play"] = 0
    #     config["learning_rate_controller"] = 1e-7
    #     config["learning_rate_dynamics"] = 0.05
    #     config["min_epochs"] = 100
    #     config["eval_var_dyn"] = "mean_dyn_trained"
    #     config["eval_var_con"] = "mean_vel"
    #     config["suc_up_down"] = -1
    #     config["use_samples"] = int(num_samples)

    #     # train environment is learnt
    #     train_dyn = SequenceCartpoleDynamics()
    #     if baseline_dyn is not None:
    #         train_dyn.load_state_dict(
    #             torch.load(os.path.join(baseline_dyn, "dynamics_model"))
    #         )
    #     eval_dyn = CartpoleDynamics({"contact": 1})
    #     trainer = TrainCartpole(train_dyn, eval_dyn, config, train_seq_dyn=1)
    #     trainer.initialize_model(
    #         base_model, load_dataset=f"data/cartpole_seq_2000{mode}.npz"
    #     )
    #     # RUN
    #     trainer.run_dynamics(config)

    # # USED TO PRETRAIN CONTROLLER: (set random init in evaluate!)
    base_model = None
    baseline_dyn = None
    config["save_name"] = "test_cartpole"
    config["sample_in"] = "train_env"
    config["resample_every"] = 1000
    config["train_dyn_for_epochs"] = -1
    config["thresh_div_start"] = 0.2
    config["learning_rate_controller"] = 1e-6

    # train environment is learnt
    train_dyn = CartpoleDynamics()
    eval_dyn = CartpoleDynamics()
    trainer = TrainSequenceCartpole(train_dyn, eval_dyn, config)
    trainer.initialize_model(
        base_model, load_dataset="data/cartpole_img_28_nocontact.npz"
    )
    trainer.run_dynamics(config)
