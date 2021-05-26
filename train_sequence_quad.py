import json
import os
import torch
import numpy as np

from train_drone import TrainDrone
from neural_control.dataset import QuadSequenceDataset
from neural_control.controllers.network_wrapper import NetworkWrapper
from neural_control.dynamics.quad_dynamics_flightmare import (
    FlightmareDynamics
)
from neural_control.dynamics.quad_dynamics_trained import SequenceQuadDynamics
from neural_control.drone_loss import quad_mpc_loss
from neural_control.models.hutter_model import Net
from evaluate_drone import QuadEvaluator


class TrainSequenceQuad(TrainDrone):

    # def run_epoch(self):
    #     pass

    def initialize_model(self, base_model=None, base_model_name="model_quad"):
        # temporary data
        self_play_tmp = self.config["self_play"]
        epoch_size_tmp = self.epoch_size
        self.config["self_play"] = 0
        self.epoch_size = 2
        super().initialize_model(base_model=base_model)
        self.config["self_play"] = self_play_tmp
        self.epoch_size = epoch_size_tmp
        if base_model is None:
            self.net = Net(
                (18 + 4),
                self.nr_actions,
                self.ref_dim,
                self.action_dim * self.nr_actions,
                conv=1,
                hist_conv=True
            )

        self.state_data = QuadSequenceDataset(self.epoch_size, **self.config)
        self.init_optimizer()
        # set is seq for evaluation
        self.config["is_seq"] = True

    def run_epoch(self, train="controller"):
        running_loss = 0
        self.state_data.return_timestamp = True

        for i, data in enumerate(self.trainloader, 0):
            (
                in_state, state_action_history, in_ref_state, ref_states,
                timestamps
            ) = data

            # np.set_printoptions(suppress=1, precision=3)
            # print("-------------------------")
            # print("preprocessed history")
            # print(in_state[0].detach().numpy())
            # print("history")
            # print(state_action_history[0].detach().numpy())
            # print("ref")
            # print(in_ref_state[0].detach().numpy())
            # print()
            current_state = state_action_history[:, 0, :12]

            actions = self.net(in_state, in_ref_state, is_seq=1, hist_conv=1)
            actions = torch.sigmoid(actions)
            action_seq = torch.reshape(
                actions, (-1, self.nr_actions, self.action_dim)
            )
             if i % 20 == 0 and train == "dynamics":
                action_seq = torch.rand(
                    current_state.size()[0], self.nr_actions, 4
                )
            # print("actions")
            # print(action_seq[0].detach().numpy())
            if train == "controller":
                self.optimizer_controller.zero_grad()
                intermediate_states = torch.zeros(
                    current_state.size()[0], self.nr_actions_rnn,
                    self.state_size
                )
                # to double check
                # eval_dyn_state = current_state.clone()
                # bl_dyn_state = current_state.clone()
                # bl_dyn = FlightmareDynamics()
                self.eval_dynamics.timestamp = timestamps[0]
                for k in range(self.nr_actions_rnn):
                    # extract action
                    action = action_seq[:, k]
                    # compare to eval
                    # eval_dyn_state = self.eval_dynamics(
                    #     eval_dyn_state, action_seq[:, k], dt=self.delta_t
                    # )
                    # bl_dyn_state = bl_dyn(
                    #     bl_dyn_state, action, dt=self.delta_t
                    # )
                    # print()
                    # print(k)
                    # print("preprocessed history")
                    # print(in_state[0].detach().numpy())
                    # print("history")
                    # print(state_action_history[0].detach().numpy())
                    # print("action", action[0].detach().numpy())
                    if isinstance(self.train_dynamics, SequenceQuadDynamics):
                        current_state = self.train_dynamics(
                            current_state,
                            in_state,
                            action,
                            dt=self.delta_t_train
                        )
                    else:
                        # current_state = self.train_dynamics(
                        #     current_state, action, dt=self.delta_t_train
                        # )
                        next_state_d2 = torch.zeros(current_state.size())
                        # need to do all samples in batch separately
                        for sample in range(timestamps.size()[0]):
                            self.train_dynamics.timestamp = timestamps[sample]
                            current_state_in = torch.unsqueeze(
                                current_state[sample], 0
                            )
                            action_in = torch.unsqueeze(action[sample], 0)
                            next_state_d2[sample] = self.train_dynamics(
                                current_state_in, action_in, dt=self.delta_t
                            )
                        current_state = next_state_d2
                    intermediate_states[:, k] = current_state
                    # roll history
                    state_action_cat = torch.unsqueeze(
                        torch.cat((current_state, action), dim=1), dim=1
                    )
                    state_action_history = torch.cat(
                        (state_action_cat, state_action_history[:, :-1]),
                        dim=1
                    )
                    in_state = self.state_data.prepare_history(
                        state_action_history.clone()
                    )

                # if i == 0:
                #     np.set_printoptions(suppress=1, precision=3)
                #     print(timestamps[0])
                #     print("compare trained dyn to gt dyn wo force")
                #     print(intermediate_states[0].detach().numpy())
                #     print("eval and bl")
                #     print(eval_dyn_state[0].detach().numpy())
                #     print(bl_dyn_state[0].detach().numpy())
                loss = quad_mpc_loss(
                    intermediate_states, ref_states, action_seq, printout=0
                )
                # Backprop
                loss.backward()
                # for name, param in self.net.named_parameters():
                #     if param.grad is not None:
                #         self.writer.add_histogram(name + ".grad", param.grad)
                self.optimizer_controller.step()
            elif train == "dynamics":
                self.optimizer_dynamics.zero_grad()
                next_state_d1 = self.train_dynamics(
                    current_state,
                    in_state,
                    action_seq[:, 0].detach(),
                    dt=self.delta_t
                )
                bl_dyn = FlightmareDynamics()
                next_state_bl = bl_dyn(
                    current_state, action_seq[:, 0], dt=self.delta_t
                )
                next_state_d2 = torch.zeros(next_state_d1.size())
                # need to do all samples in batch separately
                for sample in range(timestamps.size()[0]):
                    self.eval_dynamics.timestamp = timestamps[sample]
                    current_state_in = torch.unsqueeze(
                        current_state[sample], 0
                    )
                    action_in = torch.unsqueeze(
                        action_seq[sample, 0].detach(), 0
                    )
                    next_state_d2[sample] = self.eval_dynamics(
                        current_state_in, action_in, dt=self.delta_t
                    )
                if i == 0:
                    np.set_printoptions(suppress=1, precision=3)
                    print(torch.sin(timestamps[0]))
                    print("tr", next_state_d1[0].detach().numpy())
                    d2_example = next_state_d2[0].detach().numpy()
                    d1_example = next_state_bl[0].detach().numpy()
                    print("d2", d2_example)
                    print("d1", d1_example)
                    if torch.sin(timestamps[0]) > .5:
                        assert d2_example[1] > d1_example[1]
                    elif torch.sin(timestamps[0]) < -.5:
                        assert d2_example[1] < d1_example[1]
                    else:
                        if not d2_example[1] == d1_example[1]:
                            print(timestamps.detach().numpy())
                            print(next_state_d2.detach().numpy())
                            print(next_state_d1.detach().numpy())
                            print(next_state_bl.detach().numpy())
                loss = torch.sum((next_state_d1 - next_state_d2)**2)
                loss.backward()
                self.optimizer_dynamics.step()
                # delta squared is loss divided by batch size
                self.results_dict["loss_delta_squared"].append(
                    loss.item() / next_state_d1.size()[0]
                )
                # approximate delta times 1000 / dt
                loss = loss * 1000 / (self.batch_size * self.delta_t)
            running_loss += loss.item()

        epoch_loss = running_loss / i

        # log losses
        not_trained = "controller" if train == "dynamics" else "dynamics"
        self.results_dict["loss_" + train].append(epoch_loss)
        try:
            self.results_dict["loss_" + not_trained].append(
                self.results_dict["loss_" + not_trained][-1]
            )
        except IndexError:
            self.results_dict["loss_" + not_trained].append(0)

        self.results_dict["loss"].append(epoch_loss)
        self.results_dict["trained"].append(train)
        print(f"Loss ({train}): {round(epoch_loss, 2)}")
        # self.writer.add_scalar("Loss/train", epoch_loss)
        return epoch_loss


if __name__ == "__main__":
    # LOAD CONFIG
    with open("configs/quad_config.json", "r") as infile:
        config = json.load(infile)

    # # USED TO PRETRAIN CONTROLLER: (set random init in evaluate!)
    # OR FINETUNE WO updated residual
    # base_model = "trained_models/quad/baseline_con_seq_conv"
    # config["save_name"] = "baseline_con_seq_conv_2"
    # # "baseline_seq_quad_finetuned" # finetune
    # config["sample_in"] = "train_env"
    # # "eval_env" # finetune
    # config["resample_every"] = 2
    # config["train_dyn_for_epochs"] = -1
    # config["epoch_size"] = 1000
    # config["self_play"] = 1000
    # # config["epoch_size"] = 500 # finetune
    # # config["self_play"] = 500 # finetune
    # config["buffer_len"] = 5

    # # for finetuning
    # # config["thresh_div_start"] = 3
    # # config["thresh_stable_start"] = 2
    # # config["suc_up_down"] = -1
    # # config["return_div"] = 1

    # # train environment is learnt
    # train_dyn = FlightmareDynamics()
    # eval_dyn = FlightmareDynamics()  # modified_params={"wind": 2}) # finetune
    # trainer = TrainSequenceQuad(train_dyn, eval_dyn, config)
    # trainer.initialize_model(base_model)
    # trainer.run_control(config, curriculum=1)  # finetune 0

    # # FINETUNE DYNAMICS ITERATIVELY
    base_model = "trained_models/quad/final_baseline_con_seq"
    baseline_dyn = "trained_models/quad/iterative_seq_newwind_dyn"
    config["save_name"] = "iterative_seq_newwind_con"

    # mod_param = {'translational_drag': np.array([0.3, 0.3, 0.3])}
    mod_param = {"wind": 2}
    config["learning_rate_controller"] = 0.000001
    config["learning_rate_dynamics"] = 0.001
    config["thresh_div_start"] = 1
    config["thresh_div_end"] = 1.2
    config["thresh_stable_start"] = 2
    config["sample_in"] = "eval_env"
    config["epoch_size"] = 500  # 200  # for dyn training
    config["self_play"] = 500  # 200  # for dyn training
    config["buffer_len"] = 5
    config["eval_var_dyn"] = "mean_trained_delta"
    config["eval_var_con"] = "mean_div"
    config["min_epochs"] = 3  # for dyn training # 5 for con training
    config["suc_up_down"] = -1
    config["self_play_every_x"] = 5
    config["return_div"] = 1

    # train_dyn = FlightmareDynamics(modified_params=mod_param)
    train_dyn = SequenceQuadDynamics(buffer_length=3)
    if baseline_dyn is not None:
        train_dyn.load_state_dict(
            torch.load(os.path.join(baseline_dyn, "dynamics_model"))
        )
    eval_dyn = FlightmareDynamics(modified_params=mod_param)
    trainer = TrainSequenceQuad(train_dyn, eval_dyn, config)
    trainer.initialize_model(base_model)
    # trainer.run_iterative(config, start_with="controller")
    trainer.run_sequentially(config, start_with="controller")
