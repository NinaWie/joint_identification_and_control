import json
import os
import torch
import numpy as np

from train_fixed_wing import TrainFixedWing
from neural_control.dataset import WingSequenceDataset
from neural_control.controllers.network_wrapper import FixedWingNetWrapper
from neural_control.dynamics.fixed_wing_dynamics import (
    FixedWingDynamics, SequenceFixedWingDynamics
)
from evaluate_fixed_wing import FixedWingEvaluator
from neural_control.drone_loss import fixed_wing_mpc_loss
from neural_control.models.hutter_model import Net


class TrainSequenceWing(TrainFixedWing):

    # def run_epoch(self):
    #     pass

    def initialize_model(self, base_model=None, base_model_name="model_wing"):
        if base_model is not None:
            self.net = torch.load(os.path.join(base_model, base_model_name))
        else:
            self.net = Net(
                (self.state_size + self.action_dim) *
                self.config.get("buffer_len", 3),
                1,
                self.ref_dim,
                self.action_dim * self.nr_actions,
                conv=False
            )

        self.state_data = WingSequenceDataset(self.epoch_size, **self.config)
        self.model_wrapped = FixedWingNetWrapper(
            self.net, self.state_data, **self.config
        )
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

            actions = self.net(in_state, in_ref_state)
            actions = torch.sigmoid(actions)
            action_seq = torch.reshape(
                actions, (-1, self.nr_actions, self.action_dim)
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
                # bl_dyn = FixedWingDynamics()
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
                    if isinstance(
                        self.train_dynamics, SequenceFixedWingDynamics
                    ):
                        current_state = self.train_dynamics(
                            current_state,
                            in_state,
                            action,
                            dt=self.delta_t_train
                        )
                    else:
                        current_state = self.train_dynamics(
                            current_state, action, dt=self.delta_t_train
                        )
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
                loss = fixed_wing_mpc_loss(
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
                # if i == 0:
                #     np.set_printoptions(suppress=1, precision=3)
                #     print(timestamps[0])
                #     print(next_state_d1[0].detach().numpy())
                #     print(next_state_d2[0].detach().numpy())
                #     print()
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
    with open("configs/wing_config.json", "r") as infile:
        config = json.load(infile)

    # # USED TO PRETRAIN CONTROLLER: (set random init in evaluate!)
    # # OR FINETUNE WO updated residual
    # base_model = None
    # #"trained_models/wing/final_baseline_seq_wing" # finetune
    # baseline_dyn = None
    # config["save_name"] = "final_baseline_seq_wing"
    # # "baseline_seq_wing_finetuned" # finetune
    # config["sample_in"] = "train_env"
    # # "eval_env" # finetune
    # config["resample_every"] = 1000
    # config["train_dyn_for_epochs"] = -1
    # config["epoch_size"] = 2000
    # config["self_play"] = 2000
    # # config["epoch_size"] = 500 # finetune
    # # config["self_play"] = 500 # finetune
    # config["buffer_len"] = 3

    # # train environment is learnt
    # train_dyn = FixedWingDynamics()
    # eval_dyn = FixedWingDynamics()  # modified_params={"wind": 2}) # finetune
    # trainer = TrainSequenceWing(train_dyn, eval_dyn, config)
    # trainer.initialize_model(base_model)
    # trainer.run_dynamics(config)

    # # FINETUNE DYNAMICS
    # base_model = "trained_models/wing/final_baseline_seq_wing"
    # baseline_dyn = None
    # config["save_name"] = "dyn_seq_wing_5"

    # mod_param = {"wind": 2}

    # config["learning_rate_dynamics"] = 0.01  # was 0.0001
    # config["sample_in"] = "eval_env"
    # config["thresh_div_start"] = 20
    # config["thresh_stable_start"] = 1.5
    # config["train_dyn_for_epochs"] = 200
    # config["epoch_size"] = 200
    # config["self_play"] = 200
    # config["resample_every"] = 10
    # config["waypoint_metric"] = True

    # train_dyn = SequenceFixedWingDynamics()
    # eval_dyn = FixedWingDynamics(modified_params=mod_param)
    # trainer = TrainSequenceWing(train_dyn, eval_dyn, config)
    # trainer.initialize_model(base_model)
    # trainer.run_dynamics(config)

    # FINETUNE CONTROLLER
    base_model = "trained_models/wing/final_baseline_seq_wing"
    baseline_dyn = None  # "trained_models/wing/dyn_seq_wing_2"
    config["save_name"] = "iterative_seq_wing_3"

    config["sample_in"] = "eval_env"
    # config["train_dyn_for_epochs"] = -1
    config["learning_rate_controller"] = 0.00001  # was 0.0001
    config["learning_rate_dynamics"] = 0.005
    # config["thresh_div_start"] = 20
    # config["thresh_stable_start"] = 1.5
    config["epoch_size"] = 500
    config["self_play"] = 500
    # config["resample_every"] = 2
    config["buffer_len"] = 3
    # variables to check whether we have converged
    config["eval_var_dyn"] = "mean_trained_delta"
    config["eval_var_con"] = "mean_div_linear"
    config["min_epochs"] = 8

    # train environment is learnt
    # train_dyn = FixedWingDynamics()
    train_dyn = SequenceFixedWingDynamics()
    if baseline_dyn is not None:
        train_dyn.load_state_dict(
            torch.load(os.path.join(baseline_dyn, "dynamics_model"))
        )
    eval_dyn = FixedWingDynamics(modified_params={"wind": 2})
    trainer = TrainSequenceWing(train_dyn, eval_dyn, config)
    trainer.initialize_model(base_model)
    trainer.run_iterative(config)
