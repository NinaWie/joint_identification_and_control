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
from neural_control.models.hutter_model import Net


class TrainSequenceWing(TrainFixedWing):

    # def run_epoch(self):
    #     pass

    def initialize_model(
        self, base_model=None, base_model_name="model_cartpole"
    ):
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


if __name__ == "__main__":
    # LOAD CONFIG
    with open("configs/wing_config.json", "r") as infile:
        config = json.load(infile)

    # # USED TO PRETRAIN CONTROLLER: (set random init in evaluate!)
    base_model = None
    baseline_dyn = None
    config["save_name"] = "baseline_seq_wing"
    config["sample_in"] = "train_env"
    config["resample_every"] = 1000
    config["train_dyn_for_epochs"] = -1
    config["epoch_size"] = 2000
    config["self_play"] = 2000
    config["buffer_len"] = 3

    # train environment is learnt
    train_dyn = FixedWingDynamics()
    eval_dyn = FixedWingDynamics()
    trainer = TrainSequenceWing(train_dyn, eval_dyn, config)
    trainer.initialize_model(base_model)
    trainer.run_dynamics(config)

    # # FINETUNE DYNAMICS
    # base_model = None
    # baseline_dyn = None
    # config["save_name"] = "dyn_seq_wing"

    # mod_param = {"wind": 2}

    # config["sample_in"] = "eval_env"  # TODO
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
    # base_model = "trained_models/cartpole/baseline_seq_wing"
    # baseline_dyn = "trained_models/cartpole/dyn_seq_wing"
    # config["general"]["save_name"] = "con_seq_wing"

    # config["sample_in"] = "eval_env"
    # config["train_dyn_for_epochs"] = -1
    # config["thresh_div_start"] = 20
    # config["thresh_stable_start"] = 1.5
    # config["train_dyn_for_epochs"] = 200
    # config["epoch_size"] = 2000
    # config["self_play"] = 2000  # TODO

    # mod_param = {"wind": 2}

    # # train environment is learnt
    # train_dyn = SequenceFixedWingDynamics()
    # train_dyn.load_state_dict(
    #     torch.load(os.path.join(baseline_dyn, "dynamics_model"))
    # )
    # eval_dyn = FixedWingDynamics(modified_params=mod_param)
    # trainer = TrainSequenceWing(train_dyn, eval_dyn, config)
    # trainer.initialize_model(base_model)
    # trainer.run_dynamics(config)
