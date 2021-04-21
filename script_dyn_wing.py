import json
import os
import torch

from train_fixed_wing import TrainFixedWing
from neural_control.dynamics.fixed_wing_dynamics import (
    FixedWingDynamics, LearntFixedWingDynamics
)

never_trainable = ["rho", "g", "residual_factor"]


def train_dynamics(base_model, config, not_trainable):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
    config["sample_in"] = "train_env"

    # train environment is learnt
    train_dynamics = LearntFixedWingDynamics(not_trainable=not_trainable)
    eval_dynamics = FixedWingDynamics(modified_params=modified_params)

    trainer = TrainFixedWing(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    # RUN
    trainer.run_dynamics(config)


if __name__ == "__main__":
    # LOAD CONFIG
    with open("configs/wing_config.json", "r") as infile:
        config = json.load(infile)

    baseline_model = "trained_models/wing/current_model"
    config["save_name"] = "train_residual_matrix"

    # set high thresholds because not training from scratch
    config["thresh_div_start"] = 20
    config["thresh_stable_start"] = 1.5
    # set self play to zero to avoid bad actions
    config["self_play"] = 0
    config["epoch_size"] = 1000
    # lambda: how much delta network is penalized
    config["l2_lambda"] = 0
    config["waypoint_metric"] = True

    mod_params = {"residual_factor": 0.0001}
    #  {"rho": 1.6}
    # {
    #     "CL0": 0.3,  # 0.39
    #     "CD0": 0.02,  #  0.0765,
    #     "CY0": 0.02,  # 0.0,
    #     "Cl0": -0.01,  # 0.0,
    #     "Cm0": 0.01,  # 0.02,
    #     "Cn0": 0.0,
    # }
    # # {"rho": 1.4, "mass": 1.2, "S": 0.32}  # mass: 1.4
    config["modified_params"] = mod_params

    # TRAIN
    config["nr_epochs"] = 20

    train_dynamics(
        baseline_model,
        config,
        not_trainable=never_trainable + ["vel_drag_factor"]
    )
