import json
import os

from train_fixed_wing import TrainFixedWing
from neural_control.dynamics.fixed_wing_dynamics import (
    FixedWingDynamics, LearntFixedWingDynamics
)


def train_dynamics(base_model, config):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
    config["sample_in"] = "train_env"

    # train environment is learnt
    train_dynamics = LearntFixedWingDynamics()
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
    config["save_name"] = "train_residual_minus_w_params"

    # set high thresholds because not training from scratch
    config["thresh_div_start"] = 20
    config["thresh_stable_start"] = 1.5
    # set self play to zero to avoid bad actions
    config["self_play"] = 0
    config["l2_lambda"] = 0
    config["waypoint_metric"] = False

    mod_params = {"residual_factor": 0.03}
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
    config["nr_epochs"] = 10
    train_dynamics(baseline_model, config)
