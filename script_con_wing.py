import json
import os
import torch

from train_fixed_wing import TrainFixedWing
from neural_control.dynamics.fixed_wing_dynamics import (
    FixedWingDynamics, LearntFixedWingDynamics
)


def finetune_control(base_model, config, load_dynamics):
    """
    Train a controller from scratch or with an initial model
    """
    modified_params = config["modified_params"]
    # TODO: might be problematic
    train_dynamics = LearntFixedWingDynamics()
    train_dynamics.load_state_dict(torch.load(load_dynamics))
    eval_dynamics = FixedWingDynamics(modified_params)

    # make sure that also the self play samples are collected in same env
    if config["self_play"] > 0:
        config["sample_in"] = "train_env"
    else:
        config["sample_in"] = "eval_env"
    # start at higher thresh div:
    config["thresh_div_start"] = 10
    config["thresh_stable_start"] = 1.5

    trainer = TrainFixedWing(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    trainer.run_control(config, curriculum=0)


if __name__ == "__main__":
    # LOAD CONFIG
    with open("configs/wing_config.json", "r") as infile:
        config = json.load(infile)

    baseline_model = "trained_models/wing/current_model"
    config["save_name"] = "train_con_residual_matrix_test"

    # config["self_play"] = 0
    # config["epoch_size"] = 1000
    config["learning_rate_controller"] = 0.00001

    config["modified_params"] =  {"residual_factor": 0.0001}

    load_dyn_path = "trained_models/wing/train_residual_matrix/dynamics_model"

    # train_dynamics(baseline_model, config)
    finetune_control(baseline_model, config, load_dyn_path)
