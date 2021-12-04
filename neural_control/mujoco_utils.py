from mbrl.models import model
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import hydra

import mbrl.models
import mbrl.planning
import mbrl.types
import omegaconf
from mbrl.models.model_env import ModelEnv
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns

# Cheetah environment
from mbrl.env.pets_halfcheetah import HalfCheetahEnv
env = HalfCheetahEnv()
reward_fn = reward_fns.halfcheetah
term_fn = termination_fns.no_termination

trial_length = 200  # 200
num_trials = 100  # 10
ensemble_size = 5

cfg_dict = {
    # dynamics model configuration
    "dynamics_model":
        {
            "model":
                {
                    "_target_": "mbrl.models.GaussianMLP",
                    "device": "cpu",
                    "num_layers": 3,
                    "ensemble_size": ensemble_size,
                    "hid_size": 200,
                    "use_silu": True,
                    "in_size": "???",
                    "out_size": "???",
                    "deterministic": False,
                    "propagation_method": "fixed_model"
                }
        },
    # options for training the dynamics model
    "algorithm":
        {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True,
        },
    # these are experiment specific options
    "overrides":
        {
            "trial_length": trial_length,
            "num_steps": num_trials * trial_length,
            "model_batch_size": 32,
            "validation_ratio": 0.05
        }
}
cfg = omegaconf.OmegaConf.create(cfg_dict)
act_shape = env.action_space.shape
obs_shape = env.observation_space.shape
model_cfg = cfg.dynamics_model.model

if model_cfg._target_ == "mbrl.models.BasicEnsemble":
    model_cfg = model_cfg.member_cfg
if model_cfg.get("in_size", None) is None:
    model_cfg.in_size = obs_shape[0] + (act_shape[0] if act_shape else 1)
if model_cfg.get("out_size", None) is None:
    model_cfg.out_size = obs_shape[0] + int(cfg.algorithm.learned_rewards)


class ControllerModel(nn.Module):
    """
    Simple MLP with three hidden layers, based on RL work of Marco Hutter's
    group
    """

    def __init__(self, state_dim, out_size, nr_actions=1):
        """
        in_size: number of input neurons (features)
        out_size: number of output neurons
        """
        super(ControllerModel, self).__init__()
        self.out_size = out_size
        self.nr_actions = nr_actions
        self.states_in = nn.Linear(state_dim, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, out_size * nr_actions)

    def forward(self, state):
        # concatenate
        x = torch.relu(self.states_in(state))
        # normal feed-forward
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = torch.tanh(self.bn2(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc_out(x))
        if self.nr_actions > 1:
            x = torch.reshape(x, (-1, self.nr_actions, self.out_size))
        return x


class DynamicsModelPETS:

    def __init__(
        self, path="trained_models/out_mbrl/test_halfcheetah/eps_189_38200"
    ):
        model = hydra.utils.instantiate(cfg.dynamics_model.model)
        self.dynamics_model = mbrl.models.OneDTransitionRewardModel(
            model,
            target_is_delta=cfg.algorithm.target_is_delta,
            normalize=cfg.algorithm.normalize,
            normalize_double_precision=cfg.algorithm.get(
                "normalize_double_precision", False
            ),
            learned_rewards=cfg.algorithm.learned_rewards,
            obs_process_fn=None,
            no_delta_list=cfg.overrides.get("no_delta_list", None),
            num_elites=cfg.overrides.get("num_elites", None),
        )

        self.dynamics_model.load(path)
        self.normalizer = self.dynamics_model.input_normalizer

    def forward(self, obs, normed_obs, normed_act):
        model_inp = torch.cat((normed_obs, normed_act), dim=1)
        # normed = self.normalizer.normalize(model_inp).float()
        model_inp = model_inp.repeat(5, 1)

        preds = self.dynamics_model.model.sample(
            model_inp,
            deterministic=True,
            # TODO: here was this rng shit, maybe we need it
        )[0]
        preds_batched = torch.reshape(preds, (-1, 5, obs.size()[1]))
        pred_obs_diff = torch.mean(preds_batched, dim=1)
        # TODO: check whether reshaping works correctly
        return obs + pred_obs_diff
