import numpy as np
import torch

import torch.nn as nn
import torch
import torch.nn.functional as F

from mbrl.env.pets_halfcheetah import HalfCheetahEnv
from neural_control.mujoco_utils import ControllerModel, DynamicsModelPETS
from scripts.evaluate_mujoco import evaluate_cheetah


class MujocoDataset(torch.utils.data.Dataset):

    def __init__(self, num_states) -> None:
        self.num_states = num_states

    def __len__(self):
        return self.num_states

    def collect_data(self, env, controller):
        data = []
        while len(data) < self.num_states:
            collected_obs = evaluate_cheetah(env, controller)
            data.extend(collected_obs)
        data = np.array(data)
        print(data.shape)
        self.data = data

    def __getitem__(self, index):
        return self.data[index]


BATCH_SIZE = 4

env = HalfCheetahEnv()
train_dynamics = DynamicsModelPETS()
obs_len, act_len = (env.observation_space.shape[0], env.action_space.shape[0])
controller = ControllerModel(obs_len, act_len)
dataset = MujocoDataset(500)
dataset.collect_data(env, controller)

obs_mean, obs_std = (
    train_dynamics.normalizer.mean[0, :obs_len],
    train_dynamics.normalizer.std[0, :obs_len]
)
act_mean, act_std = (
    train_dynamics.normalizer.mean[0, obs_len:],
    train_dynamics.normalizer.std[0, obs_len:]
)

trainloader = trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
for obs in trainloader:
    # --- testing with env -----
    obs_init = env.reset()
    obs = torch.from_numpy(np.expand_dims(obs_init, 0))
    # --- testing with env -----

    normed_obs = ((obs - obs_mean) / obs_std).float()

    act = controller(normed_obs)

    normed_act = (act - act_mean) / act_std

    next_state = train_dynamics.forward(obs, normed_obs, normed_act)

    print(next_state)

    real_next_obs, _, _, _ = env.step(act.detach().numpy())
    print(torch.from_numpy(real_next_obs))

    exit()