import numpy as np
import torch

import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import torch.nn.functional as F

from mbrl.env.pets_halfcheetah import HalfCheetahEnv
from neural_control.mujoco_utils import (ControllerModel, DynamicsModelPETS)
from scripts.evaluate_mujoco import evaluate_cheetah, run_eval


class MujocoDataset(torch.utils.data.Dataset):

    def __init__(self, num_states) -> None:
        self.num_states = num_states

    def __len__(self):
        return self.num_states

    def collect_data(self, env, controller):
        data = []
        while len(data) < self.num_states:
            collected_obs = evaluate_cheetah(env, controller, return_obs=True)
            data.extend(collected_obs)
        data = np.array(data)
        self.data = data
        print("-------------- RESAMPLE", np.mean(self.data))

    def __getitem__(self, index):
        return self.data[index]


class RandomController:

    def __call__(self, obs):
        return torch.from_numpy(np.expand_dims(np.random.rand(6), axis=0))


nr_epochs = 2000
batch_size = 8
samples_per_epoch = 5000
resample_every_x_epochs = 5
learning_rate_controller = 0.001
# NOTES
# currently using random controller only in the beginning for sampling
# loss function: negative values seem to be fine!
# careful: in evaluate mujoco, obs is currently not normalized!!

env = HalfCheetahEnv()
train_dynamics = DynamicsModelPETS()
obs_len, act_len = (env.observation_space.shape[0], env.action_space.shape[0])
controller = ControllerModel(obs_len, act_len)
dataset = MujocoDataset(samples_per_epoch)
dataset.collect_data(env, RandomController())

obs_mean, obs_std = (
    train_dynamics.normalizer.mean[0, :obs_len],
    train_dynamics.normalizer.std[0, :obs_len]
)
act_mean, act_std = (
    train_dynamics.normalizer.mean[0, obs_len:],
    train_dynamics.normalizer.std[0, obs_len:]
)

# -------- TESTING: print mean and std ---------------
# print("mean and std in thing")
# print(obs_mean)
# print(obs_std)
# print("mean and std in my dataset")
# print(np.mean(dataset.data, axis=0))
# print(np.std(dataset.data, axis=0))
# exit()

trainloader = trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

zero_vec = torch.zeros(batch_size)


def loss_fn(next_obs, act, conversion_factor=1):
    # TODO: rewrite with torch
    loss_ctrl = 0.1 * torch.sum(act**2, axis=1)
    loss_move = conversion_factor - next_obs[:, 9]
    # torch.maximum((conversion_factor - next_obs[:, 0]), zero_vec)
    loss = torch.sum(loss_ctrl + loss_move)
    # Does not work because obs[:, 0] can be negative!
    # loss = (torch.sum(act**2, axis=1) / next_obs[:, 0])
    return loss


optimizer = torch.optim.SGD(
    controller.parameters(), lr=learning_rate_controller, momentum=0.9
)

# fix gradients of dynamics
for param in train_dynamics.dynamics_model.parameters():
    param.requires_grad = False

try:
    loss_sum = []
    rew_list = []
    for epoch in range(nr_epochs):
        # test_list = []
        eval_reward = run_eval(env, controller, 100, 10)
        print("epoch", epoch - 1, "rewards eval", eval_reward)
        rew_list.append(eval_reward)
        losses = []
        for i, obs in enumerate(trainloader):
            # --- testing with env -----
            # obs_init = env.reset()
            # obs = torch.from_numpy(np.expand_dims(obs_init, 0))
            # --- testing with env -----
            optimizer.zero_grad()

            act = controller(obs.float())

            # normalize and feed through dynamics
            normed_obs = ((obs - obs_mean) / obs_std).float()
            normed_act = (act - act_mean) / act_std
            next_state = train_dynamics.forward(obs, normed_obs, normed_act)

            loss = loss_fn(next_state, act)

            loss.backward()
            if epoch % 5 == 0 and i == 10:
                print("act", act[:3])
                print("grad", torch.sum(torch.abs(controller.fc3.weight.grad)))
            optimizer.step()

            losses.append(loss.item())

        loss_sum.append(np.sum(losses))

        # evaluate:
        print()
        print("epoch", epoch, "sum of losses", round(loss_sum[-1], 2))

        if (epoch + 1) % resample_every_x_epochs == 0:
            dataset.collect_data(env, RandomController())

        # print(next_state)

        # --- testing with env -----
        # real_next_obs, real_rew, _, _ = env.step(act.detach().numpy())
        # print(torch.from_numpy(real_next_obs))

        # test_list.append([real_rew, loss.item()])
except KeyboardInterrupt:
    pass
    # import matplotlib.pyplot as plt
    # test_list = np.array(test_list)
    # plt.scatter(test_list[:, 0], test_list[:, 1])
    # plt.show()
    # --- testing with env -----
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(loss_sum, label="loss", color="blue")
ax.set_ylabel('Loss', color="blue", fontsize=18)
ax2 = ax.twinx()
ax2.plot(rew_list, label="rew", color="red")
ax2.set_ylabel('Reward', color="red", fontsize=18)
plt.savefig("trained_models/mujoco/results.png")

torch.save(controller, "trained_models/mujoco/cheetah_model_petsdyn")