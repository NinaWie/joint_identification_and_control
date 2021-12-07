import numpy as np
from numpy.lib.function_base import flip
import torch
import pickle

import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import torch.nn.functional as F

from mbrl.env.pets_halfcheetah import HalfCheetahEnv
from neural_control.mujoco_utils import (ControllerModel, DynamicsModelPETS)
from scripts.evaluate_mujoco import evaluate_cheetah, run_eval, RandomController


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

    def sample_partly(self, env, controller, how_much=100):
        data = []
        while len(data) < how_much:
            collected_obs = evaluate_cheetah(
                env, controller, nr_steps=100, return_obs=True
            )
            data.extend(collected_obs)
        # roll and replace
        self.data = np.roll(self.data, len(data), axis=0)
        self.data[:len(data)] = data

    def __getitem__(self, index):
        return self.data[index]


load_model = None  # "trained_models/mujoco/cheetah_model_petsdyn_2"
recurrent = True
nr_epochs = 4000
batch_size = 4
samples_per_epoch = 5000
resample_every_x_epochs = 5
learning_rate_controller = 1e-6
thresh_flip = 0.4
grad_clip_val = 8
nr_actions = 15  # increase in several iterations is maybe the best
ctrl_weight, pos_weight, vel_weight, flip_weight = (0.02, 0.3, 2, 10)
# NOTES
# currently using random controller only in the beginning for sampling
# loss function: negative values seem to be fine!
# careful: in evaluate mujoco, obs is currently not normalized!!

env = HalfCheetahEnv()
train_dynamics = DynamicsModelPETS()
obs_len, act_len = (env.observation_space.shape[0], env.action_space.shape[0])
con_actions = 1 if recurrent else nr_actions
controller = ControllerModel(obs_len, act_len, nr_actions=con_actions)
if load_model is not None:
    controller = torch.load(load_model)
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

vec_05 = torch.tensor([thresh_flip**2])


def loss_fn_horizon(obs, act, conversion_factor=1):
    loss_ctrl = torch.sum(act**2)
    loss_move_pos = torch.mean(conversion_factor - obs[:, :, 0])
    loss_move_vel = torch.mean(conversion_factor - obs[:, :, 9])
    loss_flip = torch.sum(obs[:, :, 2]**2)
    # torch.mean(
    #     torch.maximum(obs[:, :, 2]**2, vec_05) - thresh_flip**2
    # )
    # print(
    #     ctrl_weight * loss_ctrl, pos_weight * loss_move_pos,
    #     vel_weight * loss_move_vel, flip_weight * loss_flip
    # )
    return loss_ctrl, loss_move_pos, loss_move_vel, loss_flip
    # return (
    #     ctrl_weight * loss_ctrl + pos_weight * loss_move_pos +
    #     vel_weight * loss_move_vel + flip_weight * loss_flip
    # )


optimizer = torch.optim.SGD(
    controller.parameters(), lr=learning_rate_controller, momentum=0.9
)

# fix gradients of dynamics
for param in train_dynamics.dynamics_model.parameters():
    param.requires_grad = False

try:
    loss_divided = []
    loss_sum = []
    rew_list = []
    for epoch in range(nr_epochs):
        # test_list = []
        controller.eval()
        eval_reward = run_eval(env, controller, 100, 10)
        controller.train()
        print("epoch", epoch - 1, "rewards eval", eval_reward)
        rew_list.append(eval_reward)
        losses = []
        for i, obs in enumerate(trainloader):
            # --- testing with env -----
            # obs_init = env.reset()
            # obs = torch.from_numpy(np.expand_dims(obs_init, 0))
            # --- testing with env -----
            optimizer.zero_grad()

            intermediate_states = torch.zeros(
                obs.size()[0], nr_actions, obs_len
            )
            if recurrent:
                all_actions = torch.zeros(obs.size()[0], nr_actions, act_len)
                for j in range(nr_actions):
                    act = controller(obs.float())

                    # normalize and feed through dynamics
                    normed_act = (act - act_mean) / act_std

                    normed_obs = ((obs - obs_mean) / obs_std).float()
                    obs = train_dynamics.forward(obs, normed_obs, normed_act)
                    # save obs
                    intermediate_states[:, j] = obs
                    all_actions[:, j] = act
            else:
                all_actions = controller(obs.float())

                # normalize and feed through dynamics
                normed_act = (all_actions - act_mean) / act_std

                for j in range(nr_actions):
                    normed_obs = ((obs - obs_mean) / obs_std).float()
                    obs = train_dynamics.forward(
                        obs, normed_obs, normed_act[:, j]
                    )
                    # save obs
                    intermediate_states[:, j] = obs

            (loss_ctrl, loss_move_pos, loss_move_vel,
             loss_flip) = loss_fn_horizon(intermediate_states, act)
            loss = (
                ctrl_weight * loss_ctrl + pos_weight * loss_move_pos +
                vel_weight * loss_move_vel + flip_weight * loss_flip
            )
            # # without horizon:
            # normed_obs = ((obs - obs_mean) / obs_std).float()
            # next_state = train_dynamics.forward(
            #     obs, normed_obs, normed_act[:, 0]
            # )
            # loss = loss_fn(next_state, act[:, 0])

            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                controller.parameters(), max_norm=grad_clip_val
            )
            # if torch.sum(torch.abs(controller.fc3.weight.grad)) > 100:
            #     print(
            #         "gradient problem",
            #         torch.sum(torch.abs(controller.fc3.weight.grad))
            #     )
            if epoch % 5 == 0 and i == 10:
                print("act", act[0, :2])
                print("grad", torch.sum(torch.abs(controller.fc3.weight.grad)))
            optimizer.step()

            losses.append(
                [
                    loss.item(),
                    loss_ctrl.item(),
                    loss_move_pos.item(),
                    loss_move_vel.item(),
                    loss_flip.item()
                ]
            )

        losses = np.array(losses)

        loss_sum.append(np.sum(losses[:, 0]))
        loss_divided.append(np.mean(losses[:, 1:], axis=0))

        # evaluate:
        print()
        print("epoch", epoch, "sum of losses", round(loss_sum[-1], 2))

        # if (epoch + 1) % resample_every_x_epochs == 0:
        #     dataset.collect_data(env, )  # RandomController())
        dataset.sample_partly(env, RandomController(), how_much=400)
        controller.eval()
        dataset.sample_partly(env, controller, how_much=200)

        if epoch % 100 == 0:
            torch.save(
                controller, "trained_models/mujoco/cheetah_model_petsdyn"
            )

        # # show it
        # if (epoch + 1) % 10 == 0:
        #     evaluate_cheetah(env, controller, render=True)
        #     env.close()

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

loss_divided = np.array(loss_divided)

results_dict = {
    "rewards": rew_list,
    "loss": loss_sum,
    "loss_divided": loss_divided
}
with open("trained_models/mujoco/cheetah_results.pkl", "wb") as outfile:
    pickle.dump(results_dict, outfile)