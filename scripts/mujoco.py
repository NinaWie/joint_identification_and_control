import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import gym
from gym import spaces
import matplotlib.pyplot as plt

from neural_control.dynamics.learnt_dynamics import LearntDynamicsMPC


# ENV TO USE
class DummyEnv(gym.Env):

    def __init__(self) -> None:
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, ))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2, ))

    def reset(self):
        self.obs = np.random.rand(2) * 2 - 1
        self.counter = 0
        return self.obs

    def step(self, action):
        self.counter += 1
        self.obs = self.obs + action
        return self.obs, action[0], self.counter > 200, {}


def list_to_torch(in_list):
    return torch.from_numpy(np.array(in_list)).float()


class DummyController:

    def __call__(self, inp):
        return np.random.rand(act_size) * 2 - 1


class MujocoDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.collect_new_data(DummyController())

    def collect_new_data(self, model=DummyController()):
        with torch.no_grad():
            self.states_x, self.actions, self.states_y = collect_episode(
                model=model
            )

    def __len__(self):
        return len(self.states_x)

    def __getitem__(self, index):
        return self.states_x[index], self.actions[index], self.states_y[index]


class LearntDynamicsNew(LearntDynamicsMPC):

    def forward(self, state, action):
        if self.transform_action:
            action = torch.matmul(self.linear_at, torch.unsqueeze(action,
                                                                  2))[:, :, 0]
        # run through residual network delta
        added_new_state = self.state_transformer(state, action)
        return state + added_new_state


class ControllerWrapper:

    def __init__(self, model, nr_actions):
        self.model = model
        self.nr_actions = nr_actions

    def __call__(self, obs):
        inp_torch = list_to_torch([obs])
        act_pred = self.model(inp_torch)
        first_act_pred = torch.reshape(
            act_pred, (-1, self.nr_actions, act_size)
        )
        return first_act_pred[0, 0].numpy()


class ControllerNet(nn.Module):

    def __init__(self, in_size, out_size):
        super(ControllerNet, self).__init__()
        # conf: in channels, out channels, kernel size
        self.fc0 = nn.Linear(in_size, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, out_size)

    def forward(self, x):
        x = torch.tanh(self.fc0(x))
        # x = x * torch.from_numpy(np.array([0, 1, 1, 1]))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc_out(x))
        return x


def collect_episode(model=None):
    # TODO: use model itself
    if model is None:
        model = DummyController()

    print("collect data")
    obs = env.reset()
    done = False
    count = 0
    states_before, actions, states_after = list(), list(), list()
    while not done:
        # predict random action
        action = model(obs)
        states_before.append(obs)
        actions.append(action)

        obs, _, done, _ = env.step(action)

        states_after.append(obs)

        count += 1
    return list_to_torch(states_before), list_to_torch(actions), list_to_torch(
        states_after
    )


def test_dynamics(dynamics_model):
    obs_gt = env.reset()
    done = False
    count = 0
    with torch.no_grad():
        while count < 100:
            action = np.random.rand(act_size) * 2 - 1
            state_before = obs_gt.copy()

            obs_gt, rew, done, info = env.step(action)
            env.render()
            time.sleep(0.5)
            obs_pred = dynamics_model(
                list_to_torch([state_before]), list_to_torch([action])
            )
            # print(np.around(np.sum(obs_gt[-3:]**2), 4))
            print(np.around(state_before, 2)[-3:-1])
            print(np.around(obs_gt, 2)[-3:-1])
            print(np.around(obs_pred[0].numpy(), 2)[-3:-1])
            print()
            count += 1


def train_dynamics(out_path, nr_epochs=3000):
    dynamics_model = LearntDynamicsNew(obs_size, act_size)
    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=0.0001)

    dataset = MujocoDataset()
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0
    )

    for epoch in range(nr_epochs):
        epoch_loss = 0
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()

            (state, act, state_out) = data
            state_pred = dynamics_model(state, act)

            loss = torch.sum((state_pred - state_out)**2)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} loss {round(epoch_loss / i, 2)}")
            dataset.collect_new_data()

    torch.save(dynamics_model, out_path)


def evaluate(controller):
    obs = env.reset()
    done = False
    count = 0

    with torch.no_grad():
        x_pos_change = 0
        reward_sum = 0

        while not done:  #  and count < 20:
            x_pos = obs[0]
            action = controller(obs)
            obs, rew, done, _ = env.step(action)

            reward_sum += rew
            x_pos_change += (obs[0] - x_pos)

            count += 1
        print("In real env: sum of rewards", round(reward_sum, 2))
        #, "x", x_pos_change)
    return reward_sum


def eval_in_trained_dyn(controller, dynamics, nr_actions=3, episode_len=1000):
    obs = env.reset()
    done = False
    count = 0

    obs = list_to_torch([obs])

    with torch.no_grad():
        x_pos_change = 0

        while count < episode_len:
            x_pos = obs[0, 0]
            # predict action
            action_seq = controller(obs)
            action_seq = torch.reshape(action_seq, (-1, nr_actions, act_size))
            action = action_seq[:, 0]
            # pass through dynamics
            obs = dynamics(obs, action)

            x_pos_change += torch.sum(obs[0, -3:]**2).item()
            # x_pos_change += (obs[0, 0] - x_pos)

            count += 1
        print("In trained env: should decrease: ", round(x_pos_change, 2))


def loss_cheetah(in_obs, out_obs):
    # assume maximal 2 erreichbar von x und x new difference
    return 2 - torch.mean((out_obs[:, 0] - in_obs[:, 0]))


def loss_reacher(in_obs, out_obs):
    return torch.sum(out_obs[:, -3:]**2)


def train_controller(
    dynamics_model,
    out_path,
    nr_epochs=1000,
    nr_actions=3,
    controller_loss=loss_cheetah
):
    controller_model = ControllerNet(obs_size, act_size * nr_actions)
    optimizer_controller = torch.optim.Adam(
        controller_model.parameters(), lr=0.0001
    )
    optimizer_dynamics = torch.optim.Adam(
        dynamics_model.parameters(), lr=0.0001
    )

    dataset = MujocoDataset()
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0
    )
    train_model = "controller"

    best_rew = -500
    con_loss, dyn_loss, rewards = [], [], []

    for epoch in range(nr_epochs):
        epoch_loss = 0
        for i, data in enumerate(trainloader):
            if train_model == "controller":
                optimizer_controller.zero_grad()

                (in_state, _, _) = data
                state = in_state.clone()

                act = controller_model(state)

                action_seq = torch.reshape(act, (-1, nr_actions, act_size))

                for act_ind in range(nr_actions):
                    state = dynamics_model(state, action_seq[:, act_ind, :])

                loss = controller_loss(in_state, state)

                loss.backward()
                optimizer_controller.step()
            else:
                optimizer_dynamics.zero_grad()

                (state, act, state_out) = data
                state_pred = dynamics_model(state, act)

                loss = torch.sum((state_pred - state_out)**2)

                loss.backward()
                optimizer_dynamics.step()

            epoch_loss += loss.item()

        if epoch % 20 == 0:
            print()
            print(f"Epoch {epoch} trained ", train_model)
            print(f"Loss: {round(epoch_loss / i, 2)}")
            if train_model == "dynamics":
                train_model = "controller"
                dyn_loss.append(epoch_loss / i)
            elif train_model == "controller":
                train_model = "dynamics"
                con_loss.append(epoch_loss / i)
                wrapped_model = ControllerWrapper(controller_model, nr_actions)
                reward_sum = evaluate(wrapped_model)
                rewards.append(reward_sum)
                eval_in_trained_dyn(
                    controller_model, dynamics_model, nr_actions=nr_actions
                )
                if reward_sum > best_rew:
                    best_rew = reward_sum
                    torch.save(controller_model, out_path + "controller")
                    torch.save(dynamics_model, out_path + "dynamics")
                    print("Saved models, best episode")

                    dataset.collect_new_data(wrapped_model)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.plot(con_loss)
    plt.title("controller loss")
    plt.subplot(1, 3, 2)
    plt.plot(dyn_loss)
    plt.title("dynamics loss")
    plt.subplot(1, 3, 3)
    plt.plot(rewards)
    plt.title("rewards")
    plt.show()


env_dict = {"reacher": "Reacher-v1", "cheetah": 'HalfCheetah-v1'}
loss_dict = {"cheetah": loss_cheetah, "reacher": loss_reacher}

if __name__ == "__main__":
    nr_actions = 1
    env_name = "reacher"
    # env = DummyEnv()
    env = gym.make(env_dict[env_name])

    obs_size, act_size = len(env.observation_space.high
                             ), len(env.action_space.high)
    print("state sizes", obs_size, act_size)
    print(env.action_space.low, env.action_space.high)
    dynamics_path = "trained_models/mujoco/dynamics_" + env_name
    out_path = "trained_models/mujoco/" + env_name
    # train_dynamics(nr_epochs=3000, out_path=dynamics_path)

    dynamics_model = torch.load(dynamics_path)
    # test_dynamics(dynamics_model)
    train_controller(
        dynamics_model,
        out_path,
        nr_epochs=4000,
        nr_actions=nr_actions,
        controller_loss=loss_dict[env_name]
    )
