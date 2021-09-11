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

    def __init__(self, normalize=True, load_mean_path=None):
        self.normalize = normalize
        states_x, _, _ = collect_episode(nr_data=3000)
        if self.normalize:
            if load_mean_path:
                # print("Loading mean and std from files..")
                self.mean = torch.from_numpy(np.load(load_mean_path))
                self.std = torch.from_numpy(
                    np.load(load_mean_path.replace("mean", "std"))
                )
            else:
                self.mean = torch.mean(states_x, dim=0)
                self.std = torch.std(states_x, dim=0)
            # print("Mean and std", self.mean.size())

        self.collect_new_data(DummyController())
        # TODO: normalize? - but difference is entscheidend!
        # self.mean = torch.mean(self.states_x, dim=0)
        # print("means", self.mean.size())
        # self.std = torch.std(self.states_x, dim=0)

    def normalize_data(self, states):
        return (states - self.mean) / self.std

    def denormalize_data(self, states):
        return states * self.std + self.mean

    def collect_new_data(self, model=DummyController()):
        with torch.no_grad():
            states_x, self.actions, states_y = collect_episode(model=model)
        if self.normalize:
            self.states_x = self.normalize_data(states_x)
            self.states_y = self.normalize_data(states_y)
        else:
            self.states_x = states_x
            self.states_y = states_y

        # if self.mean is None:
        #     states_x, actions, states_y =

    def __len__(self):
        return len(self.states_x)

    def __getitem__(self, index):
        return self.states_x[index], self.actions[index], self.states_y[index]


class LearntDynamicsNew(nn.Module):

    def __init__(
        self,
        state_size,
        action_size,
        out_state_size=None,
    ):
        super(LearntDynamicsNew, self).__init__()
        if out_state_size is None:
            out_state_size = state_size

        self.lin1 = nn.Linear(state_size + act_size, 128)
        self.lin2 = nn.Linear(128, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, out_state_size)

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        x1 = torch.relu(self.lin1(state_action))
        x2 = torch.relu(self.lin2(x1))
        x3 = torch.relu(self.lin3(x2))
        delta_state = torch.tanh(self.lin4(x3)) * 3
        # delta_state[:, -1] *= 0  # last one is always zero # TODO reacher
        return state + delta_state


# class LearntDynamicsNew(LearntDynamicsMPC):
# def forward(self, state, action):
#     if self.transform_action:
#         action = torch.matmul(self.linear_at, torch.unsqueeze(action,
#                                                               2))[:, :, 0]
#     # run through residual network delta
#     added_new_state = self.state_transformer(state, action)
#     return torch.tanh(added_new_state)  # state +  #TODO


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


def collect_episode(model=None, nr_data=1000):
    if model is None:
        model = DummyController()
    count = 0
    states_before, actions, states_after = list(), list(), list()
    # run episodes until we got enough data
    while len(states_before) < nr_data:
        obs = env.reset()
        done = False
        while not done:
            # predict random action
            action = model(obs)
            states_before.append(obs.copy())
            actions.append(action)

            prev_dist = np.sum(obs[-3:]**2) * 100

            obs, rew, done, _ = env.step(action)

            after_dist = np.sum(obs[-3:]**2) * 100
            # env.render()
            # time.sleep(0.5)
            # print(round(prev_dist - after_dist, 2))

            # We want to predict only how much closer this action brought us
            # want to maximize this, so we want to minimize 2- this
            states_after.append(obs.copy())
            # rew)  # prev_dist - after_dist)
            # [np.sum(obs[-3:]**2)])

            count += 1
    return list_to_torch(states_before), list_to_torch(actions), list_to_torch(
        states_after
    )


def test_dynamics(
    dynamics_model,
    con_model=DummyController(),
    render=False,
    normalize_dataset=None
):
    obs_gt = env.reset()
    done = False
    count = 0
    actu_list, pred_list = [], []
    mse = []
    avg_pred = []
    with torch.no_grad():
        while count < 3:
            action = con_model(obs_gt)  # np.random.rand(act_size) * 2 - 1

            state_before = list_to_torch([obs_gt.copy()])
            # print()
            # print("state before", state_before)
            if normalize_dataset:
                state_before = normalize_dataset.normalize_data(state_before)
            # print("state before normed", state_before)
            obs_pred = dynamics_model(state_before, list_to_torch([action]))
            # print("obs pred", obs_pred)
            if normalize_dataset:
                obs_pred = normalize_dataset.denormalize_data(obs_pred)
            # print("obs pred denormed", obs_pred)
            # print(
            #     "Loss:", loss_reacher(list_to_torch([state_before]), obs_pred)
            # )
            obs_pred = obs_pred.numpy()[0]

            # prev_dist = np.sum(state_before[-3:]**2) * 100
            obs_gt, rew_gt, done, info = env.step(action)
            # after_dist = np.sum(obs_gt[-3:]**2) * 100

            # rew_gt = prev_dist - after_dist

            # save outputs
            mse.append(np.sum((obs_gt - obs_pred)**2))
            avg_pred.append(obs_pred)

            # store the differences
            diff_to_bef = np.sum((state_before.numpy()[0] - obs_gt)**2)
            diff_to_gt = np.sum((obs_pred - obs_gt)**2)
            # print("actu", rew_gt)  # prev_dist - after_dist)
            actu_list.append(diff_to_bef)
            # obs_gt[-3:])  # prev_dist - after_dist)
            pred_list.append(diff_to_gt)

            if render:
                env.render()
                time.sleep(.5)
                # print(np.around(state_before, 2))
                # print(np.around(obs_gt[-3:], 2))
                # print(np.around(obs_pred, 2)) # obs_pred[-3:])
                # # print("actu", after_dist - prev_dist)
                # print("pred", rew_pred)
                # print(np.around(np.sum(obs_gt[-3:]**2), 4))
                # print(np.around(state_before, 2)[-3:-1])
                print(np.around(state_before, 2)[-3:-1])
                print(np.around(obs_gt, 2)[-3:-1])
                print(np.around(obs_pred, 2)[-3:-1])
                print()
                # print()
            count += 1
    print(
        "Test dynamics: mean diff to bef", np.mean(actu_list), "mean error",
        np.mean(pred_list)
    )
    # if render:
    #     plt.scatter(actu_list, pred_list)
    #     plt.show()
    # print(
    #     "Average predicted reward and std", np.mean(avg_pred),
    #     np.std(avg_pred)
    # )
    # print("Average mse", np.mean(mse))


def train_dynamics(out_path, load_model=None, nr_epochs=3000):
    loaded_model = PPOWrapper()

    dynamics_model = LearntDynamicsNew(obs_size, act_size)
    # POSSIBLY LOAD MODEL and mean and std with which it was trained
    load_mean = None
    if load_model:
        dynamics_model = torch.load(load_model)
        load_mean = load_model + "_mean.npy"

    dataset = MujocoDataset(normalize=True, load_mean_path=load_mean)
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0
    )
    optimizer = torch.optim.SGD(dynamics_model.parameters(), lr=1e-6)

    for epoch in range(nr_epochs):
        epoch_loss = 0
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()

            (state, act, state_out) = data
            state_pred = dynamics_model(state, act)

            # if i == 0:
            # print(state[0])
            # print(act[0])
            # print(state_pred[0], state_out[0])

            loss = torch.sum((state_pred - state_out)**2)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} loss {round(epoch_loss / i, 2)}")
            dataset.collect_new_data(loaded_model)
            test_dynamics(dynamics_model, normalize_dataset=dataset)

    np.save(out_path + "_mean.npy", dataset.mean)
    np.save(out_path + "_std.npy", dataset.std)
    torch.save(dynamics_model, out_path)


def evaluate_render(controller):
    obs = env.reset()
    count = 0
    reward_sum = 0
    with torch.no_grad():
        while count < 1000:
            action = controller(obs)
            obs, rew, _, _ = env.step(action)
            # print(np.around(obs, 2), obs[7])

            env.render()
            time.sleep(0.1)
            reward_sum += rew

            count += 1
        print("In real env: sum of rewards", round(reward_sum, 2))
    return reward_sum


def evaluate(controller, render=False, nr_iters=1000):
    if render:
        return evaluate_render(controller)

    with torch.no_grad():
        x_pos_change = 0
        reward_sum = 0

        for i in range(nr_iters):
            if i % 50 == 0:
                obs = env.reset()

            x_pos = (obs[-3:])**2

            action = controller(obs)
            obs, rew, done, _ = env.step(action)

            reward_sum += rew
            x_pos_change += np.sqrt(np.sum(obs[-3:]**2))

        print("distance in real", x_pos_change / nr_iters * 50)
        # print("In real env: sum of rewards", round(reward_sum, 2))
        #, "x", x_pos_change)
    return reward_sum


def eval_in_trained_dyn(controller, dynamics, nr_actions=3, episode_len=500):
    count = 0

    with torch.no_grad():
        x_pos_change = 0

        while count < episode_len:
            if count % 50 == 0:
                # print("-----------------------------\n")
                obs = env.reset()
                obs = list_to_torch([obs])

            x_pos = (obs[0, -3:])**2
            # print(x_pos[-3:-1])
            # predict action
            action_seq = controller(obs)
            action_seq = torch.reshape(action_seq, (-1, nr_actions, act_size))
            action = action_seq[:, 0]

            # compare to random action
            # action = torch.randn(action.size())
            # pass through dynamics
            obs = dynamics(obs, action)
            # print(obs)

            # distance afterwards - distance before --> should be low
            x_pos_change += torch.sqrt(torch.sum(obs[0, -3:]**2)).item()

            count += 1
        print(
            "distance in trained env: ",
            round(x_pos_change / episode_len * 50, 2)
        )
    return x_pos_change / episode_len * 50


def loss_cheetah(in_obs, out_obs):
    # assume maximal 2 erreichbar von x und x new difference
    return 2 - torch.mean((out_obs[:, 0] - in_obs[:, 0]))


def loss_reacher(obs_list, reference):
    return torch.sum((obs_list - reference)**2)
    # start_diff = torch.sum(in_obs[:, -3:]**2)
    # end_diff = torch.sum(out_obs[:, -3:]**2)
    # return 2 - (start_diff - end_diff)


def set_not_trainable(model):
    model.trainable = False
    for param in model.parameters():
        param.requires_grad = False
    return model


def set_trainable(model):
    model.trainable = True
    for param in model.parameters():
        param.requires_grad = True
    return model


class PPOWrapper:

    def __init__(self) -> None:
        self.model = torch.load("../ppo_mujoco/trained_model")

    def __call__(self, obs):
        a = self.model.choose_action(
            torch.from_numpy(np.array(obs).astype(np.float32)).unsqueeze(0)
        )[0]
        return a


def train_controller(
    dynamics_model,
    out_path,
    controller_model=None,
    nr_epochs=1000,
    nr_actions=3,
    batch_size=8,
    controller_loss=loss_cheetah
):
    if controller_model is None:
        controller_model = ControllerNet(obs_size, act_size * nr_actions)
    optimizer_controller = torch.optim.Adam(
        controller_model.parameters(), lr=0.0000001
    )
    optimizer_dynamics = torch.optim.Adam(
        dynamics_model.parameters(), lr=0.0001
    )

    dataset = MujocoDataset()
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    train_model = "controller"
    set_not_trainable(dynamics_model)

    best_rew = -500
    con_loss, dyn_loss, rewards = [], [], []

    for epoch in range(nr_epochs):
        epoch_loss = 0
        # if epoch < 200:
        # train_model = "dynamics"
        # TODO: atm only train the controller
        # set_trainable(dynamics_model)
        train_model = "controller"
        for i, data in enumerate(trainloader):
            if train_model == "controller":
                optimizer_controller.zero_grad()

                (in_state, _, _) = data
                state = in_state.clone()

                # design reference
                dist_normed = in_state[:, -3:-1]
                use_step = torch.minimum(
                    torch.abs(dist_normed),
                    # TODO: Assumes that we can move 0.02 per step in each dir
                    torch.ones(dist_normed.size()) * 0.02 * nr_actions
                ) / nr_actions * torch.sign(dist_normed)
                reference = torch.zeros(in_state.size()[0], nr_actions, 2)
                for ind in range(nr_actions):
                    reference[:,
                              ind] = in_state[:, -3:-1] - use_step * (ind + 1)

                act = controller_model(state)

                collect_obs = torch.zeros(in_state.size()[0], nr_actions, 2)
                action_seq = torch.reshape(act, (-1, nr_actions, act_size))
                for act_ind in range(nr_actions):
                    state = dynamics_model(state, action_seq[:, act_ind, :])
                    collect_obs[:, act_ind] = state[:, -3:-1]
                loss = controller_loss(collect_obs, reference)  # TODO
                loss.backward()
                optimizer_controller.step()
            else:
                optimizer_dynamics.zero_grad()

                (state, act, state_out) = data
                state_pred = dynamics_model(state, act)

                loss = torch.sum((state_pred - state_out)**2 / 8)

                loss.backward()
                optimizer_dynamics.step()

            epoch_loss += loss.item()

        if epoch % 20 == 0:
            wrapped_model = ControllerWrapper(controller_model, nr_actions)
            print()
            print(f"Epoch {epoch} trained ", train_model)
            print(f"Loss: {round(epoch_loss / i * 100, 2)}")
            if train_model == "dynamics":
                set_not_trainable(dynamics_model)
                set_trainable(controller_model)
                train_model = "controller"
                dyn_loss.append(epoch_loss / i)
                test_dynamics(dynamics_model)
                # print("with controller:")
                # test_dynamics(dynamics_model, con_model=wrapped_model)
            elif train_model == "controller":
                # TODO: for iterative training
                # set_not_trainable(controller_model)
                # set_trainable(dynamics_model)
                # train_model = "dynamics"
                con_loss.append(epoch_loss / i)
                # reward_sum = evaluate(wrapped_model, render=False)
                reward_sum = eval_in_trained_dyn(
                    controller_model, dynamics_model, nr_actions=nr_actions
                )
                rewards.append(reward_sum)
                if reward_sum > best_rew:
                    best_rew = reward_sum
                    torch.save(controller_model, out_path + "controller")
                    torch.save(dynamics_model, out_path + "dynamics")
                    print("Saved models, best episode")

                dataset.collect_new_data()
                if epoch % 40 == 0:
                    # every now and then, collect data with the controller
                    dataset.collect_new_data(wrapped_model)

    torch.save(controller_model, out_path + "controller_final")
    torch.save(dynamics_model, out_path + "dynamics_final")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(con_loss)
    plt.title("controller loss")
    plt.subplot(1, 3, 2)
    plt.plot(dyn_loss)
    plt.title("dynamics loss")
    plt.subplot(1, 3, 3)
    plt.plot(rewards)
    plt.title("rewards")
    plt.savefig("out_plot.png")
    plt.show()


env_dict = {"reacher": "Reacher-v1", "cheetah": 'HalfCheetah-v1'}
loss_dict = {"cheetah": loss_cheetah, "reacher": loss_reacher}

if __name__ == "__main__":
    nr_actions = 3
    env_name = "cheetah"
    # env = DummyEnv()
    env = gym.make(env_dict[env_name])

    obs_size, act_size = len(env.observation_space.high
                             ), len(env.action_space.high)
    print("state sizes", obs_size, act_size)
    print(env.action_space.low, env.action_space.high)
    dynamics_path = "trained_models/mujoco/dynamics_" + env_name
    out_path = "trained_models/mujoco/" + env_name

    train_dynamics(
        nr_epochs=4000,
        load_model=dynamics_path + "_2",
        out_path=dynamics_path + "_3"
    )

    # dynamics_model = torch.load(out_path + "dynamics_final")
    # controller_model = torch.load(out_path + "controller_final")
    # eval_in_trained_dyn(controller_model, dynamics_model)
    # evaluate(
    #     # DummyController(),
    #     ControllerWrapper(controller_model, nr_actions=nr_actions),
    #     render=False
    # )
    # exit()
    # train from scatch
    # dynamics_model = LearntDynamicsNew(
    #     obs_size, act_size, out_state_size=1
    # )  # only distance as output

    # dynamics_model = torch.load(out_path + "dynamics_final")
    # dynamics_model = torch.load(dynamics_path)
    # test_dynamics(
    #     dynamics_model,
    #     # con_model=ControllerWrapper(controller_model, nr_actions=3),
    #     render=True
    # )

    # controller_model = None  # torch.load(
    #     "trained_models/mujoco/working_a_bit/reachercontroller"
    # )
    # train_controller(
    #     dynamics_model,
    #     out_path,
    #     controller_model=controller_model,
    #     nr_epochs=20000,
    #     nr_actions=nr_actions,
    #     controller_loss=loss_dict[env_name]
    # )
