import gym
import math
import numpy as np
import torch
import time
from gym import spaces
from gym.utils import seeding

from neural_control.environments.cartpole_env import CartPoleEnv
from neural_control.environments.wing_env import SimpleWingEnv
from neural_control.environments.drone_env import QuadRotorEnvBase
from neural_control.trajectory.q_funcs import project_to_line
from neural_control.dataset import WingDataset, QuadDataset
from neural_control.trajectory.generate_trajectory import (
    load_prepare_trajectory
)
from neural_control.trajectory.random_traj import PolyObject
metadata = {'render.modes': ['human']}

buffer_len = 3
img_width, img_height = (200, 300)
crop_width = 60


class CartPoleEnvRL(gym.Env, CartPoleEnv):

    def __init__(self, dynamics, dt=0.05, **kwargs):
        CartPoleEnv.__init__(self, dynamics, dt=dt)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array(
            [
                self.x_threshold * 2, 20, self.theta_threshold_radians * 2, 20,
                1
            ] * buffer_len
        )
        self.thresh_div = 0.21
        self.obs_dim = len(high)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1, ))
        self.observation_space = spaces.Box(-high, high)
        self.init_buffers()

    def init_buffers(self):
        self.state_buffer = np.zeros((buffer_len, 4))
        self.action_buffer = np.zeros((buffer_len, 1))

    def set_state(self, state):
        self.state = state
        self._state = state

    def get_reward(self):
        survive_reward = 0.1
        angle_reward = .5 * (1.5 - abs(self.state[2]))
        vel_reward = .4 * (1.5 - abs(self.state[1]))
        return survive_reward + angle_reward + vel_reward

    def get_obs(self):
        state_action_history = np.concatenate(
            (self.state_buffer, self.action_buffer), axis=1
        )
        obs = np.reshape(state_action_history, (self.obs_dim))
        return obs

    def step(self, action):
        super()._step(action, is_torch=False)
        # print(self.state)
        done = not self.is_upright() or self.step_ind > 250

        # this reward is positive if theta is smaller 0.1 and else negative
        if not done:
            # training to stay stable with low velocity
            reward = 2.0 - abs(self.state[1])
        else:
            reward = 0.0

        info = {}
        self.step_ind += 1

        # update state buffer with new state
        self.state_buffer = np.roll(self.state_buffer, 1, axis=0)
        self.state_buffer[0] = self.state.copy()
        self.action_buffer = np.roll(self.action_buffer, 1, axis=0)
        self.action_buffer[0] = action.copy()

        self.obs = self.get_obs()

        return self.obs, reward, done, info

    def reset(self):
        super()._reset_upright()
        for i in range(buffer_len):
            self.state_buffer[i] = self.state
        self.action_buffer = np.zeros(self.action_buffer.shape)
        self.step_ind = 0
        return self.get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        self._render(mode=mode)

    def close(self):
        if self.viewer:
            self.viewer.close()
