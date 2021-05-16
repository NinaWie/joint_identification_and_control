import gym
import math
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding

from neural_control.environments.cartpole_env import CartPoleEnv
from neural_control.environments.wing_env import SimpleWingEnv
from neural_control.environments.drone_env import QuadRotorEnvBase
from neural_control.trajectory.q_funcs import project_to_line
from neural_control.dataset import WingDataset
metadata = {'render.modes': ['human']}


class CartPoleEnvRL(gym.Env, CartPoleEnv):

    def __init__(self):
        pass

    def __init__(self, dynamics, dt=0.05):
        CartPoleEnv.__init__(self, dynamics, dt=dt)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array(
            [self.x_threshold * 2, 20, self.theta_threshold_radians * 2, 20]
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(1, ))
        self.observation_space = spaces.Box(-high, high)

    def step(self, action):
        super()._step(action, is_torch=False)
        done = not self.is_upright()
        # this reward is positive if theta is smaller 0.1 and else negative
        # TODO: would need to include velocity for mismatch business
        # reward = 0.1 - abs(self.state[2])
        if not done:
            reward = 1.0 - abs(action[0])  # subtract velocity from reward
        else:
            reward = 0.0

        info = {}
        return self.state, reward, done, info

    def reset(self):
        super()._reset_upright()
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        self._render(mode=mode)

    def close(self):
        if self.viewer:
            self.viewer.close()


# class QuadEnvRL(gym.Env, QuadRotorEnvBase):

#     def __init__(self, dynamics, dt):
#         QuadRotorEnvBase.__init__(self, dynamics, dt)
#         # TODO
#         pass


class WingEnvRL(gym.Env, SimpleWingEnv):

    def __init__(self, dynamics, dt):
        SimpleWingEnv.__init__(self, dynamics, dt)
        self.action_space = spaces.Box(low=0, high=1, shape=(4, ))

        obs_dim = 12
        # high = np.array([20 for k in range(obs_dim)])
        high = np.array([20, 20, 20, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        self.observation_space = spaces.Box(-high, high, shape=(obs_dim, ))
        # Observations could be what we use as input data to my NN

        # thresholds for done (?)
        self.thresh_stable = .5
        self.thresh_div = 4

        # for making observation:
        self.dataset = WingDataset(0, dt=self.dt)

    def done(self):
        # x is greater
        passed = self.state[0] > self.target_point[0]
        # drone unstable
        unstable = np.any(np.absolute(self._state[6:8]) >= self.thresh_stable)
        return unstable or passed

    def prepare_obs(self):
        obs_state, _, obs_ref, _ = self.dataset.prepare_data(
            self.state, self.target_point
        )
        return obs_state, obs_ref

    def state_to_obs(self):
        # get from dataset
        obs_state, obs_ref = self.prepare_obs()
        # concatenate relative position and observation
        obs = torch.cat((obs_ref, obs_state), dim=1)[0].numpy()
        return obs

    def reset(self, x_dist=20, x_std=3):
        rand_y, rand_z = tuple((np.random.rand(2) - .5) * x_std)
        self.target_point = np.array([x_dist, rand_y, rand_z])
        self.zero_reset()
        self.state = self._state
        self.obs = self.state_to_obs()

        self.drone_render_object.set_target([self.target_point])
        return self.obs

    def get_divergence(self):
        drone_on_line = project_to_line(
            np.zeros(3), self.target_point, self.state[:3]
        )
        div = np.linalg.norm(drone_on_line - self.state[:3])
        return div

    def step(self, action):
        self.state, _ = SimpleWingEnv.step(self, action)
        self.obs = self.state_to_obs()

        div = self.get_divergence()

        done = self.done() or div > self.thresh_div

        if not done:
            reward = self.thresh_div - div
        else:
            reward = 0
        info = {}

        # print()
        # np.set_printoptions(precision=3, suppress=1)
        # print(self.state)
        # print(self.obs)
        # print(div, reward)

        return self.obs, reward, done, info

    def render(self, mode="human"):
        SimpleWingEnv.render(self, mode=mode)
