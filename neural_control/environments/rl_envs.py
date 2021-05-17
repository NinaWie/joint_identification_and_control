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
from neural_control.dataset import WingDataset, QuadDataset
from neural_control.trajectory.generate_trajectory import (
    load_prepare_trajectory
)
from neural_control.trajectory.random_traj import PolyObject
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
        if not done:
            # training to stay stable with low velocity
            reward = 3.0 - abs(self.state[1])
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


class QuadEnvRL(gym.Env, QuadRotorEnvBase):

    def __init__(self, dynamics, dt, speed_factor=.2, nr_actions=10, **kwargs):
        self.dt = dt
        self.speed_factor = speed_factor
        self.nr_actions = nr_actions

        QuadRotorEnvBase.__init__(self, dynamics, dt)
        self.action_space = spaces.Box(low=0, high=1, shape=(4, ))

        # state and reference
        self.state_inp_dim = 15
        self.obs_dim = self.state_inp_dim + 10 * 9
        high = np.array([10 for _ in range(self.obs_dim)])
        self.observation_space = spaces.Box(
            -high, high, shape=(self.obs_dim, )
        )

        self.thresh_stable = 1.5
        self.thresh_div = 2

        kwargs["dt"] = dt
        kwargs['speed_factor'] = speed_factor
        kwargs["self_play"] = 0
        self.dataset = QuadDataset(1, **kwargs)

    def prepare_obs(self):
        obs_state, _, obs_ref, _ = self.dataset.prepare_data(
            self.state.copy(),
            self.current_ref[self.current_ind + 1:self.current_ind +
                             self.nr_actions + 1].copy()
        )
        return obs_state, obs_ref

    def state_to_obs(self):
        # get from dataset
        obs_state, obs_ref = self.prepare_obs()
        # flatten obs ref
        obs_ref = obs_ref.reshape((-1, self.obs_dim - self.state_inp_dim))
        # concatenate relative position and observation
        obs = torch.cat((obs_ref, obs_state), dim=1)[0].numpy()
        return obs

    def reset(self, test=0):
        # load random trajectory from train
        self.current_ref = load_prepare_trajectory(
            "data/traj_data_1", self.dt, self.speed_factor, test=test
        )
        self.renderer.add_object(PolyObject(self.current_ref, shift_one=0))

        self.state = np.zeros(12)
        self.state[:3] = self.current_ref[0, :3]
        self._state.from_np(self.state)

        self.current_ind = 0
        self.obs = self.state_to_obs()
        return self.obs

    def get_divergence(self):
        return np.linalg.norm(
            self.current_ref[self.current_ind, :3] - self.state[:3]
        )

    def get_reward(self, action):
        """
        MPC type cost function turned into reward
        """
        pos_factor = 10
        u_thrust_factor = 5
        u_rates_factor = 0.1
        av_factor = 0.1
        vel_factor = 1

        pos_div = np.linalg.norm(
            self.current_ref[self.current_ind, :3] - self.state[:3]
        )
        pos_rew = self.thresh_div - pos_div
        vel_div = np.linalg.norm(
            self.current_ref[self.current_ind, 6:9] - self.state[6:9]
        )
        vel_rew = self.thresh_div - vel_div  # How high is velocity diff?
        u_rew = .25 - (.5 - action)**2
        # have to use abs because otherwise not comparable to thresh div
        av_rew = np.sum(self.thresh_stable - (np.absolute(self.state[9:12])))

        # print(action)
        # print(pos_rew)
        # print(vel_rew)
        # print(av_rew)
        # print(u_rew)
        # print()
        reward = (
            pos_factor * pos_rew + vel_factor * vel_rew + av_factor * av_rew +
            u_rates_factor * np.sum(u_rew[1:]) + u_thrust_factor * u_rew[0]
        )

        return reward

    def step(self, action):
        self.state, is_stable = QuadRotorEnvBase.step(
            self, action, thresh=self.thresh_stable
        )
        self.obs = self.state_to_obs()
        self.current_ind += 1

        pos_div = self.get_divergence()

        done = (
            (not is_stable) or pos_div > self.thresh_div
            or self.current_ind > len(self.current_ref) - self.nr_actions - 1
        )

        reward = 0
        if not done:
            reward = self.get_reward(action)
        info = {}
        # print()
        # np.set_printoptions(precision=3, suppress=1)
        # # print(self.current_ref[:3, :3])
        # print(
        #     self.current_ind, self.state[:3],
        #     self.current_ref[self.current_ind, :3]
        # )
        # print(self.state)
        # print(self.obs.shape)
        # print(div, reward)
        return self.obs, reward, done, info

    def render(self, mode="human"):
        QuadRotorEnvBase.render(self, mode=mode)


class WingEnvRL(gym.Env, SimpleWingEnv):

    def __init__(self, dynamics, dt, **kwargs):
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
        self.dataset = WingDataset(0, dt=self.dt, **kwargs)

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

    def reset(self, x_dist=50, x_std=5):
        rand_y, rand_z = tuple((np.random.rand(2) - .5) * 2 * x_std)
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
