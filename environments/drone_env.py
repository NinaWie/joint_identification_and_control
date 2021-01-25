"""
Adapted from https://github.com/ngc92/quadgym
"""
import math
import time
import pprint
from types import SimpleNamespace
import numpy as np
import torch

import gym
from gym import spaces
from gym.utils import seeding

from utils.trajectory import straight_training_sample, get_reference
try:
    from .rendering import Renderer, Ground, QuadCopter
    from .copter import copter_params, DynamicsState, Euler
    from .drone_dynamics import simulate_quadrotor, linear_dynamics
except ImportError:
    from rendering import Renderer, Ground, QuadCopter
    from copter import copter_params, DynamicsState, Euler
    from drone_dynamics import simulate_quadrotor, linear_dynamics

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuadRotorEnvBase(gym.Env):
    """
    Simple simulation environment for a drone
    Drone parameters are defined in file copter.py (copter_params dict)
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    action_space = spaces.Box(0, 1, (4, ), dtype=np.float32)
    observation_space = spaces.Box(0, 1, (6, ), dtype=np.float32)

    def __init__(self):

        # set up the renderer
        self.renderer = Renderer()
        self.renderer.add_object(Ground())
        self.renderer.add_object(QuadCopter(self))

        # set to supplied copter params, or to default value
        self.setup = SimpleNamespace(**copter_params)

        # just initialize the state to default, the rest will be done by reset
        self._state = DynamicsState()
        self.random_state = None
        self.seed()
        # added from motor one
        self._error_target = 1 * np.pi / 180
        self._in_target_reward = 0.5
        self._boundary_penalty = 1.0
        # self._attitude_reward = AttitudeReward(
        #     1.0, attitude_error_transform=np.sqrt
        # )
        self.dt = 0.02

    @staticmethod
    def get_is_stable(np_state):
        """
        Return for a given state whether the drone is stable or failure
        Returns bool --> if true then still stable
        """
        # only roll and pitch are constrained
        attitude_condition = np.all(np.absolute(np_state[3:5]) < .8)
        return attitude_condition

    def get_acceleration(self):
        """
        Compute acceleration from current state (pos, vel and att)
        """
        torch_state = torch.from_numpy(np.array([self._state.as_np])
                                       ).to(device)
        acceleration = linear_dynamics(
            torch_state[:, 9:13], torch_state[:, 3:6], torch_state[:, 6:9]
        )
        return acceleration[0]

    def step(self, action):
        """
        Apply action to the current drone state
        Returns:
            New state as np array
            bool indicating whether drone failed
        """
        action = np.clip(self._process_action(action), 0.0, 1.0)
        assert action.shape == (4, ), f"action not size 4 but {action.shape}"

        # set the blade speeds. as F ~ w², and we want F ~ action.
        torch_state = torch.from_numpy(np.array([self._state.as_np])
                                       ).to(device)
        torch_action = torch.from_numpy(np.array([action])).to(device)
        new_state_arr = simulate_quadrotor(
            torch_action, torch_state, dt=self.dt
        )
        numpy_out_state = new_state_arr.cpu().numpy()[0]
        # update internal state
        self._state.from_np(numpy_out_state)
        # attitude = self._state.attitude

        # How this works: Check whether the angle is above the boundaries
        # (threshold), if yes, clip them (in the clip_attitude function) and
        # return whether it needed to be clipped, give penalty on reward if yes
        # if clip_attitude(self._state, np.pi / 4):
        #     reward -= self._boundary_penalty
        # resets the velocity after each step --> we don't want to do that
        # ensure_fixed_position(self._state, 1.0)

        return numpy_out_state, self.get_is_stable(numpy_out_state)

    def render(self, mode='human', close=False):
        if not close:
            self.renderer.setup()

            # update the renderer's center position
            self.renderer.set_center(0)  # self._state.position[0])

        return self.renderer.render(mode, close)

    def close(self):
        self.renderer.close()

    def zero_reset(self, position_x=0, position_y=0, position_z=2):
        """
        Reset to easiest state: zero velocities and attitude and given position
        Arguments:
            Position (in three floats)
        """
        zero_state = np.zeros(20)
        zero_state[9:13] = 400
        zero_state[:3] = [position_x, position_y, position_z]
        self._state.from_np(zero_state)

    def render_reset(self, strength=.8):
        """
        Reset to a random state, but require z position to be at least 1
        """
        self.reset(strength=strength)
        self._state.position[2] += 2

    def reset(self, strength=.8):
        """
        Reset drone to a random state
        """
        self._state = DynamicsState()
        # # possibility 1: reset to zero
        #

        self.randomize_angle(5 * strength)
        self.randomize_angular_velocity(2.0 * strength)
        self._state.attitude.yaw = self.random_state.uniform(low=-1, high=1)
        self._state.position[:3] = np.random.rand(3) * 2 - 1
        self.randomize_rotor_speeds(200, 500)
        # yaw control typically expects slower velocities
        self._state.angular_velocity[2] *= 0.5  # * strength
        self.randomize_velocity(1.3 * strength)

        # self.renderer.set_center(None)

        return self._state

    def get_copter_state(self):
        return self._state

    def _process_action(self, action):
        return action

    # ------------ Functions to randomize the state -----------------------
    # env functions
    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

    # utility functions
    def randomize_angle(self, max_pitch_roll: float):
        self._state._attitude = random_angle(self.random_state, max_pitch_roll)

    def randomize_velocity(self, max_speed: float):
        self._state.velocity[:] = self.random_state.uniform(
            low=-max_speed, high=max_speed, size=(3, )
        )

    def randomize_rotor_speeds(self, min_speed: float, max_speed: float):
        self._state.rotor_speeds[:] = self.random_state.uniform(
            low=min_speed, high=max_speed, size=(4, )
        )

    def randomize_angular_velocity(self, max_speed: float):
        self._state.angular_velocity[:] = self.random_state.uniform(
            low=-max_speed, high=max_speed, size=(3, )
        )

    def randomize_altitude(self, min_: float, max_: float):
        self._state.position[2] = self.random_state.uniform(
            low=min_, high=max_
        )


def random_angle(random_state: np.random.RandomState, max_pitch_roll: float):
    """
    Returns a random Euler angle where roll and pitch are limited to [-max_pitch_roll, max_pitch_roll].
    :param random_state: The random state used to generate the random numbers.
    :param max_pitch_roll: Maximum roll/pitch angle, in degrees.
    :return Euler: A new `Euler` object with randomized angles.
    """
    mpr = max_pitch_roll * math.pi / 180

    # small pitch, roll values, random yaw angle
    roll = random_state.uniform(low=-mpr, high=mpr)
    pitch = random_state.uniform(low=-mpr, high=mpr)
    yaw = random_state.uniform(low=-math.pi, high=math.pi)

    return Euler(roll, pitch, yaw)


# --------------------- Auxilary functions ----------------------


def construct_states(num_data, episode_length=10, reset_strength=1, **kwargs):
    """
    Sample states for training the model
    Arguments:
        num_data: How much states to sample
        episode_length: Maximum number of states before resetting the env
        reset_strength (float between 0.5 - 1.5): How much randomization, i.e.
                How far from target should the states be
    """
    # data = np.load("data.npy")
    # assert not np.any(np.isnan(data))
    const_action_runs = .8
    # return data
    env = QuadRotorEnvBase()
    data = []
    is_stable_list = list()
    while len(data) < num_data:
        env.reset(strength=reset_strength)
        is_stable = True
        time_stable = 0
        # Sample one episode
        while is_stable and time_stable < episode_length:
            # perform random action
            action = np.random.rand(4) * .4 - .2 + .3
            if len(data) > num_data * const_action_runs:
                # add some states with very monotone actions
                action = np.ones(4) * .5
            new_state, is_stable = env.step(action)
            # print(new_state[2])
            data.append(new_state)
            time_stable += 1
        is_stable_list.append(time_stable)
    data = np.array(data)
    # np.save("data_backup/collected_data.npy", data)
    # print("saved first data", np.mean(is_stable_list))
    return data


def trajectory_training_data(
    len_data,
    step_size=0,
    max_drone_dist=0.1,
    ref_length=5,
    reset_strength=1,
    load_selfplay=None,  # "data/jan_2021.npy",
    **kwargs
):
    """
    Generate training dataset for trajectories as input
    Arguments:
        len_data: int, how much data to generate
        step_size: Distance between two points on the trajectory
        max_drone_dist: Maximum distance of the drone from the first state
        ref_length: Number of states sampled from reference traj
        reset_strength: How much to reset the model
        load_selfplay: file path where data from self play is located
    Returns:
        Array of size (len_data, reference_shape) with the training data
    """
    if load_selfplay is not None:
        # load training data from np array
        data = np.load(load_selfplay)
        rand_inds = np.random.permutation(len(data))
        ind_counter = 0

    env = QuadRotorEnvBase()
    drone_states, ref_states = [], []
    for _ in range(len_data):
        if load_selfplay is not None and np.random.rand() < .5:
            drone_state = data[rand_inds[ind_counter]]
            ind_counter += 1
        else:
            env.reset(strength=reset_strength)
            # sample a drone state
            drone_state = env._state.as_np
        pos0, vel0 = (drone_state[:3], drone_state[6:9])
        acc0 = env.get_acceleration().numpy()

        # sample a direction where the next position is located
        norm_vel = np.linalg.norm(vel0)
        # should not diverge too much from velocity direction of drone
        sampled_pos_dir = (
            vel0 + (np.random.rand(3) - .5) * reset_strength * norm_vel
        )
        sampled_pos_dir = sampled_pos_dir / np.linalg.norm(sampled_pos_dir)
        # sample direction where the velocity should show in the end
        sampled_vel_dir = (
            sampled_pos_dir + (np.random.rand(3) - .5) * reset_strength
        )

        # final goal state:
        posf = pos0 + sampled_pos_dir * max_drone_dist
        # multiply by norm to get a velocity of similar strength
        velf = sampled_vel_dir * (norm_vel * (1 + np.random.rand(1) * .2 - .1))

        reference_states = get_reference(
            pos0,
            vel0,
            acc0,
            posf,
            velf,
            ref_length=ref_length,
            delta_t=env.dt
        )

        drone_states.append(drone_state)
        ref_states.append(reference_states)
    drone_states = np.array(drone_states)
    ref_states = np.array(ref_states)
    # np.save("ref_states.npy", ref_states)
    # exit()
    return drone_states, ref_states


def get_avg_distance():
    """
    Get average distance of the states from the target (zero)
    """
    states = construct_states(10000)
    sum_squares = np.sqrt(np.sum(states[:, :3]**2, axis=1))
    print(sum_squares.shape, np.mean(sum_squares))
    return np.mean(sum_squares)


if __name__ == "__main__":
    env = QuadRotorEnvBase()
    env.reset()
    # a1, a2 = trajectory_training_data(1000, step_size=0.1)
    # np.save("drone_states.npy", a1)
    # np.save("ref_states.npy", a2)
    # env = gym.make("QuadrotorStabilizeAttitude-MotorCommands-v0")
    states = construct_states(100)
    # states = np.load("check_added_data.npy")
    print(np.mean(states[:, :6], axis=0))
    print(np.std(states[:, :6], axis=0))
    #  np.load("data_backup/collected_data.npy")
    for j in range(100):
        print([round(s, 2) for s in states[j, :6]])
        env._state.from_np(states[j])
        env.render()
        time.sleep(.1)
