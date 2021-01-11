import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from mpc_dynamics import *

#
class Quadrotor_v0(object):
    #
    def __init__(self, dt):
        #
        self._state = np.zeros(shape=s_dim)
        self._state[kQuatW] = 1.0
        #
        self._actions = np.zeros(shape=a_dim)

        #
        self._dt = dt
        self._arm_l = 0.3   # m
        
        # Sampling range of the quadrotor's initial position
        self._xyz_dist = np.array(
            [ [-3.0, -1.0], # x 
              [-2.0, 2.],   # y
              [0.0, 2.5]]   # z
        )
        # Sampling range of the quadrotor's initial velocity
        self._vxyz_dist = np.array(
            [ [-1.0, 1.0],  # vx
              [-1.0, 1.0],  # vy
              [-1.0, 1.0]]  # vz
        )
        
        # x, y, z, r, p, y, vx, vy, vz
        self.obs_low = np.array([-10, -10, -10, -np.pi, -np.pi, -np.pi, -10, -10, -10])
        self.obs_high = np.array([10, 10, 10, np.pi, np.pi, np.pi, 10, 10, 10])
        #
        self.reset()
        # self._t = 0.0
    
    def reset(self):
        self._state = np.zeros(shape=s_dim)
        self._state[kQuatW] = 1.0 # 
        #
        # initialize position, randomly
        self._state[kPosX] = np.random.uniform(
            low=self._xyz_dist[0, 0], high=self._xyz_dist[0, 1])
        self._state[kPosY] = np.random.uniform(
            low=self._xyz_dist[1, 0], high=self._xyz_dist[1, 1])
        self._state[kPosZ] = np.random.uniform(
            low=self._xyz_dist[2, 0], high=self._xyz_dist[2, 1])
        
        # initialize rotation, randomly
        quad_quat0 = np.random.uniform(low=0.0, high=1, size=4)
        # normalize the quaternion
        self._state[kQuatW:kQuatZ+1] = quad_quat0 / np.linalg.norm(quad_quat0)
        
        # initialize velocity, randomly
        self._state[kVelX] = np.random.uniform(
            low=self._vxyz_dist[0, 0], high=self._vxyz_dist[0, 1])
        self._state[kVelY] = np.random.uniform(
            low=self._vxyz_dist[1, 0], high=self._vxyz_dist[1, 1])
        self._state[kVelZ] = np.random.uniform(
            low=self._vxyz_dist[2, 0], high=self._vxyz_dist[2, 1])
        #
        return self._state

    def is_stable(self):
        return self._state[2] > 0 

    def step(self, action):
        """
        Set the vehicle's state
        """
        torch_state, torch_action = (torch.from_numpy(self._state).unsqueeze(0), torch.from_numpy(action))
        next_state = dynamics(torch_state, torch_action, self._dt)
        self._state = next_state.numpy()[0]
        return self._state

    def get_state(self):
        """
        Get the vehicle's state
        """
        return self._state

    def get_cartesian_state(self):
        """
        Get the Full state in Cartesian coordinates
        """
        cartesian_state = np.zeros(shape=9)
        cartesian_state[0:3] = self.get_position()
        cartesian_state[3:6] = self.get_euler()
        cartesian_state[6:9] = self.get_velocity()
        return cartesian_state
    
    def get_position(self,):
        """
        Retrieve Position
        """
        return self._state[kPosX:kPosZ+1]
    
    def get_velocity(self,):
        """
        Retrieve Linear Velocity
        """
        return self._state[kVelX:kVelZ+1]
    
    def get_quaternion(self,):
        """
        Retrieve Quaternion
        """
        quat = np.zeros(4)
        quat = self._state[kQuatW:kQuatZ+1]
        quat = quat / np.linalg.norm(quat)
        return quat

    def get_euler(self,):
        """
        Retrieve Euler Angles of the Vehicle
        """
        quat = self.get_quaternion()
        euler = self._quatToEuler(quat)
        return euler

    def get_axes(self):
        """
        Get the 3 axes (x, y, z) in world frame (for visualization only)
        """
        # axes in body frame
        b_x = np.array([self._arm_l, 0, 0])
        b_y = np.array([0, self._arm_l, 0])
        b_z = np.array([0, 0,  -self._arm_l])
        
        # rotation matrix
        rot_matrix = R.from_quat(self.get_quaternion()).as_matrix()
        quad_center = self.get_position()
        
        # axes in body frame
        w_x = rot_matrix@b_x + quad_center
        w_y = rot_matrix@b_y + quad_center
        w_z = rot_matrix@b_z + quad_center
        return [w_x, w_y, w_z]

    def get_motor_pos(self):
        """
        Get the 4 motor poses in world frame (for visualization only)
        """
        # motor position in body frame
        b_motor1 = np.array([np.sqrt(self._arm_l/2), np.sqrt(self._arm_l/2), 0])
        b_motor2 = np.array([-np.sqrt(self._arm_l/2), np.sqrt(self._arm_l/2), 0])
        b_motor3 = np.array([-np.sqrt(self._arm_l/2), -np.sqrt(self._arm_l/2), 0])
        b_motor4 = np.array([np.sqrt(self._arm_l/2), -np.sqrt(self._arm_l/2), 0])
        #
        rot_matrix = R.from_quat(self.get_quaternion()).as_matrix()
        quad_center = self.get_position()
        
        # motor position in world frame
        w_motor1 = rot_matrix@b_motor1 + quad_center
        w_motor2 = rot_matrix@b_motor2 + quad_center
        w_motor3 = rot_matrix@b_motor3 + quad_center
        w_motor4 = rot_matrix@b_motor4 + quad_center
        return [w_motor1, w_motor2, w_motor3, w_motor4]

    @staticmethod
    def _quatToEuler(quat):
        """
        Convert Quaternion to Euler Angles
        """
        quat_w, quat_x, quat_y, quat_z = quat[0], quat[1], quat[2], quat[3]
        euler_x = np.arctan2(2*quat_w*quat_x + 2*quat_y*quat_z, quat_w*quat_w - quat_x*quat_x - quat_y*quat_y + quat_z*quat_z)
        euler_y = -np.arcsin(2*quat_x*quat_z - 2*quat_w*quat_y)
        euler_z = np.arctan2(2*quat_w*quat_z+2*quat_x*quat_y, quat_w*quat_w + quat_x*quat_x - quat_y*quat_y - quat_z*quat_z)
        return [euler_x, euler_y, euler_z]

    def construct_states(num_data, episode_length=10, reset_strength=1, **kwargs):
        """
        Sample states for training the model
        Arguments:
            num_data: How much states to sample
            episode_length: Maximum number of states before resetting the env
            reset_strength (float between 0.5 - 1.5): How much randomization, i.e.
                    How far from target should the states be
        """
        # return data
        env = Quadrotor_v0()
        data = []
        # is_stable_list = list()
        while len(data) < num_data:
            env.reset()
            data.append(env.get_state())
        data = np.array(data)
        # np.save("data_backup/collected_data.npy", data)
        # print("saved first data", np.mean(is_stable_list))
        return data