import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from environments.mpc_drone_env import trajectory_training_data
from environments.cartpole_env import construct_states
from environments.drone_dynamics import world_to_body_matrix

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


def raw_states_to_torch(
    states, normalize=False, std=None, mean=None, return_std=False
):
    """
    Helper function to convert numpy state array to normalized tensors
    Argument states:
            One state (list of length 4) or array with x states (x times 4)
    """
    # either input one state at a time (evaluation) or an array
    if len(states.shape) == 1:
        states = np.expand_dims(states, 0)

    # save mean and std column wise
    if normalize:
        # can't use mean!
        if std is None:
            std = np.std(states, axis=0)
        if mean is None:
            mean = np.mean(states, axis=0)
        states = (states - mean) / std
        # assert np.all(np.isclose(np.std(states, axis=0), 1))
    else:
        std = 1

    # np.save("data_backup/quad_data.npy", states)

    states_to_torch = torch.from_numpy(states).float()

    # if we computed mean and std here, return it
    if return_std:
        return states_to_torch, mean, std
    return states_to_torch.to(device)


class DroneDataset(torch.utils.data.Dataset):

    def __init__(self, num_states=1000, mean=None, std=None, **kwargs):
        # First constructor: New dataset for training
        self.mean = mean
        self.std = std
        self.num_states = num_states
        states, ref_states = trajectory_training_data(num_states, **kwargs)
        if mean is None:
            # sample states
            self.mean = np.mean(states, axis=0)
            self.std = np.std(states, axis=0)

        self.kwargs = kwargs
        (self.states, self.ref_world,
         self.ref_body) = self.prepare_data(states, ref_states)

        # count how much of the data was replaced by self play
        self.eval_counter = 0
        self.self_play = 0

    def sample_data(self, self_play=0):
        """
        Sample new training data and replace dataset with it
        """
        self.self_play = self_play
        states, ref_states = trajectory_training_data(
            self.num_states, **self.kwargs
        )
        self.states, self.ref_world, self.ref_body = self.prepare_data(
            states, ref_states
        )
        self.eval_counter = 0

    def get_and_add_eval_data(self, states, ref_states):
        """
        While evaluating, add the data to the dataset with some probability
        to achieve self play
        """
        states, ref_world, ref_body = self.prepare_data(states, ref_states)
        if (np.random.rand() < self.self_play
            ) and (self.eval_counter < self.self_play * self.num_states):
            # self.self_play * s
            # replace data with eval data if below max eval data thresh
            self.states[self.eval_counter] = states[0]
            self.ref_world[self.eval_counter] = ref_world[0]
            self.ref_body[self.eval_counter] = ref_body[0]
            self.eval_counter += 1

        return states, ref_world, ref_body

    def to_torch(self, states):
        return torch.from_numpy(states).float().to(device)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # Select sample
        return self.states[index], self.ref_world[index], self.ref_body[index]

    def prepare_data(self, states, ref_states):
        """
        Prepare numpy data for input in ANN:
        - expand dims
        - normalize
        - world to body
        """
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
            ref_states = np.expand_dims(ref_states, 0)

        # Normalize
        unnormalized_state = self.to_torch(states)
        normed_states = (states - self.mean) / self.std

        # To torch tensors
        torch_states = self.to_torch(normed_states)
        torch_ref_states = self.to_torch(ref_states)
        for i in range(ref_states.shape[1]):
            torch_ref_states[:, i, :3] = (
                torch_ref_states[:, i, :3] - unnormalized_state[:, :3]
            )

        # transform acceleration
        torch_ref_states[:, :, 6:] *= self.kwargs["dt"]

        # # World to body frame - TODO: not working properly
        quaternions = states[:, 3:7]
        rot_matrices = self.to_torch(R.from_quat(quaternions).as_matrix())

        drone_vel_body = torch.matmul(
            rot_matrices, torch.unsqueeze(torch_states[:, 7:], 2)
        )[:, :, 0]
        # reshape and concatenate
        # TODO: only first two columns: [:, :, :2]
        rotation_matrix = torch.reshape(rot_matrices, (-1, 9))

        drone_states = torch.hstack(
            (torch_states, rotation_matrix, drone_vel_body)
        )
        ref_states_body = torch_ref_states.clone()
        return drone_states, torch_ref_states, ref_states_body


class CartpoleDataset(torch.utils.data.Dataset):
    """
    Dataset for training on cartpole task
    """

    def __init__(self, num_states=1000, **kwargs):
        # sample states
        state_arr_numpy = construct_states(num_states, **kwargs)
        # convert to normalized tensors
        self.labels = self.to_torch(state_arr_numpy)
        self.states = self.labels.copy()

    def to_torch(self, states):
        return torch.from_numpy(state_arr_numpy).float().to(device)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # Select sample
        return self.states[index], self.labels[index]

    def add_data(self, new_numpy_data):
        """
        Add numpy data that was generated in evaluation to the random data
        """
        self.labels = torch.vstack(
            (self.labels, self.to_torch(new_numpy_data))
        )
        self.states = torch.vstack(
            (self.states, self.to_torch(new_numpy_data))
        )

    @staticmethod
    def prepare_data(states):
        """
        Transform into input to NN
        """
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        return self.to_torch(states)
