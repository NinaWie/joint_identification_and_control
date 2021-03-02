import torch
import numpy as np
import os
from neural_control.environments.drone_env import (
    trajectory_training_data, full_state_training_data, load_training_data
)
from neural_control.environments.wing_env import sample_training_data
from neural_control.environments.cartpole_env import construct_states
from neural_control.environments.drone_dynamics import world_to_body_matrix

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

    def __init__(self, num_states, self_play, mean=None, std=None, **kwargs):
        # First constructor: New dataset for training
        self.self_play = 0
        self.mean = mean
        self.std = std
        self.num_sampled_states = num_states
        self.num_self_play = int(self_play * num_states)
        self.total_dataset_size = self.num_sampled_states + self.num_self_play

        states, ref_states = load_training_data(
            self.total_dataset_size, **kwargs
        )
        # np.save("states.npy", states)
        # np.save("ref_state.npy", ref_states)
        if mean is None:
            # sample states
            self.mean = np.mean(states, axis=0)
            self.std = np.std(states, axis=0)

        self.kwargs = kwargs
        (self.normed_states, self.states,
         self.ref_states) = self.prepare_data(states, ref_states)

        # count where to add new evaluation data
        self.eval_counter = 0

    def get_eval_index(self):
        """
        compute current index where to add new data
        """
        if self.num_self_play > 0:
            return (
                self.eval_counter % self.num_self_play
            ) + self.num_sampled_states

    def set_num_sampled_states(self, num_sampled_states):
        self.num_sampled_states = num_sampled_states

    def resample_data(self):
        """
        Sample new training data and replace dataset with it
        """
        states, ref_states = load_training_data(
            self.num_sampled_states, **self.kwargs
        )
        (prep_normed_states, prep_states,
         prep_ref_states) = self.prepare_data(states, ref_states)

        # add to first (the sampled) part of dataset
        num = self.num_sampled_states
        self.normed_states[:num] = prep_normed_states
        self.states[:num] = prep_states
        self.ref_states[:num] = prep_ref_states

    def get_and_add_eval_data(self, states, ref_states, add_to_dataset=False):
        """
        DEPRECATED - not used anymore
        While evaluating, add the data to the dataset with some probability
        to achieve self play
        """
        (normed_states, states,
         ref_states) = self.prepare_data(states, ref_states)
        if add_to_dataset and self.num_self_play > 0:
            # replace previous eval data with new eval data
            counter = self.get_eval_index()
            self.normed_states[counter] = normed_states[0]
            self.states[counter] = states[0]
            self.ref_states[counter] = ref_states[0]
            self.eval_counter += 1

        return normed_states, states, ref_states

    def to_torch(self, states):
        return torch.from_numpy(states).float().to(device)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # Select sample
        return (
            self.normed_states[index], self.states[index],
            self.ref_states[index]
        )

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

        # 1) just output the unnormalized states, but with the position at zero
        drone_states = self.to_torch(states)
        drone_pos = drone_states[:, :3].clone()
        drone_states[:, :3] = 0

        # 2) Normalized states
        normed_states = self.to_torch((states - self.mean) / self.std)[:, 3:]
        # normed_states[:, 6:10] = 0  # set rotor speeds to zero # TODO
        # first part: attitude and velocity
        # normed_states_first = normed_states[:, 3:9]
        # # second part: body rates
        # normed_states_second = normed_states[:, 13:]

        # Add rotation matrix to normed states
        drone_att = drone_states[:, 3:6]
        world_to_body = world_to_body_matrix(drone_att)
        drone_vel_body = torch.matmul(
            world_to_body, torch.unsqueeze(drone_states[:, 6:9], 2)
        )[:, :, 0]
        # reshape and concatenate
        rotation_matrix = torch.reshape(
            # first two columns
            world_to_body[:, :, :2],
            (-1, 6)
        )
        normed_drone_states = torch.hstack(
            (normed_states, rotation_matrix, drone_vel_body)
        )

        # 3) Reference trajectory to torch and relative to drone position
        torch_ref_states = self.to_torch(ref_states)
        for i in range(ref_states.shape[1]):
            torch_ref_states[:,
                             i, :3] = (torch_ref_states[:, i, :3] - drone_pos)
        # transform acceleration
        # torch_ref_states[:, :, 6:] *= self.kwargs["dt"]
        # TODO: normalize ref as well? rotation matrix instead of att?

        # ref_states_body = torch.unsqueeze(ref_states_body, 3)
        # for i in range(ref_states.shape[1]):
        #     # for each time step in the reference:
        #     # subtract position
        #     ref_states_body[:, i, :3] = torch.matmul(
        #         world_to_body, ref_states_body[:, i, :3]
        #     )
        #     # transform velocity
        #     ref_states_body[:, i, 3:6] = torch.matmul(
        #         world_to_body, ref_states_body[:, i, 3:6]
        #     )

        #     # transform acceleration
        #     ref_states_body[:, i, 6:9] = torch.matmul(
        #         world_to_body, ref_states_body[:, i, 6:9]
        #     )
        # ref_states_body = torch.squeeze(ref_states_body, dim=3)
        return normed_drone_states, drone_states, torch_ref_states


class CartpoleDataset(torch.utils.data.Dataset):
    """
    Dataset for training on cartpole task
    """

    def __init__(self, num_states=1000, **kwargs):
        # sample states
        state_arr_numpy = construct_states(num_states, **kwargs)
        # convert to normalized tensors
        self.labels = self.to_torch(state_arr_numpy)
        self.states = self.labels.clone()

    def to_torch(self, states):
        return torch.from_numpy(states).float().to(device)

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


class WingDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        num_states,
        mean=None,
        std=None,
        ref_mean=None,
        ref_std=None,
        **kwargs
    ):
        # First constructor: New dataset for training
        self.num_sampled_states = num_states
        states, ref_states = sample_training_data(
            self.num_sampled_states, **kwargs
        )
        if mean is None:
            # sample states
            mean = np.mean(states, axis=0)
            std = np.std(states, axis=0)
            pos_diff = ref_states[:, :2] - states[:, :2]
            ref_mean = np.mean(pos_diff, axis=0)
            ref_std = np.std(pos_diff, axis=0)

        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()
        self.ref_mean = torch.tensor(ref_mean).float()
        self.ref_std = torch.tensor(ref_std).float()

        self.kwargs = kwargs
        (
            self.normed_states, self.states, self.normed_ref_states,
            self.ref_states
        ) = self.prepare_data(states, ref_states)

        # count where to add new evaluation data
        self.eval_counter = 0

    def get_means_stds(self, param_dict):
        param_dict["mean"] = self.mean.tolist()
        param_dict["std"] = self.std.tolist()
        param_dict["ref_mean"] = self.ref_mean.tolist()
        param_dict["ref_std"] = self.ref_std.tolist()
        return param_dict

    def resample_data(self):
        """
        Sample new training data and replace dataset with it
        """
        states, ref_states = sample_training_data(
            self.num_sampled_states, **self.kwargs
        )
        (
            self.normed_states, self.states, self.normed_ref_states,
            self.ref_states
        ) = self.prepare_data(states, ref_states)

    def get_and_add_eval_data(self, states, ref_states, add_to_dataset=False):
        """
        While evaluating, add the data to the dataset with some probability
        to achieve self play
        """
        (normed_states, states, normed_ref_states,
         ref_states) = self.prepare_data(states, ref_states)
        if add_to_dataset and self.num_self_play > 0:
            # replace previous eval data with new eval data
            counter = self.get_eval_index()
            self.normed_states[counter] = normed_states[0]
            self.states[counter] = states[0]
            self.ref_states[counter] = ref_states[0]
            self.eval_counter += 1

        return normed_states, states, normed_ref_states, ref_states

    def to_torch(self, states):
        return torch.from_numpy(states).float().to(device)

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

        # 1) Normalized state and remove position
        states = self.to_torch(states)
        normed_states = ((states - self.mean) / self.std)[:, 2:]

        # TODO: transorm euler angle?

        # 3) Reference trajectory to torch and relative to drone position
        ref_states = self.to_torch(ref_states)
        # normalize
        relative_ref = ref_states - states[:, :2]
        ref_vec_norm = torch.sqrt(torch.sum(relative_ref**2, axis=1))
        normed_ref_states = (relative_ref.t() / ref_vec_norm).t()

        return normed_states, states, normed_ref_states, ref_states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # Select sample
        return (
            self.normed_states[index], self.states[index],
            self.normed_ref_states[index], self.ref_states[index]
        )
