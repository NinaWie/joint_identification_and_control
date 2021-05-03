import torch
import numpy as np
import os
from neural_control.environments.drone_env import full_state_training_data
from neural_control.environments.wing_env import sample_training_data
from neural_control.environments.cartpole_env import construct_states
from neural_control.dynamics.quad_dynamics_base import Dynamics
from neural_control.dynamics.fixed_wing_dynamics import FixedWingDynamics

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DroneDataset(torch.utils.data.Dataset):

    def __init__(self, num_states, self_play, mean=None, std=None, **kwargs):
        # First constructor: New dataset for training
        self.mean = mean
        self.std = std
        self.num_sampled_states = num_states
        self.num_self_play = int(self_play * num_states)
        self.total_dataset_size = self.num_sampled_states + self.num_self_play

        self.kwargs = kwargs

        # count where to add new evaluation data
        self.eval_counter = 0

    def get_means_stds(self, param_dict):
        param_dict["mean"] = self.mean.tolist()
        param_dict["std"] = self.std.tolist()
        return param_dict

    def get_eval_index(self):
        """
        compute current index where to add new data
        """
        if self.num_self_play > 0:
            return (
                self.eval_counter % self.num_self_play
            ) + self.num_sampled_states

    def resample_data(self):
        """
        Sample new training data and replace dataset with it
        """
        states, ref_states = self.sample_data(self.num_sampled_states)
        (prep_normed_states, prep_states, prep_in_ref_states,
         prep_ref_states) = self.prepare_data(states, ref_states)

        # add to first (the sampled) part of dataset
        num = self.num_sampled_states
        self.normed_states[:num] = prep_normed_states
        self.states[:num] = prep_states
        self.in_ref_states[:num] = prep_in_ref_states
        self.ref_states[:num] = prep_ref_states

    def get_and_add_eval_data(self, states, ref_states, add_to_dataset=False):
        """
        While evaluating, add the data to the dataset with some probability
        to achieve self play
        """
        (normed_states, states, in_ref_states,
         ref_states) = self.prepare_data(states, ref_states)
        if add_to_dataset and self.num_self_play > 0:
            # replace previous eval data with new eval data
            counter = self.get_eval_index()
            self.normed_states[counter] = normed_states[0]
            self.states[counter] = states[0]
            self.in_ref_states[counter] = in_ref_states[0]
            self.ref_states[counter] = ref_states[0]
            self.eval_counter += 1

        return normed_states, states, in_ref_states, ref_states

    def to_torch(self, states):
        return torch.from_numpy(states).float().to(device)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # Select sample
        return (
            self.normed_states[index], self.states[index],
            self.in_ref_states[index], self.ref_states[index]
        )


class QuadDataset(DroneDataset):

    def __init__(self, num_states, self_play, mean=None, std=None, **kwargs):
        super().__init__(num_states, self_play, mean=mean, std=std, **kwargs)

        states, ref_states = self.sample_data(self.total_dataset_size)
        (self.normed_states, self.states, self.in_ref_states,
         self.ref_states) = self.prepare_data(states, ref_states)

    def sample_data(self, num_states):
        states, ref_states = full_state_training_data(
            num_states, **self.kwargs
        )
        return states, ref_states

    def rot_world_to_body(self, state_vector, world_to_body):
        """
        world_to_body is rotation matrix
        vector is array of size (?, 3)
        """
        return torch.matmul(world_to_body, torch.unsqueeze(state_vector,
                                                           2))[:, :, 0]

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
        # 1) make torch arrays
        drone_states = self.to_torch(states)
        torch_ref_states = self.to_torch(ref_states)

        # 2) compute relative position and reset drone position to zero
        subtract_drone_pos = torch.unsqueeze(drone_states[:, :3], 1)
        subtract_drone_vel = torch.unsqueeze(drone_states[:, 6:9], 1)
        torch_ref_states[:, :, :3] = (
            torch_ref_states[:, :, :3] - subtract_drone_pos
        )
        drone_states[:, :3] = 0

        # get rotation matrix
        drone_vel = drone_states[:, 6:9]
        world_to_body = Dynamics.world_to_body_matrix(drone_states[:, 3:6])
        drone_vel_body = self.rot_world_to_body(drone_vel, world_to_body)
        # first two columns of rotation matrix
        drone_rotation_matrix = torch.reshape(world_to_body[:, :, :2], (-1, 6))

        # for the drone, input is: vel (body and world), av, rotation matrix
        inp_drone_states = torch.hstack(
            (
                drone_vel, drone_rotation_matrix, drone_vel_body,
                drone_states[:, 9:12]
            )
        )

        # for the reference, input is: relative pos, vel, vel-drone vel
        vel_minus_veldrone = torch_ref_states[:, :, 6:9] - subtract_drone_vel
        inp_ref_states = torch.cat(
            (
                torch_ref_states[:, :, :3], torch_ref_states[:, :, 6:9],
                vel_minus_veldrone
            ),
            dim=2
        )

        # transform acceleration
        return inp_drone_states, drone_states, inp_ref_states, torch_ref_states


class CartpoleDataset(torch.utils.data.Dataset):
    """
    Dataset for training on cartpole task
    """

    def __init__(self, num_states=1000, thresh_div=.21, **kwargs):
        self.resample_data(num_states, thresh_div)

    def resample_data(self, num_states, thresh_div):
        # sample states
        state_arr_numpy = construct_states(num_states, thresh_div=thresh_div)
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
        cutoff = min([len(new_numpy_data), self.__len__()])
        print("adding self play states", cutoff)
        self.labels[:cutoff] = self.to_torch(new_numpy_data)[:cutoff]
        self.states[:cutoff] = self.to_torch(new_numpy_data)[:cutoff]


class WingDataset(DroneDataset):

    def __init__(
        self,
        num_states,
        self_play=0,
        mean=None,
        std=None,
        ref_mean=None,
        ref_std=None,
        delta_t=0.05,
        nr_actions=10,
        **kwargs
    ):
        self.dt = delta_t
        self.nr_actions = nr_actions
        super().__init__(num_states, self_play, mean=mean, std=std, **kwargs)
        if self.mean is None:
            self.mean = torch.tensor(
                [
                    26.69862174987793, 0.3841058909893036, 0.32479792833328247,
                    11.525899887084961, -0.00016766408225521445,
                    0.16617104411125183, 0.007394296582788229, 0.018172707409,
                    0.020353179425001144, -0.0005361468647606671,
                    0.01662314310669899, 0.004487641621381044
                ]
            ).float()
            self.std = torch.tensor(
                [
                    16.626325607299805, 0.8449159860610962, 0.8879243731498718,
                    0.6243225932121277, 0.28072822093963623, 0.29176747798,
                    0.04499124363064766, 0.10370047390460968, 0.049977313727,
                    0.06449887901544571, 0.27508440613746643, 0.05634994804859
                ]
            ).float()
        else:
            self.mean = torch.tensor(self.mean)
            self.std = torch.tensor(self.std)
        states, ref_states = self.sample_data(self.total_dataset_size)
        (self.normed_states, self.states, self.in_ref_states,
         self.ref_states) = self.prepare_data(states, ref_states)

    def sample_data(self, num_samples):
        """
        Interface to training data function of fixed wing drone
        """
        if self.num_sampled_states < 5:
            # actually no sampling, only self play
            states = np.random.rand(num_samples, 12)
            ref_states = np.random.rand(num_samples, 3)
        else:
            states, ref_states = sample_training_data(
                num_samples, **self.kwargs
            )
        return states, ref_states

    def _compute_target_pos(self, current_state, ref_vector):
        # # GIVE LINEAR TRAJECTORY FOR LOSS
        # speed = torch.sqrt(torch.sum(current_state[:, 3:6]**2, dim=1))
        vec_len_per_step = 12 * self.dt
        # form auxiliary array with linear reference for loss computation
        target_pos = torch.zeros((current_state.size()[0], self.nr_actions, 3))
        for i in range(self.nr_actions):
            for j in range(3):
                target_pos[:, i, j] = current_state[:, j] + (
                    ref_vector[:, j] * vec_len_per_step * (i + 1)
                )
        return target_pos

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
        if not isinstance(states, torch.Tensor):
            states = self.to_torch(states)
            ref_states = self.to_torch(ref_states)

        # 1) Normalized state and remove position
        normed_states = ((states - self.mean) / self.std)[:, 3:]

        # rot_matrix = FixedWingDynamics.inertial_body_function(
        #     states[:, 6], states[:, 7], states[:, 8]
        # )
        # drone_rot_matrix = torch.reshape(rot_matrix, (-1, 9))

        # # stack vel, rot matrix, angular vel
        # inp_drone_states = torch.hstack(
        #     (normed_states[:, 3:6], drone_rot_matrix, normed_states[:, 9:12])
        # )

        # 3) Reference trajectory to torch and relative to drone position
        # normalize
        relative_ref = ref_states - states[:, :3]
        ref_vec_norm = torch.sqrt(torch.sum(relative_ref**2, axis=1))

        # get vector in direction of target
        normed_ref_vector = (relative_ref.t() / ref_vec_norm).t()
        # transform to the input states
        ref_states = self._compute_target_pos(states, normed_ref_vector)
        relative_ref = ref_states[:, -1] - states[:, :3]

        return normed_states, states, relative_ref, ref_states
