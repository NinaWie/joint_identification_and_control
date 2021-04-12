import numpy as np
import torch
import contextlib
import time
import torch.optim as optim

from neural_control.drone_loss import reference_loss
from neural_control.dataset import QuadDataset


@contextlib.contextmanager
def dummy_context():
    yield None


class NetworkWrapper:

    def __init__(
        self,
        model,
        dataset,
        optimizer=None,
        horizon=5,
        max_drone_dist=0.1,
        render=0,
        dt=0.02,
        take_every_x=1000,
        **kwargs
    ):
        self.dataset = dataset
        self.net = model
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist
        self.training_means = None
        self.render = render
        self.dt = dt
        self.optimizer = optimizer
        self.take_every_x = take_every_x
        self.action_counter = 0

        # four control signals
        self.action_dim = 4

    def predict_actions(self, current_np_state, ref_states):
        """
        Predict an action for the current state. This function is used by all
        evaluation functions
        """
        # preprocess state
        current_torch_state = torch.tensor([current_np_state]).float()
        ref_torch_state = torch.tensor([ref_states]).float()
        in_state, in_ref = QuadDataset.preprocess_data(
            current_torch_state, ref_torch_state
        )

        suggested_action = self.net(in_state, in_ref)

        suggested_action = torch.sigmoid(suggested_action)

        suggested_action = torch.reshape(
            suggested_action, (1, -1, self.action_dim)
        )
        numpy_action_seq = suggested_action[0].detach().numpy()
        # print([round(a, 2) for a in numpy_action_seq[0]])
        # keep track of actions
        self.action_counter += 1
        return numpy_action_seq

    def check_ood(self, drone_state, ref_states):
        if self.training_means is None:
            _, reference_training_data = trajectory_training_data(
                500, max_drone_dist=self.max_drone_dist, dt=self.dt
            )
            self.training_means = np.mean(reference_training_data, axis=0)
            self.training_std = np.std(reference_training_data, axis=0)
        drone_state_names = np.array(
            [
                "pos_x", "pos_y", "pos_z", "att_1", "att_2", "att_3", "vel_x",
                "vel_y", "vel_z", "rot_1", "rot_2", "rot_3", "rot_4",
                "att_vel_1", "att_vel_2", "att_vel_3"
            ]
        )
        ref_state_names = np.array(
            [
                "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "acc_1",
                "acc_2", "acc_3"
            ]
        )
        normed_drone_state = np.absolute(
            (drone_state - self.dataset.mean) / self.dataset.std
        )
        normed_ref_state = np.absolute(
            (ref_states - self.training_means) / self.training_std
        )
        if np.any(normed_drone_state > 3):
            print("state outlier:", drone_state_names[normed_drone_state > 3])
        if np.any(normed_ref_state > 3):
            for i in range(ref_states.shape[0]):
                print(
                    f"ref outlier (t={i}):",
                    ref_state_names[normed_ref_state[i] > 3]
                )


class FixedWingNetWrapper:

    def __init__(self, model, dataset, horizon=1, take_every_x=1000, **kwargs):
        self.net = model
        self.dataset = dataset
        self.horizon = horizon
        self.action_dim = 4
        self.action_counter = 0
        self.take_every_x = take_every_x

    def predict_actions(self, state, ref_state):
        # determine whether we also add the sample to our train data
        add_to_dataset = (self.action_counter + 1) % self.take_every_x == 0

        normed_state, _, normed_ref, _ = self.dataset.get_and_add_eval_data(
            state, ref_state, add_to_dataset=add_to_dataset
        )
        with torch.no_grad():
            suggested_action = self.net(normed_state, normed_ref)
            suggested_action = torch.sigmoid(suggested_action)[0]

            suggested_action = torch.reshape(
                suggested_action, (self.horizon, self.action_dim)
            )

        self.action_counter += 1
        return suggested_action.detach().numpy()
