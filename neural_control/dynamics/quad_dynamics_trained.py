import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from neural_control.dynamics.quad_dynamics_flightmare import (
    FlightmareDynamics
)
from neural_control.dynamics.learnt_dynamics import LearntDynamics


class LearntQuadDynamics(LearntDynamics, FlightmareDynamics):

    def __init__(self, initial_params={}, trainable_params=True):
        FlightmareDynamics.__init__(self, initial_params)
        super(LearntQuadDynamics, self).__init__(12, 4)

        # VARIABLES - dynamics parameters
        if trainable_params:
            self.torch_translational_drag = nn.Parameter(
                self.torch_translational_drag
            )
            self.torch_rotational_drag = nn.Parameter(
                self.torch_rotational_drag
            )
            self.mass = nn.Parameter(
                torch.tensor([self.mass]), requires_grad=True
            )
            self.torch_inertia_vector = nn.Parameter(
                torch.from_numpy(self.inertia_vector).float(),
                requires_grad=True,
            )
            self.torch_kinv_vector = nn.Parameter(
                torch.tensor(self.kinv_ang_vel_tau).float(),
                requires_grad=True,
            )
            # derivations from params
            self.torch_inertia_J = torch.diag(self.torch_inertia_vector)
            self.torch_kinv_ang_vel_tau = torch.diag(self.torch_kinv_vector)

    def simulate(self, state, action, dt):
        return self.simulate_quadrotor(action, state, dt)


class SequenceQuadDynamics(LearntDynamics, FlightmareDynamics):

    def __init__(self, buffer_length=3):
        FlightmareDynamics.__init__(self)
        # input state action history has 22 channels
        super(SequenceQuadDynamics,
              self).__init__(22 * buffer_length, 4, out_state_size=12)
        self.conv_history_1 = nn.Conv1d(22, 20, kernel_size=3)
        self.conv_history_2 = nn.Conv1d(20, 20, kernel_size=3)

    def simulate(self, state, action, dt):
        return self.simulate_quadrotor(action, state, dt)

    def forward(self, state, state_action_buffer, action, dt):
        # run through normal simulator f hat
        new_state = self.simulate(state, action, dt)
        # select only three
        state_action_buffer = state_action_buffer[:, [0, 2, 4]]
        # FOR CONVOLUTION
        # history = torch.transpose(state_action_buffer, 1, 2)
        # history = torch.tanh(self.conv_history_1(history))
        # # history = torch.relu(self.conv_history_2(history))
        # history = torch.reshape(history, (-1, 20))
        # state_action_buffer[:, :, 3:6] = state_action_buffer[:, :, 3:6] * .1
        history = torch.reshape(
            state_action_buffer, (
                -1,
                state_action_buffer.size()[1] * state_action_buffer.size()[2]
            )
        )
        # run through residual network delta
        added_new_state = self.state_transformer(history, action)
        return new_state + added_new_state
