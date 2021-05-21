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
        super(SequenceQuadDynamics,
              self).__init__((12 + 4) * buffer_length, 4, out_state_size=12)

    def simulate(self, state, action, dt):
        return self.simulate_quadrotor(state, action, dt)

    def forward(self, state, state_action_buffer, action, dt):
        # run through normal simulator f hat
        new_state = self.simulate(state, action, dt)
        # run through residual network delta
        added_new_state = self.state_transformer(state_action_buffer, action)
        return new_state + added_new_state
