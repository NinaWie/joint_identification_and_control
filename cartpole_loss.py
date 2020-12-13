import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
torch.pi = torch.acos(torch.zeros(1)).item() * 2

# target state means that theta is zero --> only third position matters
target_state = 0  # torch.from_numpy(np.array([0, 0, 0, 0]))

# DEFINE VARIABLES
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 0.5  # actually half the pole's length
polemass_length = (masspole * length)
max_force_mag = 40.0
tau = 0.02  # seconds between state updates
muc = 0.0005
mup = 0.000002


def state_to_theta(state, action):
    """
    Compute new state from state and action
    """
    # get state
    x = state[:, 0]
    x_dot = state[:, 1]
    theta = state[:, 2]
    theta_dot = state[:, 3]
    # (x, x_dot, theta, theta_dot) = state

    # helper variables
    force = max_force_mag * action
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    sig = muc * torch.sign(x_dot)

    # add and multiply
    temp = torch.add(
        torch.squeeze(force),
        polemass_length * torch.mul(theta_dot**2, sintheta)
    )
    # divide
    thetaacc = (
        gravity * sintheta - (costheta * (temp - sig)) -
        (mup * theta_dot / polemass_length)
    ) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))

    # swapped these two lines
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc

    # add velocity of cart
    xacc = (temp - (polemass_length * thetaacc * costheta) - sig) / total_mass
    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc

    new_state = torch.stack((x, x_dot, theta, theta_dot), dim=1)
    return new_state


def loop_states(state, action_seq):
    nr_actions = action_seq.size()[1]
    # apply actions one after another
    for i in range(nr_actions):
        state = state_to_theta(state, action_seq[:, i])
    # return final state
    return state


def end_state_loss(state):
    # update state iteratively for each proposed action
    abs_state = torch.abs(state)

    pos_loss = abs_state[:, 0] / 1.2
    # velocity losss is low when x is high
    vel_loss = .2 * abs_state[:, 1] * (2.4 - abs_state[:, 0])**2
    angle_loss = 5 * abs_state[:, 2] / torch.pi
    # high angle velocity is fine if angle itself is high
    angle_vel_loss = .2 * abs_state[:, 3] * (torch.pi - abs_state[:, 2])
    loss = torch.vstack((pos_loss, vel_loss, angle_loss, angle_vel_loss))
    return loss


def control_loss_function(action, state, lambda_factor=.4, printout=0):
    in_state = state.clone()

    # bring action into -1 1 range
    action = torch.sigmoid(action) - .5

    # compute final state
    loss_of_model = end_state_loss(loop_states(state, action))

    # for regret, compare to pure right or pure left movement or no action
    for j, action_space in enumerate([-.5, 0, .5]):
        possibly_better_action = torch.zeros(action.size()) + action_space
        state_reached = loop_states(in_state, possibly_better_action)
        according_losses = end_state_loss(state_reached)
        if j == 0:
            min_loss_possible = according_losses
        else:
            # minimum of the possible losses
            min_loss_possible = torch.minimum(
                min_loss_possible, according_losses
            )

    # regret loss
    loss_diff = loss_of_model - min_loss_possible
    # maximum over kinds of losses
    loss = torch.max(loss_diff, dim=0).values
    # print(torch.max(loss_diff, dim=0).indices)

    if printout:
        print("action", action[0])
        print("loss_model", loss_model[0].item())
        print("min_loss_possible", min_loss_possible[0].item())
        print("loss_right", loss_right[0].item())
        print("loss_left", loss_left[0].item())
        print("loss_static", loss_static[0].item())
        print()

    return torch.sum(loss)  # + angle_acc)
