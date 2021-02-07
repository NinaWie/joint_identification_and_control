import torch
import numpy as np
from neural_control.environments.copter import copter_params
from types import SimpleNamespace

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
copter_params = SimpleNamespace(**copter_params)
copter_params.translational_drag = torch.from_numpy(
    copter_params.translational_drag
).to(device)
copter_params.gravity = torch.from_numpy(copter_params.gravity).to(device)
copter_params.rotational_drag = torch.from_numpy(
    copter_params.rotational_drag
).to(device)
copter_params.frame_inertia = torch.from_numpy(copter_params.frame_inertia
                                               ).float().to(device)


def world_to_body_matrix(attitude):
    """
    Creates a transformation matrix for directions from world frame
    to body frame for a body with attitude given by `euler` Euler angles.
    :param euler: The Euler angles of the body frame.
    :return: The transformation matrix.
    """

    # check if we have a cached result already available
    roll = attitude[:, 0]
    pitch = attitude[:, 1]
    yaw = attitude[:, 2]

    Cy = torch.cos(yaw)
    Sy = torch.sin(yaw)
    Cp = torch.cos(pitch)
    Sp = torch.sin(pitch)
    Cr = torch.cos(roll)
    Sr = torch.sin(roll)

    # create matrix
    m1 = torch.transpose(torch.vstack([Cy * Cp, Sy * Cp, -Sp]), 0, 1)
    m2 = torch.transpose(
        torch.vstack(
            [Cy * Sp * Sr - Cr * Sy, Cr * Cy + Sr * Sy * Sp, Cp * Sr]
        ), 0, 1
    )
    m3 = torch.transpose(
        torch.vstack(
            [Cy * Sp * Cr + Sr * Sy, Cr * Sy * Sp - Cy * Sr, Cr * Cp]
        ), 0, 1
    )
    matrix = torch.stack((m1, m2, m3), dim=1)

    return matrix


def linear_dynamics(rotor_speed, attitude, velocity):
    """
    Calculates the linear acceleration of a quadcopter with parameters
    `copter_params` that is currently in the dynamics state composed of:
    :param rotor_speed: current rotor speeds
    :param attitude: current attitude
    :param velocity: current velocity
    :return: Linear acceleration in world frame.
    """
    m = copter_params.mass
    b = copter_params.thrust_factor
    Kt = copter_params.translational_drag

    world_to_body = world_to_body_matrix(attitude)
    body_to_world = torch.transpose(world_to_body, 1, 2)

    squared_speed = torch.sum(rotor_speed**2, axis=1)
    constant_vec = torch.zeros(3).to(device)
    constant_vec[2] = 1

    thrust = b / m * torch.mul(
        torch.matmul(body_to_world, constant_vec).t(), squared_speed
    ).t()
    Ktw = torch.matmul(
        body_to_world, torch.matmul(torch.diag(Kt).float(), world_to_body)
    )
    drag = torch.squeeze(torch.matmul(Ktw, torch.unsqueeze(velocity, 2)) / m)
    thrust_minus_drag = thrust - drag + copter_params.gravity
    # version for batch size 1 (working version)
    # summed = torch.add(
    #     torch.transpose(drag * (-1), 0, 1), thrust
    # ) + copter_params.gravity
    # print("output linear", thrust_minus_drag.size())
    return thrust_minus_drag


def to_euler_matrix(attitude):
    # attitude is [roll, pitch, yaw]
    pitch = attitude[:, 1]
    roll = attitude[:, 0]
    Cp = torch.cos(pitch)
    Sp = torch.sin(pitch)
    Cr = torch.cos(roll)
    Sr = torch.sin(roll)

    zero_vec_bs = torch.zeros(Sp.size()).to(device)
    ones_vec_bs = torch.ones(Sp.size()).to(device)

    # create matrix
    m1 = torch.transpose(torch.vstack([ones_vec_bs, zero_vec_bs, -Sp]), 0, 1)
    m2 = torch.transpose(torch.vstack([zero_vec_bs, Cr, Cp * Sr]), 0, 1)
    m3 = torch.transpose(torch.vstack([zero_vec_bs, -Sr, Cp * Cr]), 0, 1)
    matrix = torch.stack((m1, m2, m3), dim=1)

    # matrix = torch.tensor([[1, 0, -Sp], [0, Cr, Cp * Sr], [0, -Sr, Cp * Cr]])
    return matrix


def euler_rate(attitude, angular_velocity):
    euler_matrix = to_euler_matrix(attitude)
    together = torch.matmul(
        euler_matrix, torch.unsqueeze(angular_velocity.float(), 2)
    )
    # print("output euler rate", together.size())
    return torch.squeeze(together)


def propeller_torques(rotor_speeds):
    """
    Calculates the torques that are directly generated by the propellers.
    :return:
    """
    # squared
    squared_speeds = rotor_speeds**2
    r0 = squared_speeds[:, 0]
    r1 = squared_speeds[:, 1]
    r2 = squared_speeds[:, 2]
    r3 = squared_speeds[:, 3]

    Lb = copter_params.arm_length * copter_params.thrust_factor
    d = copter_params.drag_factor
    motor_torque = r3 + r1 - r2 - r0
    # print(motor_torque.size())
    B = torch.stack([Lb * (r3 - r1), Lb * (r0 - r2), d * motor_torque]).t()
    # print("propeller torque outputs:", B.size())
    return B


def net_rotor_speed(rotorspeeds):
    """
    Calculate net rotor speeds (subtract 2 from other 2)
    """
    return (
        rotorspeeds[:, 0] - rotorspeeds[:, 1] + rotorspeeds[:, 2] -
        rotorspeeds[:, 3]
    )


def angular_momentum_body_frame(rotor_speeds, angular_velocity):
    """
    Calculates the angular momentum of a quadcopter with parameters
    `copter_params` that is currently in the dynamics state `state`.
    :param av: Current angular velocity.
    :return: angular acceleration in body frame.
    """
    av = angular_velocity
    J = copter_params.rotor_inertia
    Kr = copter_params.rotational_drag
    inertia = copter_params.frame_inertia

    zeros_av = torch.zeros(av.size()[0]).to(device)

    # this is the wrong shape, should be transposed, but for multipluing later
    # in gyro we would have to transpose again - so don't do it here
    transformed_av = torch.stack((av[:, 2], -av[:, 1], zeros_av))
    # J is scalar, net rotor speed outputs vector of len batch size
    gyro = torch.transpose(
        net_rotor_speed(rotor_speeds) * J * transformed_av, 0, 1
    )
    drag = Kr * av
    Mp = propeller_torques(rotor_speeds)

    B = Mp - drag + gyro - torch.cross(av, inertia * av, dim=1)
    return B


def action_to_rotor(action, rotor_speed):
    """
    Compute new rotor speeds from previous rotor speed and action (control
    signals)
    Arguments:
        action: torch tensor of size (batchsize, 4)
        rotor_speed: torch tensor of size (batchsize, 3)
    Returns:
        rotor_speed: torch tensor of size (batchsize, 3)
    """
    # # set desired rotor speeds based on action # TODO: was sqrt action
    desired_rotor_speeds = action * copter_params.max_rotor_speed

    zero_for_rotor = torch.zeros(rotor_speed.size()).to(device)

    # let rotor speed approach desired rotor speed and avoid negative rotation
    # gamma = 1.0 - 0.5**(dt / copter_params.rotor_speed_half_time)
    # dw = gamma * (desired_rotor_speeds - rotor_speed)
    rotor_speed = rotor_speed + .3 * (desired_rotor_speeds - rotor_speed)
    rotor_speed = torch.maximum(rotor_speed, zero_for_rotor)
    return rotor_speed


def simulate_quadrotor(action, state, dt=0.02):
    """
    Simulate the dynamics of the quadrotor for the timestep given
    in `dt`. First the rotor speeds are updated according to the desired
    rotor speed, and then linear and angular accelerations are calculated
    and integrated.
    Arguments:
        action: float tensor of size (BATCH_SIZE, 4) - rotor thrust
        state: float tensor of size (BATCH_SIZE, 16) - drone state (see below)
    Returns:
        Next drone state (same size as state)
    """
    # extract state
    position = state[:, :3]
    attitude = state[:, 3:6]
    velocity = state[:, 6:9]
    rotor_speed = state[:, 9:13]
    angular_velocity = state[:, 13:16]

    rotor_speed = action_to_rotor(action, rotor_speed)

    acceleration = linear_dynamics(rotor_speed, attitude, velocity)

    ang_momentum = angular_momentum_body_frame(rotor_speed, angular_velocity)
    angular_acc = ang_momentum / copter_params.frame_inertia
    # update state variables
    position = position + 0.5 * dt * dt * acceleration + 0.5 * dt * velocity
    velocity = velocity + dt * acceleration
    angular_velocity = angular_velocity + dt * angular_acc
    attitude = attitude + dt * euler_rate(attitude, angular_velocity)
    # set final state
    state = torch.hstack(
        (position, attitude, velocity, rotor_speed, angular_velocity)
    )
    return state.float()