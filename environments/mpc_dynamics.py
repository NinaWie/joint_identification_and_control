import numpy as np
import torch

# state index
kPosX = 0
kPosY = 1
kPosZ = 2
kQuatW = 3
kQuatX = 4
kQuatY = 5
kQuatZ = 6
kVelX = 7
kVelY = 8
kVelZ = 9

s_dim = 10
a_dim = 4
gz = 9.81

# action index
kThrust = 0
kWx = 1
kWy = 2
kWz = 3


def get_quaternion(state):
    """
    Retrieve Quaternion
    """
    quat = state[:, kQuatW:kQuatZ+1]
    norm_quat = torch.sqrt(torch.sum(quat**2, dim=1))
    quat = quat / norm_quat
    return quat

def drone_model(state, action):
    """
    System dynamics: ds = f(x, u)
    """
    thrust = action[:,0]
    wx = action[:,1]
    wy = action[:,2]
    wz = action[:,3]
    #
    dstate = torch.zeros(s_dim)

    dstate[kPosX:kPosZ+1] = state[kVelX:kVelZ+1]

    qw, qx, qy, qz = get_quaternion(state)

    dstate[kQuatW] = 0.5 * ( -wx*qx - wy*qy - wz*qz )
    dstate[kQuatX] = 0.5 * (  wx*qw + wz*qy - wy*qz )
    dstate[kQuatY] = 0.5 * (  wy*qw - wz*qx + wx*qz )
    dstate[kQuatZ] = 0.5 * (  wz*qw + wy*qx - wx*qy )

    dstate[kVelX] = 2 * ( qw*qy + qx*qz ) * thrust
    dstate[kVelY] = 2 * ( qy*qz - qw*qx ) * thrust
    # dstate[kVelZ] = (1 - 2*qx*qx - 2*qy*qy) * thrust - gz
    dstate[kVelZ] = (qw*qw - qx*qx -qy*qy + qz*qz) * thrust - gz

    return dstate


def dynamics(X, action, dt):
    """
    Apply the control command on the quadrotor and transits the system to the next state
    X is the current state
    """
    # rk4 int
    M = 4
    DT = dt / M
    #
    for i in range(M):
        k1 = DT*drone_model(X, action)
        k2 = DT*drone_model(X + 0.5*k1, action)
        k3 = DT*drone_model(X + 0.5*k2, action)
        k4 = DT*drone_model(X + k3, action)
        #
        X = X + (k1 + 2.0*(k2 + k3) + k4)/6.0
    # X is the new state
    return X
