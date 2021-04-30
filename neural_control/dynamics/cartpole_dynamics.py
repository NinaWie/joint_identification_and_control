import torch
import numpy as np
import casadi as ca

from neural_control.dynamics.learnt_dynamics import LearntDynamics

# target state means that theta is zero --> only third position matters
target_state = 0  # torch.from_numpy(np.array([0, 0, 0, 0]))

# DEFINE VARIABLES
gravity = 9.81
cfg = {
    "masscart": 1.0,
    "masspole": 0.1,
    "length": 0.5,  # actually half the pole's length
    "max_force_mag": 30.0,
    "muc": 0.0005,
    "mup": 0.000002,
    "wind": 0
}


class CartpoleDynamics:

    def __init__(self, modified_params={}, test_time=0):
        self.cfg = cfg
        self.test_time = test_time
        self.cfg.update(modified_params)
        self.cfg["total_mass"] = self.cfg["masspole"] + self.cfg["masscart"]
        self.cfg["polemass_length"] = self.cfg["masspole"] * self.cfg["length"]

    def __call__(self, state, action, dt=0.02):
        return self.simulate_cartpole(state, action, dt=dt)

    def simulate_cartpole(self, state, action, dt=0.02):
        """
        Compute new state from state and action
        """
        # # get action to range [-1, 1]
        # action = torch.sigmoid(action)
        # action = action * 2 - 1
        # # Attempt to use discrete actions
        # if self.test_time:
        #     action = torch.tensor(
        #         [-1]
        #     ) if action[0, 0] > action[0, 1] else torch.tensor([-1])
        # else:
        #     action = torch.softmax(action, dim=1)
        #     action = action[:, 0] * -1 + action[:, 1]
        # get state
        x = state[:, 0]
        x_dot = state[:, 1]
        theta = state[:, 2]
        theta_dot = state[:, 3]
        # (x, x_dot, theta, theta_dot) = state

        # helper variables
        force = self.cfg["max_force_mag"] * action
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        sig = self.cfg["muc"] * torch.sign(x_dot)

        # add and multiply
        temp = torch.add(
            torch.squeeze(force),
            self.cfg["polemass_length"] * torch.mul(theta_dot**2, sintheta)
        )
        # divide
        thetaacc = (
            gravity * sintheta - (costheta * (temp - sig)) -
            (self.cfg["mup"] * theta_dot / self.cfg["polemass_length"])
        ) / (
            self.cfg["length"] * (
                4.0 / 3.0 - self.cfg["masspole"] * costheta * costheta /
                self.cfg["total_mass"]
            )
        )
        wind_drag = self.cfg["wind"] * costheta

        # swapped these two lines
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * (thetaacc + wind_drag)

        # add velocity of cart
        xacc = (
            temp - (self.cfg['polemass_length'] * thetaacc * costheta) - sig
        ) / self.cfg["total_mass"]
        x = x + dt * x_dot
        x_dot = x_dot + dt * xacc

        new_state = torch.stack((x, x_dot, theta, theta_dot), dim=1)
        return new_state


class LearntCartpoleDynamics(LearntDynamics, CartpoleDynamics):

    def __init__(self, modified_params={}, not_trainable=[]):
        CartpoleDynamics.__init__(self, modified_params=modified_params)
        super(LearntCartpoleDynamics, self).__init__(4, 1)

        dict_pytorch = {}
        for key, val in self.cfg.items():
            requires_grad = True
            # # code to avoid training the parameters
            if not_trainable == "all" or key in not_trainable:
                requires_grad = False
            dict_pytorch[key] = torch.nn.Parameter(
                torch.tensor([val]), requires_grad=requires_grad
            )
        self.cfg = torch.nn.ParameterDict(dict_pytorch)

    def simulate(self, state, action, dt):
        return self.simulate_cartpole(state, action, dt)


class CartpoleDynamicsMPC(CartpoleDynamics):

    def __init__(self, modified_params={}):
        CartpoleDynamics.__init__(self, modified_params=modified_params)

    def simulate_cartpole(self, dt):
        (x, x_dot, theta, theta_dot) = (
            ca.SX.sym("x"), ca.SX.sym("x_dot"), ca.SX.sym("theta"),
            ca.SX.sym("theta_dot")
        )
        action = ca.SX.sym("action")
        x_state = ca.vertcat(x, x_dot, theta, theta_dot)

        # helper variables
        force = self.cfg["max_force_mag"] * action
        costheta = ca.cos(theta)
        sintheta = ca.sin(theta)
        sig = self.cfg["muc"] * ca.sign(x_dot)

        # add and multiply
        temp = force + self.cfg["polemass_length"] * theta_dot**2 * sintheta

        # divide
        thetaacc = (
            gravity * sintheta - (costheta * (temp - sig)) -
            (self.cfg["mup"] * theta_dot / self.cfg["polemass_length"])
        ) / (
            self.cfg["length"] * (
                4.0 / 3.0 - self.cfg["masspole"] * costheta * costheta /
                self.cfg["total_mass"]
            )
        )
        wind_drag = self.cfg["wind"] * costheta

        # add velocity of cart
        x_acc = (
            temp - (self.cfg['polemass_length'] * thetaacc * costheta) - sig
        ) / self.cfg["total_mass"]

        x_state_dot = ca.vertcat(x_dot, x_acc, theta_dot, thetaacc + wind_drag)
        X = x_state + dt * x_state_dot

        F = ca.Function('F', [x_state, action], [X], ['x', 'u'], ['ode'])
        return F


if __name__ == "__main__":
    state_test_np = np.array([0.5, 1.3, 0.1, 0.4])
    state_test = torch.unsqueeze(torch.from_numpy(state_test_np), 0).float()
    action_test_np = np.array([0.4])
    action_test = torch.unsqueeze(torch.from_numpy(action_test_np), 0).float()

    normal_dyn = CartpoleDynamics()
    next_state = normal_dyn(state_test, action_test, 0.02)
    print("------------")
    print(next_state[0])

    # test: compare to mpc
    # if test doesnt work, remove clamp!!
    mpc_dyn = CartpoleDynamicsMPC()
    F = mpc_dyn.simulate_cartpole(0.02)
    mpc_state = F(state_test_np, action_test_np)
    print("--------------------")
    print(mpc_state)
