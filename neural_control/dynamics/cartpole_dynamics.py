import torch
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
        # get action to range [-1, 1]
        action = torch.sigmoid(action)
        action = action * 2 - 1
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
        super(LearntCartpoleDynamics, self).__init__()

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
