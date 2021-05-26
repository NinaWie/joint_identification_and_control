import torch
import torch.nn as nn


class LearntDynamics(torch.nn.Module):

    def __init__(self, state_size, action_size, transform_action=False):
        super(LearntDynamics, self).__init__()
        self.transform_action = transform_action
        if self.transform_action:
            self.linear_at = nn.Parameter(
                torch.diag(torch.ones(4)), requires_grad=True
            )
        # residual network
        self.linear_state_1 = nn.Linear(state_size + action_size, 64)
        std = 0.0001
        torch.nn.init.normal_(self.linear_state_1.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.linear_state_1.bias, mean=0.0, std=std)

        self.linear_state_2 = nn.Linear(64, state_size, bias=False)
        torch.nn.init.normal_(self.linear_state_2.weight, mean=0.0, std=std)

    def state_transformer(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        layer_1 = torch.tanh(self.linear_state_1(state_action))
        new_state = self.linear_state_2(layer_1)
        return new_state

    def forward(self, state, action, dt):
        if self.transform_action:
            action = torch.matmul(self.linear_at, torch.unsqueeze(action,
                                                                  2))[:, :, 0]
        # run through normal simulator f hat
        new_state = self.simulate(state, action, dt)
        # run through residual network delta
        added_new_state = self.state_transformer(state, action)
        return new_state + added_new_state
