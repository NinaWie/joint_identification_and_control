import torch
import torch.nn as nn
import torch.functional as F

from neural_control.environments.drone_dynamics import simulate_quadrotor


class RNN(nn.Module):

    def __init__(
        self, dataset, input_dim=4, output_dim=1, hidden_size=64, steps=5
    ):
        super(RNN, self).__init__()
        self.steps = steps
        self.dataset = dataset
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTMCell(input_dim, hidden_size=hidden_size, bias=True)
        self.in_ref = 3  # TODO
        # FOR RNN:
        # self.inp_weights = nn.Linear(input_dim, hidden_size)
        # self.inp_second = nn.Linear(hidden_size, hidden_size)
        # self.hidden_weights = nn.Linear(hidden_size, hidden_size)
        # self.hidden_second = nn.Linear(hidden_size, hidden_size)
        self.output_weights = nn.Linear(hidden_size, output_dim)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        # self.fc5 = nn.Linear(hidden_size, hidden_size)

    def update(self, x, h_t):
        # process input resenet like
        x = F.relu(self.inp_weights(x))
        # shortcut = x
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x)) + shortcut
        # # 2nd resnet block
        # shortcut = x
        # x = F.relu(self.fc4(x))
        inp_processed = F.relu(self.fc5(x))
        # inp_processed = self.inp_second(x)
        hidden_processed = self.hidden_second(F.relu(self.hidden_weights(h_t)))
        h_next = F.relu(inp_processed + hidden_processed)
        return h_next

    def forward(self, current_state, reference):
        # iterate over states
        hidden_state = torch.zeros(current_state.size()[0], self.hidden_size)
        cell_state = torch.zeros(current_state.size()[0], self.hidden_size)
        for k in range(self.steps):
            # TODO: concat with reference?
            in_state = self.dataset.normalize_states(current_state)
            in_ref = reference[:, k:k + self.in_ref]
            x = torch.hstack((in_state, in_ref))
            hidden_state, cell_state = self.LSTM(x, (hidden_state, cell_state))
            # print("hidden_state", hidden_state.size())
            # take hidden state to output the last action
            action = torch.sigmoid(self.output_weights(hidden_state))
            # apply drone dynamics
            current_state = simulate_quadrotor(current_state, action)
            # print("next state", next_state.size())
        return current_state

    def evaluate(self, x, hidden_state, cell_state):
        new_hidden_state, new_cell_state = self.LSTM(
            x, (hidden_state, cell_state)
        )
        action = torch.sigmoid(self.output_weights(new_hidden_state)) - .5
        return action, (new_hidden_state, new_cell_state)
