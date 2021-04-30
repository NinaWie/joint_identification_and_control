import torch.nn as nn
import torch


class Net(nn.Module):

    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        # conf: in channels, out channels, kernel size
        self.fc0 = nn.Linear(in_size, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, out_size)

    def forward(self, x):
        x = torch.tanh(self.fc0(x))
        # x = x * torch.from_numpy(np.array([0, 1, 1, 1]))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc_out(x))
        return x
