import torch.nn as nn
import torch
import torch.nn.functional as F


class Net(nn.Module):
    """
    Simple MLP with three hidden layers, based on RL work of Marco Hutter's
    group
    """

    def __init__(
        self,
        state_dim,
        ref_length,
        ref_dim,
        out_size,
        conv=True,
        hist_conv=False
    ):
        """
        in_size: number of input neurons (features)
        out_size: number of output neurons
        """
        super(Net, self).__init__()
        self.hist_conv = hist_conv
        self.states_in = nn.Linear(state_dim, 64)
        self.conv_ref = nn.Conv1d(ref_dim, 20, kernel_size=3)
        self.conv_history = nn.Conv1d(state_dim, 20, kernel_size=3)
        # the size will be nr_channels * (1dlength - kernel_size + 1)
        self.ref_length = ref_length
        self.conv = conv
        self.reshape_len = 20 * (ref_length - 2) if conv else 64
        self.ref_in = nn.Linear(ref_length * ref_dim, 64)
        state_dim_prev = 60 if self.hist_conv else 64
        self.fc1 = nn.Linear(state_dim_prev + self.reshape_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, out_size)

    def forward(self, history, ref, is_seq=0, hist_conv=0):
        # process state and reference differently
        # for old sequence models put False
        if hist_conv:
            history = torch.transpose(history, 1, 2)
            history = torch.relu(self.conv_history(history))
            state = torch.reshape(history, (-1, 60))
        else:
            if is_seq:
                history = torch.reshape(
                    history, (-1, history.size()[1] * history.size()[2])
                )
            state = torch.tanh(self.states_in(history))
        if self.conv:
            # ref = torch.reshape(ref, (-1, self.ref_dim, 3))
            ref = torch.transpose(ref, 1, 2)
            ref = torch.relu(self.conv_ref(ref))
            ref = torch.reshape(ref, (-1, self.reshape_len))
        else:
            ref = torch.tanh(self.ref_in(ref))
        # concatenate
        x = torch.hstack((state, ref))
        # normal feed-forward
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc_out(x)
        return x
