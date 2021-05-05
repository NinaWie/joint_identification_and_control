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
        x[:, 0] *= 0
        x = torch.tanh(self.fc0(x))
        # x = x * torch.from_numpy(np.array([0, 1, 1, 1]))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc_out(x))
        return x


class ImageControllerNet(nn.Module):

    def __init__(self, out_size=1, nr_img=5):
        super(ImageControllerNet, self).__init__()
        # all raw images and the subtraction
        self.conv1 = nn.Conv2d(nr_img * 2 - 1, 10, 5)
        self.conv2 = nn.Conv2d(10, 2, 3)

        self.flat_img_size = 2 * 94 * 294

        self.fc1 = nn.Linear(self.flat_img_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, out_size)

    def forward(self, image):
        cat_all = [image]
        for i in range(image.size()[1] - 1):
            cat_all.append(
                torch.unsqueeze(image[:, i + 1] - image[:, i], dim=1)
            )
        # img_sub1 = torch.unsqueeze(image[:, 1] - image[:, 0], dim=1)
        # img_sub2 = torch.unsqueeze(image[:, 2] - image[:, 1], dim=1)
        sub_images = torch.cat(cat_all, dim=1)
        conv1 = torch.relu(self.conv1(sub_images.float()))
        conv2 = torch.relu(self.conv2(conv1))

        x = conv2.reshape((-1, self.flat_img_size))

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc_out(x))
        return x
