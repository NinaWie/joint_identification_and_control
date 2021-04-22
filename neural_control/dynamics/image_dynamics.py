import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, width=20, height=20, radius=2):
        self.width = width
        self.height = height
        self.radius = radius

        nr_cells = (width - 2 * radius) * (height - 2 * radius)
        basic_images = np.zeros((nr_cells, width, height))
        centers = []

        count = 0
        for i in range(radius, self.width - radius):
            for j in range(radius, self.height - radius):
                centers.append([i, j])
                basic_images[count, i - radius:i + radius + 1,
                             j - radius:j + radius + 1] = 1
                count += 1

        self.images = basic_images  # torch.from_numpy(basic_images).float()
        self.centers = centers  # torch.tensor(centers).float()

        # auxiliary array: get index from new center
        self.inverse_centers = {tuple(tup): j for j, tup in enumerate(centers)}

        # commands: currently just x and y control, can be -1, 0, 1 for left,
        # stay, right
        control_options = [-1, 0, 1]
        self.commands = np.array(
            [
                [move_x, move_y] for move_x in control_options
                for move_y in control_options
            ]
        )

        # TODO: could precompute dataset: indices of out img for each inp_img -
        # command combination

        # number of data
        self.nr_data = len(centers) * len(self.commands)
        self.nr_commands = len(self.commands)

    def move_ball(self, center, u1, u2):
        """
        center: current center of the ball
        u1: control in left right direction
        u2: control in up down
        """
        x = max(
            [self.radius,
             min([center[0] + u1, self.width - self.radius - 1])]
        )
        y = max(
            [
                self.radius,
                min([center[1] + u2, self.height - self.radius - 1])
            ]
        )
        return [int(x), int(y)]

    def __len__(self):
        # can combine any center with any command
        return self.nr_data

    def __getitem__(self, index):
        chosen_center = self.centers[index // self.nr_commands]
        chosen_command = self.commands[index % self.nr_commands]

        img_index = self.inverse_centers[tuple(chosen_center)]

        next_center = self.move_ball(
            chosen_center, chosen_command[0], chosen_command[1]
        )
        next_img_center = self.inverse_centers[tuple(next_center)]

        return chosen_command, self.images[img_index], self.images[
            next_img_center]


class ImgDynamics(torch.nn.Module):

    def __init__(self, width, height, cmd_dim=2):
        super(ImgDynamics, self).__init__()
        self.lin1 = nn.Linear(width * height + cmd_dim, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 128)
        self.lin4 = nn.Linear(128, width * height)

    def forward(self, x, cmd):
        inp = x.reshape((-1, x.size()[1] * x.size()[2]))
        img_and_cmd = torch.cat([inp, cmd], dim=1)
        x1 = torch.relu(self.lin1(img_and_cmd))
        x2 = torch.relu(self.lin2(x1))
        x3 = torch.relu(self.lin3(x2))
        x4 = torch.sigmoid(self.lin4(x3))
        x4 = x4.reshape((-1, x.size()[1], x.size()[2]))
        return x4


def train_dynamics(model_save_path, nr_epochs=100):
    """Trains a model to learn the dynamics of the images

    Args:
        nr_epochs (int, optional): Defaults to 100.
    """
    dataset = ImageDataset(width=8, height=8, radius=1)
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0
    )

    dyn = ImgDynamics(8, 8)
    optimizer = optim.SGD(dyn.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(nr_epochs):
        epoch_loss = 0
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            (cmd, img_in, img_out) = data
            # print(
            img_out_predicted = dyn(img_in.float(), cmd.float())

            loss = torch.sum((img_out - img_out_predicted)**2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} loss {round(epoch_loss / i, 2)}")

    torch.save(dyn, model_save_path)


def test_dynamics(model_save_path):
    dyn = torch.load(model_save_path)

    test_img = torch.zeros(1, 8, 8)
    test_img[0, 4:7, 3:6] = 1
    test_cmd = torch.tensor([[-2, -2]])
    pred_img = dyn(test_img, test_cmd)

    plt.subplot(1, 2, 1)
    plt.imshow(test_img[0].numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(pred_img[0].detach().numpy())
    plt.show()


if __name__ == "__main__":
    model_save_path = os.path.join(
        Path(__file__).parent.absolute(), "image_model"
    )
    # train_dynamics(model_save_path)
    test_dynamics(model_save_path)
