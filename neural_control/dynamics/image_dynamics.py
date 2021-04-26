import numpy as np
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from neural_control.controllers.utils_image import (
    img_height, img_width, knight_x_options, knight_y_options, ball_x_options,
    ball_y_options, radius, one_hot_to_cmd_knight, image_bounds
)

nr_epochs = 300
learning_rate = 0.01


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, width=20, height=20, radius=2):
        self.width = width
        self.height = height
        self.radius = radius

        nr_cells = width * height
        # (width - 2 * radius) * (height - 2 * radius)
        basic_images = np.zeros((nr_cells, width, height))
        centers = []

        count = 0
        for i in range(self.width):
            for j in range(self.height):
                centers.append([i, j])
                # avoid going across borders
                start_x = max([0, i - radius])
                end_x = min([i + radius + 1, width])
                start_y = max([0, j - radius])
                end_y = min([j + radius + 1, height])

                basic_images[count, start_x:end_x, start_y:end_y] = 1
                count += 1

        self.images = basic_images
        self.centers = centers

        # auxiliary array: get index from new center
        self.inverse_centers = {tuple(tup): j for j, tup in enumerate(centers)}

        # commands: currently just x and y control, can be -1, 0, 1 for left,
        # stay, right
        self.commands = np.arange(9)

        # TODO: could precompute dataset: indices of out img for each inp_img -
        # command combination

        # number of data
        self.nr_data = len(centers) * len(self.commands)
        self.nr_commands = len(self.commands)

    def move_ball(self, center, command_index):
        """
        center: current center of the ball
        u1: control in left right direction
        u2: control in up down
        """
        u1 = ball_x_options[command_index // 3]
        u2 = ball_y_options[command_index % 3]
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

    def move_knight(self, current_pos, move_ind):
        """
        Move ind defines one of nine moves
        """
        new_pos = np.array(current_pos) + np.array(
            [knight_x_options[move_ind], knight_y_options[move_ind]]
        )
        if np.all(new_pos >= 0) and np.all(new_pos < image_bounds):
            return new_pos.astype(int)
        else:
            return current_pos
        # # previous version: crossing the borders
        # x = int(new_pos[0] % self.width)
        # y = int(new_pos[1] % self.height)
        # return [x, y]

    def __len__(self):
        # can combine any center with any command
        return self.nr_data

    def __getitem__(self, index):
        chosen_center = self.centers[index // self.nr_commands]
        chosen_command = self.commands[index % self.nr_commands]

        img_index = self.inverse_centers[tuple(chosen_center)]

        next_center = self.move_knight(chosen_center, chosen_command)
        next_center_class = next_center[0] * self.width + next_center[1]
        # next_img_center = self.inverse_centers[tuple(next_center)]

        one_hot_cmd = np.zeros(self.nr_commands)
        one_hot_cmd[chosen_command] = 1
        return one_hot_cmd, self.images[img_index], next_center_class


class ImgDynamics(torch.nn.Module):

    def __init__(self, width, height, cmd_dim=9, trainable=True):
        super(ImgDynamics, self).__init__()
        self.lin1 = nn.Linear(width * height + cmd_dim, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 128)
        self.lin4 = nn.Linear(128, width * height)

        self.trainable = trainable

    def forward(self, x, cmd):
        inp = x.reshape((-1, x.size()[1] * x.size()[2]))
        img_and_cmd = torch.cat([inp, cmd], dim=1)
        x1 = torch.relu(self.lin1(img_and_cmd))
        x2 = torch.relu(self.lin2(x1))
        x3 = torch.relu(self.lin3(x2))
        x4 = self.lin4(x3)
        if not self.trainable:
            x4 = torch.softmax(x4, dim=1)
            x4 = x4.reshape((-1, x.size()[1], x.size()[2]))
        return x4


def train_dynamics(model_save_path):
    """Trains a model to learn the dynamics of the images

    Args:
        nr_epochs (int, optional): Defaults to 100.
    """
    dataset = ImageDataset(width=img_width, height=img_height, radius=radius)
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0
    )

    dyn = ImgDynamics(img_width, img_height)
    optimizer = optim.SGD(dyn.parameters(), lr=learning_rate, momentum=0.9)
    entropy_loss = nn.CrossEntropyLoss()

    for epoch in range(nr_epochs):
        epoch_loss = 0
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            (cmd, img_in, center_out) = data
            img_out_predicted = dyn(img_in.float(), cmd.float())

            # # previous version: center loss
            # center_loss = 0
            # for k in range(8):
            #     center_loss += (
            #         img_out_predicted[k, center_out[k, 0], center_out[k, 1]] -
            #         1
            #     )**2
            # torch.sum(
            #     (img_out - img_out_predicted)**2
            # ) + 10 * center_loss
            loss = entropy_loss(img_out_predicted, center_out)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} loss {round(epoch_loss*100 / i, 2)}")

    torch.save(dyn, model_save_path)


def test_dynamics(model_save_path):
    dyn = torch.load(model_save_path)

    # set some start center and a command
    x, y = (4, 5)
    cmd = 2

    test_img = torch.zeros(1, img_width, img_height)

    start_x = max([0, x - radius])
    end_x = min([x + radius + 1, img_width])
    start_y = max([0, y - radius])
    end_y = min([y + radius + 1, img_height])

    test_img[0, start_x:end_x, start_y:end_y] = 1

    test_cmd = torch.zeros(1, 9)
    test_cmd[0, cmd] = 1
    # test_cmd = torch.rand(1, 9)
    # print("random cmd", test_cmd)

    # run network
    pred_img = dyn(test_img, test_cmd)
    pred_img = torch.softmax(pred_img,
                             dim=1).reshape((-1, img_width, img_height))
    # print(pred_img)

    np_command = one_hot_to_cmd_knight(test_cmd)
    print("command", np_command)

    # compute desired out image
    out_img = torch.zeros(1, img_width, img_height)
    out_img[0, x + np_command[0] - radius:x + np_command[0] + 1 + radius,
            y + np_command[1] - radius:y + np_command[1] + radius + 1] = 1

    plt.subplot(1, 3, 1)
    plt.imshow(test_img[0].numpy())
    plt.title("Start state")
    plt.subplot(1, 3, 2)
    plt.imshow(out_img[0].numpy())
    plt.title("Desired output")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_img[0].detach().numpy())
    plt.title("Predicted")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default="test_model",
        help="Name to save or load model"
    )
    parser.add_argument(
        "-t", "--train", action="store_true", help="if 1 then train"
    )
    args = parser.parse_args()

    model_path = os.path.join(Path(__file__).parent.absolute(), args.save)
    if args.train:
        train_dynamics(model_path)
    else:
        test_dynamics(model_path)
