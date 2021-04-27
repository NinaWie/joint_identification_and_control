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
    ball_y_options, radius, one_hot_to_cmd_knight, image_bounds, round_diff
)

nr_epochs = 2000
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


class NewImgDataset(ImageDataset):

    def __init__(self, width=20, height=20, radius=1):
        super().__init__(width=width, height=height, radius=radius)

    def __getitem__(self, index):
        # retrieve state image
        chosen_center = self.centers[index // self.nr_commands]
        img_index = self.inverse_centers[tuple(chosen_center)]

        # construct command
        if index % 5 == 0:
            cmd_ind = self.commands[index % self.nr_commands]
            rand_cmd = torch.zeros(9)
            rand_cmd[cmd_ind] = 1
        else:
            rand_cmd = torch.rand(9)**2

        new_center = self.move_knight(chosen_center, np.argmax(rand_cmd))

        return rand_cmd, self.images[
            img_index], new_center[0] * self.width + new_center[1]


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
    dataset = NewImgDataset(width=img_width, height=img_height, radius=radius)
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

            loss = entropy_loss(img_out_predicted, center_out)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 30 == 0:
            print(f"Epoch {epoch} loss {round(epoch_loss*100 / i, 2)}")
            test_dynamics(dyn=dyn)

    torch.save(dyn, model_save_path)


def test_dynamics(dyn=None, model_save_path=None):
    """
    Run test of all possible combinations
    """

    # initialize controller and loader
    if dyn is None:
        dyn = torch.load(model_save_path)
    dataset = NewImgDataset(width=img_width, height=img_height, radius=radius)
    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )
    correct = 0
    for (cmd, img_in, center_out) in testloader:
        # ATTENTION: only testing accuracy on one hots - otherwise comment:
        one_hot_cmd = torch.zeros(cmd.size())
        one_hot_cmd[0, torch.argmax(cmd)] = 1

        with torch.no_grad():
            img_out_predicted = dyn(img_in.float(), one_hot_cmd.float())
        max_arg = torch.argmax(img_out_predicted)
        if max_arg == center_out:
            correct += 1
    print(
        "Correct", correct, "all", len(testloader), "Accuracy:",
        correct / len(testloader)
    )


def test_qualitative(model_save_path):
    dyn = torch.load(model_save_path)

    dataset = NewImgDataset(width=img_width, height=img_height, radius=radius)
    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0
    )
    for i, (test_cmd, img_in, center_out) in enumerate(testloader):
        if i > 0:
            break
    # # optionally check what happens if it's very evenly distributed
    # test_cmd = torch.softmax(test_cmd, dim=1)

    # run network
    pred_img = dyn(img_in.float(), test_cmd.float())
    pred_img = torch.softmax(pred_img,
                             dim=1).reshape((-1, img_width, img_height))

    np_command = one_hot_to_cmd_knight(test_cmd)
    print("raw cmd", test_cmd)
    print("command", np_command)

    # compute desired out image
    target_cen_x, target_cen_y = [
        center_out // img_width, center_out % img_width
    ]
    out_img = torch.zeros(1, img_width, img_height)
    out_img[0, target_cen_x, target_cen_y] = 1

    plt.subplot(1, 3, 1)
    plt.imshow(img_in[0].numpy())
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
        test_dynamics(model_save_path=model_path)
        test_qualitative(model_path)
