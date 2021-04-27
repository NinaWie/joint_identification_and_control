import numpy as np
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

from neural_control.dynamics.image_dynamics import (ImageDataset, ImgDynamics)
from neural_control.controllers.utils_image import (
    img_height, img_width, number_moves_knight, number_moves_ball,
    get_torch_img, get_img, radius, test_qualitatively
)

# testing:
dynamics_path = "neural_control/dynamics/img_dyn_knight_rand_2"
nr_actions = 1
learning_rate = 0.001
nr_epochs = 1000


class ImgConDataset(ImageDataset):

    def __init__(
        self,
        width=20,
        height=20,
        radius=1,
        nr_actions=4,
        return_centers=False
    ):
        super().__init__(width=width, height=height, radius=radius)
        self.return_centers = return_centers
        self.nr_images = len(self.images)

        self.feasible_combinations = []
        for i in range(self.nr_images):
            for j in range(self.nr_images):
                required_action = np.absolute(
                    np.array(self.centers[i]) - np.array(self.centers[j])
                )
                moves_required = number_moves_knight(*tuple(required_action))
                if moves_required <= nr_actions:
                    self.feasible_combinations.append((i, j))
        # how many feasible combinations do we have
        # self.nr_data = len(self.images)**2
        self.nr_data = len(self.feasible_combinations)

    def __len__(self):
        return self.nr_data

    def __getitem__(self, index):
        (img_1_index, img_2_index) = self.feasible_combinations[index]
        img_1 = self.images[img_1_index]
        img_2 = self.images[img_2_index]
        # print(self.centers[img_1_index], self.centers[img_2_index])
        if self.return_centers:
            return (
                img_1, img_2, self.centers[img_1_index],
                self.centers[img_2_index]
            )
        return img_1, img_2


class ImgController(torch.nn.Module):

    def __init__(self, width, height, nr_actions=1, cmd_dim=9):
        super(ImgController, self).__init__()
        self.nr_actions = nr_actions
        self.cmd_dim = cmd_dim
        self.lin_img_1 = nn.Linear(width * height, 128)
        self.lin_img_2 = nn.Linear(width * height, 128)
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, cmd_dim * nr_actions)

    def forward(self, img_1, img_2):
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        img_1 = img_1.reshape((-1, img_1.size()[1] * img_1.size()[2]))
        img_2 = img_2.reshape((-1, img_2.size()[1] * img_2.size()[2]))
        img_1_layer = torch.relu(self.lin_img_1(img_1))
        img_2_layer = torch.relu(self.lin_img_2(img_2))
        both_images = torch.cat([img_1_layer, img_2_layer], dim=1)

        x1 = torch.relu(self.lin1(both_images))
        x2 = torch.relu(self.lin2(x1))
        x3 = self.lin3(x2)
        cmd = x3.reshape((-1, self.nr_actions, self.cmd_dim))
        cmd = torch.sigmoid(cmd)**2
        return cmd


def train_controller(model_save_path):
    dyn = torch.load(dynamics_path)
    dyn.trainable = False
    for param in dyn.parameters():
        param.requires_grad = False

    dataset = ImgConDataset(
        width=img_width,
        height=img_height,
        radius=radius,
        nr_actions=nr_actions
    )
    print("Number of data", len(dataset))
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0
    )

    con = ImgController(img_width, img_height, nr_actions=nr_actions)
    con_optimizer = optim.SGD(con.parameters(), lr=learning_rate, momentum=0.9)

    losses = []

    for epoch in range(nr_epochs):
        epoch_loss = 0
        for i, data in enumerate(trainloader):
            con_optimizer.zero_grad()
            # Must not use cmd!!
            (img_in, img_out) = data
            # one image as reference

            # predict command with neural network
            cmd_predicted = con(img_in.float(), img_out.float())

            # pass through dynamics
            current_state = img_in.float()
            for action_ind in range(nr_actions):
                action = cmd_predicted[:, action_ind]
                current_state = dyn(current_state, action)

            # subtract first cmd to enforce high cmd in beginning
            loss_con = (torch.sum((img_out - current_state)**2))
            loss_con.backward()
            con_optimizer.step()
            epoch_loss += loss_con.item()

        losses.append(epoch_loss / i)
        if epoch % 20 == 0:
            print(f"Epoch {epoch} loss {round(epoch_loss / i, 2)}")
            print("example command:", cmd_predicted[0])

    torch.save(con, model_save_path)
    return losses


def test_controller(model_save_path, mode="all"):
    """Run test of all possible combinations

    Args:
        model_save_path ([type]): Path to laod model from
        mode (str, optional): Receding horizon or all at once.
            Defaults to "receding".

    Returns:
        [type]: [description]
    """

    # helper function
    def center_to_np(center):
        return np.array([center[0].item(), center[1].item()])

    # initialize controller and loader
    controller = torch.load(model_save_path)
    dataset = ImgConDataset(
        width=img_width,
        height=img_height,
        radius=radius,
        nr_actions=nr_actions,
        return_centers=True
    )
    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    correct = 0
    for (img_1, img_2, center_1, center_2) in testloader:

        current_state = img_1.clone().float()
        current_center = center_1

        pred_cmd_sequence = controller(current_state, img_2.float())
        # print(pred_cmd_sequence)

        # predict action, apply
        with torch.no_grad():
            for i in range(nr_actions):
                # predict with receding horizon
                if mode == "receding":
                    pred_cmd = controller(current_state, img_2.float())[:, 0]
                else:
                    pred_cmd = pred_cmd_sequence[:, i]

                cmd_np = pred_cmd[0].detach().numpy()
                cmd_argmax = np.argmax(cmd_np)
                print(cmd_argmax)

                # apply
                current_center = dataset.move_knight(
                    current_center, cmd_argmax
                )
                # print(cmd_rounded, current_center)
                current_state = get_torch_img(
                    get_img(img_width, img_height, current_center, radius)
                )
        if np.all(center_to_np(center_2) == current_center):
            correct += 1
        # else:
        #     print(
        #         current_center, center_to_np(center_2), center_to_np(center_1)
        #     )

    print(
        "Correct", correct, "all", len(testloader), "Accuracy:",
        correct / len(testloader)
    )


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

    model_path = "trained_models/img/" + args.save
    if args.train:
        losses = train_controller(model_path)
        plt.plot(losses)
        plt.show()
    else:
        test_controller(model_path)
    test_qualitatively(model_path, dynamics_path, nr_actions)
