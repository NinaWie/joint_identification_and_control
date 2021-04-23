import numpy as np
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from neural_control.dynamics.image_dynamics import (
    ImageDataset, ImgDynamics, one_hot_to_cmd_knight, one_hot_to_cmd_ball
)


def number_moves_knight(x, y):
    # switch if necessary
    if x < y:
        temp = x
        x = y
        y = temp

    # 2 corner cases
    if x == 1 and y == 0:
        return 3
    if x == 2 and y == 2:
        return 4

    delta = x - y

    if y > delta:
        return delta - 2 * np.floor((delta - y) / 3)
    else:
        return delta - 2 * np.floor((delta - y) / 4)


def number_moves_ball(x, y):
    return max([x, y])


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


def get_img(img_width, img_height, center, radius):
    img = np.zeros((img_width, img_height))
    img[center[0] - radius:center[0] + radius + 1,
        center[1] - radius:center[1] + radius + 1] = 1
    return img


def get_torch_img(np_img):
    return torch.tensor([np_img.tolist()]).float()


def round_diff(x):
    slope = 10
    e = torch.exp(slope * (x - .5))
    return e / (e + 1)


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
        img_1 = img_1.reshape((-1, img_1.size()[1] * img_1.size()[2]))
        img_2 = img_2.reshape((-1, img_2.size()[1] * img_2.size()[2]))
        img_1_layer = torch.relu(self.lin_img_1(img_1))
        img_2_layer = torch.relu(self.lin_img_2(img_2))
        both_images = torch.cat([img_1_layer, img_2_layer], dim=1)

        x1 = torch.relu(self.lin1(both_images))
        x2 = torch.relu(self.lin2(x1))
        x3 = self.lin3(x2)
        cmd = x3.reshape((-1, self.nr_actions, self.cmd_dim))
        cmd = torch.softmax(cmd, dim=2)
        return cmd


# testing:
dynamics_path = "neural_control/dynamics/image_dyn_knight"
nr_actions = 1
radius = 1
img_width, img_height = (8, 8)
learning_rate = 0.00005
nr_epochs = 400


def train_controller(model_save_path):
    dyn = torch.load(dynamics_path)

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
                if action_ind == 0:
                    intermediate = current_state

            # subtract first cmd to enforce high cmd in beginning
            loss_con = (
                torch.sum((img_out - current_state)**2) +
                # add loss of first state to allow receding horizon!
                torch.sum((img_out - intermediate)**2)
            )
            loss_con.backward()
            con_optimizer.step()
            epoch_loss += loss_con.item()

        print(f"Epoch {epoch} loss {round(epoch_loss / i, 2)}")
        if epoch % 20 == 0:
            print("example command:", cmd_predicted[0])

    torch.save(con, model_save_path)


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

    print(
        "Correct", correct, "all", len(testloader), "Accuracy:",
        correct / len(testloader)
    )


def test_with_dyn_model(model_save_path):
    con = torch.load(model_save_path)
    dyn = torch.load(dynamics_path)

    test_img = torch.zeros(1, 8, 8)
    test_img[0, 4:7, 3:6] = 1

    test_img_out = torch.zeros(1, 8, 8)
    test_img_out[0, 0:3, 0:3] = 1

    current_state = test_img.clone().float()
    # apply all at once
    # pred_cmd = con(test_img, test_img_out)
    # print("predicted command", pred_cmd)
    # for action_ind in range(nr_actions):
    #     current_state = dyn(current_state, pred_cmd[:, action_ind])

    plt.figure(figsize=(10, 10))
    # Apply in receding horizon
    for i in range(nr_actions):
        pred_cmd = con(current_state, test_img_out)
        current_state_before = current_state.clone()
        print("predicted command", pred_cmd[0, 0])
        current_state = dyn(current_state, pred_cmd[:, 0])

        print(1 + i * 4)
        plt.subplot(4, 3, 1 + i * 3)
        plt.imshow(current_state_before[0].detach().numpy())
        plt.title("Input img")
        plt.subplot(4, 3, 2 + i * 3)
        plt.imshow(test_img_out[0].detach().numpy())
        plt.title("Target img")
        plt.subplot(4, 3, 3 + i * 3)
        plt.imshow(current_state[0].detach().numpy())
        plt.title(
            "Applying learnt\n command " +
            str(np.around(pred_cmd[0, 0].detach().numpy(), 2).tolist())
        )
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

    model_path = "trained_models/img/" + args.save
    if args.train:
        train_controller(model_path)
    else:
        test_controller(model_path)
