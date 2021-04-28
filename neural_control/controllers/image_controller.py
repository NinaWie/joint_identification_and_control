import numpy as np
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

from neural_control.dynamics.image_dynamics import (ImageDataset, ImgDynamics)
from neural_control.controllers.utils_image import (
    img_height, img_width, number_moves_knight, number_moves_ball,
    get_torch_img, get_img, radius, test_qualitatively, round_diff,
    one_hot_to_cmd_knight
)

# testing:
dynamics_path = "neural_control/dynamics/img_knight_cmd"
nr_actions = 1
learning_rate = 0.001
nr_epochs = 2000

knight_x_torch = torch.tensor([0, 2, 1, -1, -2, -2, -1, 1, 2])
knight_y_torch = torch.tensor([0, 1, 2, 2, 1, -1, -2, -2, -1])


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
        # index = 2000
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
        self.conv1 = nn.Conv2d(2, 6, 3)
        self.conv2 = nn.Conv2d(6, 10, 3)
        self.lin_img_1 = nn.Linear(width * height, 128)
        self.lin_img_2 = nn.Linear(width * height, 128)
        self.lin1 = nn.Linear(160, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, cmd_dim * nr_actions)

    def forward(self, img_1, img_2):
        both_images = torch.cat(
            (torch.unsqueeze(img_1, 1), torch.unsqueeze(img_2, 1)), dim=1
        )
        conv1 = torch.relu(self.conv1(both_images))
        conv2 = torch.relu(self.conv2(conv1))
        flattened = conv2.reshape((-1, 160))

        x1 = torch.relu(self.lin1(flattened))
        x2 = torch.relu(self.lin2(x1))
        x3 = self.lin3(x2)
        cmd = x3.reshape((-1, self.nr_actions, self.cmd_dim))
        action_probs = torch.softmax(cmd, dim=2)
        x_vals = torch.unsqueeze(
            torch.sum(action_probs * knight_x_torch, dim=2), dim=2
        )
        y_vals = torch.unsqueeze(
            torch.sum(action_probs * knight_y_torch, dim=2), dim=2
        )
        action = torch.cat((x_vals, y_vals), dim=2)
        return action


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
    min_loss = np.inf
    entropy_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

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
                ### sampling
                # replace action above with probs
                # m = Categorical(probs)
                # action_ind = m.sample()
                # action = one_hot(action_ind, num_classes=9)
                # action = action * (
                #     torch.unsqueeze(torch.exp(m.log_prob(action_ind)), 1)
                # )
                # print("probs", probs[0])
                # print(probs.size(), action_ind.size(), action.size())
                # print("action ind", action_ind)
                # print("action", action)
                # print(torch.exp(m.log_prob(action_ind)))
                ###
                current_state = dyn(current_state, action)
            # np.set_printoptions(precision=2, suppress=1)
            # print(action[0])
            # print(current_state[0].detach().numpy())
            # print(img_out[0])

            current_state_flat = current_state.reshape(
                (-1, img_width * img_height)
            )
            target_flat = img_out.reshape((-1, img_width * img_height)).float()
            # gt_argmax = torch.argmax(target_flat, dim=1)
            # loss_con = entropy_loss(current_state_flat, gt_argmax)
            # loss_con = (torch.sum((img_out - current_state)**2))
            loss_con = bce_loss(current_state_flat, target_flat)

            loss_con.backward()
            con_optimizer.step()
            epoch_loss += loss_con.item()

        losses.append(epoch_loss / i)
        if epoch % 20 == 0:
            print()
            print(f"Epoch {epoch} loss {round(epoch_loss / i * 100, 2)}")
            print("example command:", cmd_predicted[0])
            if losses[-1] <= min_loss:
                min_loss = losses[-1]
                torch.save(con, model_save_path + "_min")
            test_controller(model_save_path=None, con=con)

    torch.save(con, model_save_path)
    return losses


def test_controller(model_save_path=None, con=None, mode="receding"):
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
    if con is not None:
        controller = con
    else:
        controller = torch.load(model_save_path)

    dyn = torch.load(dynamics_path)
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
    for k, (img_1, img_2, center_1, center_2) in enumerate(testloader):
        # if k > 0:
        #     break
        # print(center_1, center_2)

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

                # cmd_np = pred_cmd[0].detach().numpy()
                # cmd_argmax = np.argmax(cmd_np)
                # print("command in testing", one_hot_to_cmd_knight(cmd_np))
                # print(pred_cmd.size(), current_state.size())
                out_dist = dyn(current_state, pred_cmd)
                current_center = torch.argmax(out_dist)
                current_center_x = current_center // img_height
                current_center_y = current_center % img_height
                current_center = (current_center_x, current_center_y)
                # apply
                # current_center = dataset.move_knight(
                #     current_center, cmd_argmax
                # )
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
        test_controller(model_path + "_min")
    test_qualitatively(model_path + "_min", dynamics_path, nr_actions)
