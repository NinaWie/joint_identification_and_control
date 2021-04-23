import numpy as np
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from neural_control.dynamics.image_dynamics import ImageDataset, ImgDynamics


class ImgConDataset(ImageDataset):

    def __init__(self, width=20, height=20, radius=1, nr_actions=4):
        super().__init__(width=width, height=height, radius=radius)

        self.nr_images = len(self.images)

        self.feasible_combinations = []
        for i in range(self.nr_images):
            for j in range(self.nr_images):
                required_action = np.absolute(
                    np.array(self.centers[i]) - np.array(self.centers[j])
                )
                if np.all(required_action <= nr_actions):
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

    def __init__(self, width, height, nr_actions=1, cmd_dim=2):
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
        x3 = torch.tanh(self.lin3(x2))
        cmd = x3.reshape((-1, self.nr_actions, self.cmd_dim))
        return cmd


# testing:
dynamics_path = "neural_control/dynamics/image_model"
nr_actions = 4
learning_rate = 0.00005
nr_epochs = 200


def train_controller(model_save_path):
    dyn = torch.load(dynamics_path)

    dataset = ImgConDataset(width=8, height=8, radius=1, nr_actions=nr_actions)
    print("Number of data", len(dataset))
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0
    )

    con = ImgController(8, 8, nr_actions=nr_actions)
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

            loss_con = torch.sum((img_out - current_state)**2)
            loss_con.backward()
            con_optimizer.step()
            epoch_loss += loss_con.item()

        print(f"Epoch {epoch} loss {round(epoch_loss / i, 2)}")
        if epoch % 20 == 0:
            print("example command:", cmd_predicted[0])

    torch.save(con, model_save_path)


def test_controller(model_save_path):
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
