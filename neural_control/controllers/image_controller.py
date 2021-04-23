import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from neural_control.dynamics.image_dynamics import ImageDataset, ImgDynamics


class ImgConDataset(ImageDataset):

    def __init__(self, width=20, height=20, radius=1):
        super().__init__(width=width, height=height, radius=radius)

        # we can combine all images with each other
        self.nr_data = len(self.images)**2
        self.nr_images = len(self.images)

    def __len__(self):
        return self.nr_data

    def __getitem__(self, index):
        img_1 = self.images[index // self.nr_images]
        img_2 = self.images[index % self.nr_images]
        return img_1, img_2


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
learning_rate = 0.00001


def train_controller(model_save_path):
    dyn = torch.load(dynamics_path)

    dataset = ImgConDataset(width=8, height=8, radius=1)
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0
    )

    con = ImgController(8, 8, nr_actions=nr_actions)
    con_optimizer = optim.SGD(con.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(100):
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
    test_img_out[0, 5:8, 2:5] = 1

    pred_cmd = con(test_img, test_img_out)
    print("predicted command", pred_cmd)
    pred_img = dyn(test_img, pred_cmd[:, 0])

    plt.subplot(1, 3, 1)
    plt.imshow(test_img[0].numpy())
    plt.title("Input img")
    plt.subplot(1, 3, 2)
    plt.imshow(test_img_out[0].detach().numpy())
    plt.title("Desired out img")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_img[0].detach().numpy())
    plt.show()


if __name__ == "__main__":
    model_save_path = "trained_models/img/four_action"
    train_controller(model_save_path)
    # test_controller(model_save_path)
