import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from neural_control.models.simple_model import StateToImg
from neural_control.dataset import StateImageDataset

data = StateImageDataset(load_data_path="data/state_to_img.npz")
trainloader = torch.utils.data.DataLoader(
    data, batch_size=1, shuffle=True, num_workers=0
)

state_img_net = StateToImg()
optimizer = torch.optim.SGD(
    state_img_net.parameters(), lr=0.0001, momentum=0.9
)

# TRAIN
for epoch in range(200):
    running_loss = 0
    for i, (state_inp, img_gt) in enumerate(trainloader):
        optimizer.zero_grad()

        img_pred = state_img_net(state_inp.float())

        # backprop
        loss = torch.sum((img_pred - img_gt)**2)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(running_loss / i)

# Testing
with torch.no_grad():
    test_state = torch.tensor([[0.1, 0.1]])
    out_img = state_img_net(test_state)
    plt.imshow(out_img[0].numpy())
    plt.show()
    # other direction
    test_state = torch.tensor([[-0.1, -0.1]])
    out_img = state_img_net(test_state)
    plt.imshow(out_img[0].numpy())
    plt.show()

torch.save(state_img_net, "trained_models/cartpole/state_img_net")
