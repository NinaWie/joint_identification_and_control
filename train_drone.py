import os
import json
import time
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F

from dataset import Dataset
from drone_loss import drone_loss_function, trajectory_loss
from environments.drone_dynamics import simulate_quadrotor
from evaluate_drone import QuadEvaluator
from models.hutter_model import Net
from environments.mpc_drone_env import construct_states
from utils.plotting import plot_loss, plot_success

EPOCH_SIZE = 5000
USE_NEW_DATA = 250
PRINT = (EPOCH_SIZE // 30)
NR_EPOCHS = 200
BATCH_SIZE = 8
NR_EVAL_ITERS = 5
STATE_SIZE = 16
NR_ACTIONS = 5
ACTION_DIM = 4
LEARNING_RATE = 0.001
SAVE = os.path.join("trained_models/drone/test_model")
BASE_MODEL = None  # os.path.join("trained_models/drone/first_traj_model")
BASE_MODEL_NAME = 'model_quad'

# Load model or initialize model
if BASE_MODEL is not None:
    net = torch.load(os.path.join(BASE_MODEL, BASE_MODEL_NAME))
    # load std or other parameters from json
    with open(os.path.join(BASE_MODEL, "param_dict.json"), "r") as outfile:
        param_dict = json.load(outfile)
    STD = np.array(param_dict["std"]).astype(float)
    MEAN = np.array(param_dict["mean"]).astype(float)
else:
    net = Net(STATE_SIZE, ACTION_DIM * NR_ACTIONS)
    reference_data = Dataset(
        construct_states, normalize=True, num_states=EPOCH_SIZE
    )
    (STD, MEAN) = (reference_data.std, reference_data.mean)

# Use cuda if available
global device
device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# define optimizer and torch normalization parameters
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

torch_mean, torch_std = (
    torch.from_numpy(MEAN).float(), torch.from_numpy(STD).float()
)

# save std for normalization during test time
param_dict = {"std": STD.tolist(), "mean": MEAN.tolist()}
with open(os.path.join(SAVE, "param_dict.json"), "w") as outfile:
    json.dump(param_dict, outfile)

loss_list, success_mean_list, success_std_list = list(), list(), list()

# target_state = torch.zeros(STATE_SIZE)
# mask = torch.ones(STATE_SIZE)
# mask[6:13] = 0  # rotor speeds and xyz velocity don't matter
# loss_weights = mask.clone()
# target_state = ((target_state - torch_mean) / torch_std) * mask


def adjust_learning_rate(optimizer, epoch, every_x=5):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = LEARNING_RATE * (0.1**(epoch // every_x))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


highest_success = 0
for epoch in range(NR_EPOCHS):

    # Generate data dynamically
    if epoch % 2 == 0:
        state_data = Dataset(
            construct_states,
            normalize=True,
            mean=MEAN,
            std=STD,
            num_states=EPOCH_SIZE,
            # reset_strength=.6 + epoch / 50
        )

    print()
    print(f"Epoch {epoch} (before)")
    eval_env = QuadEvaluator(net, MEAN, STD)
    _ = eval_env.stabilize(nr_iters=5)
    suc_mean, suc_std, new_data = eval_env.evaluate(
        nr_hover_iters=NR_EVAL_ITERS, nr_traj_iters=NR_EVAL_ITERS
    )
    success_mean_list.append(suc_mean)
    success_std_list.append(suc_std)
    if epoch > 0:
        if suc_mean > highest_success:
            highest_success = suc_mean
            print("Best model")
            torch.save(net, os.path.join(SAVE, "model_quad" + str(epoch)))
        print("Loss:", round(running_loss / i, 2))

    # self-play: add acquired data
    if USE_NEW_DATA > 0 and epoch > 2 and len(new_data) > 0:
        rand_inds_include = np.random.permutation(len(new_data))[:USE_NEW_DATA]
        selected_new_data = np.array(new_data)[rand_inds_include]
        # np.save("selected_new_data.npy", selected_new_data)
        state_data.add_data(selected_new_data)
        # if (epoch + 1) % 10 == 0:
        #     np.save("check_added_data.npy", np.array(new_data))
        print("new added data:", selected_new_data.shape)

    # Initialize train loader
    trainloader = torch.utils.data.DataLoader(
        state_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    # Training
    tic_epoch = time.time()
    running_loss = 0
    try:
        for i, data in enumerate(trainloader, 0):
            # inputs are normalized states, current state is unnormalized in
            # order to correctly apply the action
            inputs, current_state = data

            # zero the parameter gradients
            optimizer.zero_grad()

            loss_weights = 1 + torch.abs(current_state)  # TODO

            # ------------ VERSION 1 (x states at once)-----------------
            actions = net(inputs)
            actions = torch.sigmoid(actions)
            action_seq = torch.reshape(actions, (-1, NR_ACTIONS, ACTION_DIM))
            # loss = 0
            start_state = current_state.clone()
            intermediate_states = []
            for k in range(NR_ACTIONS):
                # normalize loss by the start distance
                action = action_seq[:, k]
                # ----------- VERSION 2: predict one action at a time --------
                # start_dist = torch.sum(current_state[:, :3]**2, axis=1)
                # net_input_state = (current_state - torch_mean) / torch_std
                # action = net(net_input_state)
                # action = torch.sigmoid(action)
                current_state = simulate_quadrotor(action, current_state)
                intermediate_states.append(current_state)

                # Only compute loss after last action
                # 1) --------- drone loss function --------------
            loss = drone_loss_function(
                current_state, start_state=start_state, printout=0
            )
            # ------------- VERSION 3: Trajectory loss -------------
            # drone_state = (current_state - torch_mean) / torch_std
            # loss = trajectory_loss(
            #     inputs,
            #     target_state,
            #     drone_state,
            #     loss_weights=loss_weights,
            #     mask=mask,
            #     printout=0
            # )

            # Backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_list.append(running_loss / i)
    except KeyboardInterrupt:
        break
    print("time one epoch", time.time() - tic_epoch)

if not os.path.exists(SAVE):
    os.makedirs(SAVE)

# Save model
torch.save(net, os.path.join(SAVE, "model_quad"))
plot_loss(loss_list, SAVE)
plot_success(success_mean_list, success_std_list, SAVE)
print("finished and saved.")
