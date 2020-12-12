import os
import time
import argparse
import numpy as np
import torch

from environments.cartpole_env import CartPoleEnv
from dataset import raw_states_to_torch
from models.resnet_like_model import Net
from cartpole_loss import control_loss_function

APPLY_UNTIL = 1


class Evaluator:

    def __init__(self, std=1):
        self.std = std

    def make_swingup(
        self, net, nr_iters=10, max_iters=50, success_over=20, render=False
    ):
        """
        Check if the pendulum can make a swing up
        """
        data_collection = []
        # average over 50 runs # 10 is length of action sequence
        success = []  # np.zeros((success_over * 10 * nr_iters, 4))
        eval_env = CartPoleEnv()
        collect_loss = []
        with torch.no_grad():
            for it in range(nr_iters):
                # # Set angle to somewhere at the bottom # TODO
                random_hanging_state = (np.random.rand(4) - .5)
                random_hanging_state[2] = (-1) * (
                    (np.random.rand() > .5) * 2 - 1
                ) * (1 - (np.random.rand() * .2)) * np.pi
                eval_env.state = random_hanging_state

                # # Set angle to somewhere on top
                # eval_env.state[2] = (np.random.rand(1) - .5) * .2

                # set x position to zero
                # random_hanging_state[0] = 0 TODO
                new_state = eval_env.state

                # Start balancing
                for j in range(max_iters + success_over):
                    # Transform state in the same way as the training data
                    # and normalize
                    torch_state = raw_states_to_torch(new_state, std=self.std)
                    # Predict optimal action:
                    predicted_action = net(torch_state)
                    collect_loss.append(
                        control_loss_function(predicted_action,
                                              torch_state).item()
                    )
                    action_seq = torch.sigmoid(predicted_action) - .5
                    # print([round(act, 2) for act in action_seq[0].numpy()])
                    # if render:
                    #     print("state before", new_state)
                    # print("new action seq", action_seq[0].numpy())
                    # print()
                    for action in action_seq[0].numpy()[:APPLY_UNTIL]:
                        # run action in environment
                        new_state, _, _, _ = eval_env._step(action)
                        data_collection.append(new_state)
                        # print(new_state)
                        if render:
                            eval_env._render()
                            time.sleep(.1)
                        if j >= max_iters:
                            success.append(new_state)
                        # check only whether it was able to swing up the pendulum
                        # if np.abs(new_state[2]) < np.pi / 15 and not render:
                        #     made_it = 1
                        #     break
                # success[it] = made_it
                eval_env._reset()
        success = np.absolute(np.array(success))
        mean_rounded = [round(m, 2) for m in np.mean(success, axis=0)]
        std_rounded = [round(m, 2) for m in np.std(success, axis=0)]
        return mean_rounded, std_rounded, collect_loss, data_collection

    def evaluate_in_environment(self, net, nr_iters=1, render=False):
        """
        Measure success --> how long can we balance the pole on top
        """
        eval_env = CartPoleEnv()
        with torch.no_grad():
            success = np.zeros(nr_iters)
            # observe also the oscillation
            avg_angle = np.zeros(nr_iters)
            for it in range(nr_iters):
                # only set the theta to the top, and reduce speed
                eval_env.state = eval_env.state * .25
                eval_env.state[2] = (np.random.rand(1) - .5) * .2
                is_fine = False
                episode_length_counter = 0
                new_state = eval_env.state

                angles = list()
                # Start balancing
                while not is_fine:
                    # Transform state in the same way as the training data
                    # and normalize
                    torch_state = raw_states_to_torch(new_state, std=self.std)
                    # Predict optimal action:
                    action_seq = torch.sigmoid(net(torch_state)) - .5
                    for action in action_seq[0].numpy()[:APPLY_UNTIL]:
                        # run action in environment
                        new_state, _, is_fine, _ = eval_env._step(action)
                        angles.append(np.absolute(new_state[2]))
                        if render:
                            eval_env._render()
                            time.sleep(.1)
                        # track number of timesteps until failure
                        episode_length_counter += 1
                    # if render:
                    #     print("state after", new_state)
                    if episode_length_counter > 250:
                        break
                avg_angle[it] = np.mean(angles)
                success[it] = episode_length_counter
                eval_env._reset()
        return success, avg_angle


def run_saved_arr(path):
    """
    Load a saved sequence of states and visualize it
    """
    states = np.load(path)
    eval_env = CartPoleEnv()
    for state in states:
        eval_env.state = state
        eval_env._render()
        time.sleep(.1)


if __name__ == "__main__":
    # make as args:
    parser = argparse.ArgumentParser("Model directory as argument")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="minimize_x",
        help="Directory of model"
    )
    parser.add_argument(
        "-e", "--epoch", type=str, default="", help="Saved epoch"
    )
    parser.add_argument(
        "-save_data",
        action="store_true",
        help="save the episode as training data"
    )
    args = parser.parse_args()

    MODEL_NAME = args.model  # "theta_max_normalize"  # "best_model_2"

    # run a saver sequence
    # run_saved_arr("saved_states.npy")
    # exit()

    net = torch.load(
        os.path.join(
            "trained_models", MODEL_NAME, "model_pendulum" + args.epoch
        )
    )
    net.eval()

    evaluator = Evaluator()
    # angles = evaluator.run_for_fixed_length(net, render=True)
    # success, angles = evaluator.evaluate_in_environment(net, render=True)
    try:
        swingup_mean, swingup_std, _, data_collection = evaluator.make_swingup(
            net, nr_iters=1, render=True, max_iters=300
        )
        print(swingup_mean, swingup_std)
    except KeyboardInterrupt:
        pass
    # Save sequence?
    if len(input("save? enter anything for yes")) > 0:
        data_collection = np.asarray(data_collection)
        np.save("saved_states.npy", data_collection)
