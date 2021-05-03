import os
import time
import argparse
import json
import numpy as np
import torch

from neural_control.environments.cartpole_env import CartPoleEnv
from neural_control.dynamics.cartpole_dynamics import CartpoleDynamics
from neural_control.controllers.mpc import MPC
from neural_control.controllers.network_wrapper import CartpoleWrapper
from neural_control.models.resnet_like_model import Net

APPLY_UNTIL = 1


class Evaluator:

    def __init__(self, controller, eval_env):
        self.controller = controller
        self.eval_env = eval_env
        self.mpc = isinstance(self.controller, MPC)

    def make_swingup(
        self, net, nr_iters=10, max_iters=100, success_over=20, render=False
    ):
        """
        Check if the pendulum can make a swing up
        """
        data_collection = []
        # average over 50 runs # 10 is length of action sequence
        success = []  # np.zeros((success_over * 10 * nr_iters, 4))
        with torch.no_grad():
            for it in range(nr_iters):
                # # Set angle to somewhere at the bottom
                random_hanging_state = (np.random.rand(4) - .5)
                random_hanging_state[2] = (-1) * (
                    (np.random.rand() > .5) * 2 - 1
                ) * (1 - (np.random.rand() * .2)) * np.pi
                self.eval_env.state = random_hanging_state

                # # Set angle to somewhere on top
                # self.eval_env.state[2] = (np.random.rand(1) - .5) * .2

                # set x position to zero
                new_state = self.eval_env.state

                # Start balancing
                for j in range(max_iters + success_over):
                    # Transform state in the same way as the training data
                    # and normalize
                    torch_state = raw_states_to_torch(new_state, std=self.std)
                    # Predict optimal action:
                    action_seq = net(torch_state)
                    # print([round(act, 2) for act in action_seq[0].numpy()])
                    # if render:
                    #     print("state before", new_state)
                    # print("new action seq", action_seq[0].numpy())
                    # print()
                    for action_ind in range(APPLY_UNTIL):
                        # run action in environment
                        new_state = self.eval_env._step(
                            action_seq[:, action_ind]
                        )
                        data_collection.append(new_state)
                        # print(new_state)
                        if render:
                            self.eval_env._render()
                            time.sleep(.1)
                        if j >= max_iters:
                            success.append(new_state)
                        # check only whether it was able to swing up the pendulum
                        # if np.abs(new_state[2]) < np.pi / 15 and not render:
                        #     made_it = 1
                        #     break
                # success[it] = made_it
                self.eval_env._reset()
        success = np.absolute(np.array(success))
        mean_rounded = [round(m, 2) for m in np.mean(success, axis=0)]
        std_rounded = [round(m, 2) for m in np.std(success, axis=0)]
        return mean_rounded, std_rounded, data_collection

    def evaluate_in_environment(
        self, nr_iters=1, max_steps=250, render=False, burn_in_steps=50
    ):
        """
        Measure success --> how long can we balance the pole on top
        """
        data_collection = []
        with torch.no_grad():
            success = np.zeros(nr_iters)
            # observe also the oscillation
            avg_angle = np.zeros(nr_iters)
            for n in range(nr_iters):
                # only set the theta to the top, and reduce speed
                self.eval_env._reset_upright()
                new_state = self.eval_env.state

                angles = list()
                # Start balancing
                for i in range(max_steps):
                    # Transform state in the same way as the training data
                    # and normalize
                    # Predict optimal action:
                    action_seq = self.controller.predict_actions(new_state, 0)

                    for action_ind in range(APPLY_UNTIL):
                        # run action in environment
                        new_state = self.eval_env._step(
                            action_seq[:, action_ind], is_torch=self.mpc == 0
                        )
                        data_collection.append(new_state)
                        if i > burn_in_steps:
                            angles.append(np.absolute(new_state[2]))
                        if render:
                            self.eval_env._render()
                            # test = self.eval_env._render(mode="rgb_array")
                            # time.sleep(.1)
                    if not self.eval_env.is_upright():
                        break
                        # track number of timesteps until failure

                avg_angle[n] = np.mean(angles) if len(angles) > 0 else 100
                success[n] = i
                self.eval_env._reset()
        # print(success)
        mean_err = np.mean(success)
        std_err = np.std(success)
        print("Average success: %3.2f (%3.2f)" % (mean_err, std_err))
        return mean_err, std_err, data_collection


def run_saved_arr(path):
    """
    Load a saved sequence of states and visualize it
    """
    states = np.load(path)
    self.eval_env = CartPoleEnv()
    for state in states:
        self.eval_env.state = state
        self.eval_env._render()
        time.sleep(.1)


def load_model(model_name, epoch):
    with open(
        os.path.join("trained_models", "cartpole", model_name, "config.json"),
        "r"
    ) as infile:
        config = json.load(infile)

    path_load = os.path.join(
        "trained_models", "cartpole", model_name, "model_pendulum" + epoch
    )
    if not os.path.exists(path_load):
        path_load = os.path.join(
            "trained_models", "cartpole", model_name, "model_cartpole" + epoch
        )
    net = torch.load(path_load)
    net.eval()
    controller_model = CartpoleWrapper(net, **config)
    return controller_model


if __name__ == "__main__":
    # make as args:
    parser = argparse.ArgumentParser("Model directory as argument")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="current_model",
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

    if args.model == "mpc":
        load_dynamics = None
        controller_model = MPC(
            horizon=20,
            dt=0.02,
            dynamics="cartpole",
            load_dynamics=load_dynamics
        )
    else:
        controller_model = load_model(args.model, args.epoch)

    modified_params = {}

    # define dynamics and environmen
    dynamics = CartpoleDynamics(modified_params=modified_params)
    eval_env = CartPoleEnv(dynamics)
    evaluator = Evaluator(controller_model, eval_env)
    # angles = evaluator.run_for_fixed_length(net, render=True)
    success, suc_std, _ = evaluator.evaluate_in_environment(
        render=True, max_steps=500
    )
    # try:
    #     swingup_mean, swingup_std, _, data_collection = evaluator.make_swingup(
    #         net, nr_iters=1, render=True, max_iters=500
    #     )
    #     print(swingup_mean, swingup_std)
    # except KeyboardInterrupt:
    #     pass
    # # Save sequence?
    # if len(input("save? enter anything for yes")) > 0:
    #     data_collection = np.asarray(data_collection)
    #     np.save("saved_states.npy", data_collection)
