import os
import time
import argparse
import json
import numpy as np
import torch
import cv2

from neural_control.environments.cartpole_env import CartPoleEnv
from neural_control.dynamics.cartpole_dynamics import CartpoleDynamics
from neural_control.controllers.mpc import MPC
from neural_control.controllers.network_wrapper import (
    CartpoleWrapper, CartpoleImageWrapper
)
from neural_control.models.simple_model import Net, ImageControllerNet

# mpc like receding horizon
APPLY_UNTIL = 1

# current state, predicted action, images of prev states, next state after act
collect_states, collect_actions, collect_img, collect_next = [], [], [], []
buffer_len = 4
img_width, img_height = (200, 300)


class Evaluator:

    def __init__(self, controller, eval_env, collect_image_dataset=0):
        self.controller = controller
        self.eval_env = eval_env
        self.mpc = isinstance(self.controller, MPC)
        self.image_buffer = np.zeros((buffer_len, img_width, img_height))
        self.image_dataset = collect_image_dataset

    def _preprocess_img(self, image):
        resized = cv2.resize(
            np.mean(image, axis=2),
            dsize=(img_height, img_width),
            interpolation=cv2.INTER_LINEAR
        )
        return 255 - resized

    def _convert_image_buffer(self, state, crop_width=30):
        # image and corresponding state --> normalize x pos in image buffer!
        x_pos = state[0] / self.eval_env.state_limits[0]
        img_width_half = self.image_buffer.shape[2] // 2
        x_img = int(img_width_half + x_pos * img_width_half)
        return self.image_buffer[:, 75:175,
                                 x_img - crop_width:x_img + crop_width]

    def evaluate_in_environment(
        self, nr_iters=1, max_steps=250, render=False, burn_in_steps=50
    ):
        """
        Measure success --> how long can we balance the pole on top
        """
        if nr_iters == 0:
            return 0, 0, []
        data_collection = []
        velocities = []
        with torch.no_grad():
            success = np.zeros(nr_iters)
            # observe also the oscillation
            avg_angle = np.zeros(nr_iters)
            for n in range(nr_iters):
                # only set the theta to the top, and reduce speed
                self.eval_env._reset_upright()
                if self.image_dataset:
                    self.eval_env.state = (np.random.rand(4) - .5) * .1
                    self.eval_env.state[3] = 0

                new_state = self.eval_env.state
                if render:
                    start_img = self._preprocess_img(
                        self.eval_env._render(mode="rgb_array")
                    )
                    self.image_buffer = np.array(
                        [start_img for _ in range(buffer_len)]
                    )

                angles = list()
                # Start balancing
                for i in range(max_steps):
                    # Transform state in the same way as the training data
                    # and normalize
                    # Predict optimal action:
                    if self.controller.inp_img:
                        action_seq = self.controller.predict_actions(
                            self._convert_image_buffer(new_state)[:-1]
                        )
                    else:
                        action_seq = self.controller.predict_actions(
                            new_state, 0
                        )
                    if self.image_dataset:
                        action_seq = torch.rand(1, 4) - .5

                    prev_state = new_state.copy()
                    for action_ind in range(APPLY_UNTIL):
                        # run action in environment
                        new_state = self.eval_env._step(
                            action_seq[:, action_ind], is_torch=self.mpc == 0
                        )
                        data_collection.append(new_state)
                        velocities.append(np.absolute(new_state[1]))
                        if i > burn_in_steps:
                            angles.append(np.absolute(new_state[2]))
                        if render:
                            new_img = self.eval_env._render(mode="rgb_array")
                            # test = self.eval_env._render(mode="rgb_array")
                            # time.sleep(.1)

                    # save for image task
                    if self.image_dataset and i > buffer_len:
                        assert APPLY_UNTIL == 1
                        collect_states.append(prev_state)
                        collect_next.append(new_state)
                        collect_actions.append(action_seq[0, 0].numpy())
                    if render:
                        self.image_buffer = np.roll(
                            self.image_buffer, 1, axis=0
                        )
                        self.image_buffer[0] = self._preprocess_img(new_img)
                        # add images to dataset --> to have next img as label
                        if self.image_dataset and i > buffer_len:
                            collect_img.append(
                                self._convert_image_buffer(prev_state)
                            )

                    if not self.eval_env.is_upright():
                        break
                        # track number of timesteps until failure

                avg_angle[n] = np.mean(angles) if len(angles) > 0 else 100
                success[n] = i
                self.eval_env._reset()
        # print(success)
        mean_err = np.mean(success)
        std_err = np.std(success)
        if not self.image_dataset:
            print(
                "Average velocity: %3.2f (%3.2f)" %
                (np.mean(velocities), np.std(velocities))
            )
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
    if isinstance(net, Net):
        controller_model = CartpoleWrapper(net, **config)
    else:
        controller_model = CartpoleImageWrapper(net, **config)
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
        "-a", "--eval", type=int, default=0, help="number eval runs"
    )
    parser.add_argument(
        "-save_data",
        action="store_true",
        help="save the episode as training data"
    )
    args = parser.parse_args()

    # PARAMs
    dt = 0.05
    thresh_div = 0.21

    if args.model == "mpc":
        load_dynamics = None
        controller_model = MPC(
            horizon=20,
            dt=dt,
            dynamics="cartpole",
            load_dynamics=load_dynamics
        )
    else:
        controller_model = load_model(args.model, args.epoch)

    modified_params = {"wind": .5}
    # wind 0.01 works for wind added to x directlt, needs much higher (.5)
    # to affect the acceleration much

    # define dynamics and environmen
    dynamics = CartpoleDynamics(modified_params=modified_params)
    eval_env = CartPoleEnv(dynamics, dt, thresh_div=thresh_div)
    evaluator = Evaluator(controller_model, eval_env)
    # angles = evaluator.run_for_fixed_length(net, render=True)

    image_dataset = False

    if image_dataset:
        evaluator.image_dataset = 1
        for n in range(100):
            success, suc_std, _ = evaluator.evaluate_in_environment(
                render=True, max_steps=30
            )
        collect_actions = np.array(collect_actions)
        # cut off bottom and top
        collect_img = np.array(collect_img)
        collect_states = np.array(collect_states)
        collect_next = np.array(collect_next)
        print(
            collect_states.shape, collect_actions.shape, collect_img.shape,
            collect_next.shape
        )
        np.savez(
            "data/cartpole_img_12.npz", collect_img, collect_actions,
            collect_states, collect_next
        )
    elif args.eval > 0:
        success, suc_std, _ = evaluator.evaluate_in_environment(
            render=True, max_steps=500, nr_iters=args.eval
        )
    else:
        success, suc_std, _ = evaluator.evaluate_in_environment(
            render=True, max_steps=500, nr_iters=1
        )
