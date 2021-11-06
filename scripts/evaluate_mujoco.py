import numpy as np
import torch

from mbrl.env.pets_halfcheetah import HalfCheetahEnv
from neural_control.mujoco_utils import DynamicsModelPETS, ControllerModel


def loss_fn(next_obs, act, conversion_factor=1):
    # TODO: rewrite with torch
    loss_ctrl = 0.1 * np.sum(act**2, axis=1)
    loss_move = conversion_factor - np.sum(next_obs[:, 0], axis=0)
    return loss_ctrl + loss_move


def evaluate_cheetah(env, controller, nr_steps=100):
    collect_obs = []

    act_shape = env.action_space.shape
    obs_shape = env.observation_space.shape

    # create real environment
    obs = env.reset()

    test_list = []
    for _ in range(nr_steps):
        # print()

        obs_reshaped = torch.from_numpy(np.expand_dims(obs, 0)).float()

        act = controller(obs_reshaped)[0].detach().numpy()

        # ----- DYNAMICS TESTING --------
        # act = np.expand_dims(np.random.rand(act_shape[0]), 0)
        # act_reshaped = torch.from_numpy(act).float()
        # obs_len = obs_shape[0]
        # obs_mean, obs_std = (
        #     dynamics.normalizer.mean[0, :obs_len],
        #     dynamics.normalizer.std[0, :obs_len]
        # )
        # act_mean, act_std = (
        #     dynamics.normalizer.mean[0, obs_len:],
        #     dynamics.normalizer.std[0, obs_len:]
        # )
        # normed_obs = (obs_reshaped - obs_mean) / obs_std
        # normed_act = (act_reshaped - act_mean) / act_std
        # pred_obs_final = dynamics.forward(obs_reshaped, normed_obs, normed_act)
        # print(pred_obs_final[0, :5])
        # ----- DYNAMICS TESTING --------

        obs, rew, done, _ = env.step(act)
        # print(obs[:5])
        collect_obs.append(obs)

        loss = loss_fn(np.expand_dims(obs, axis=0), np.array([act]))

        test_list.append([rew, loss])
    return collect_obs


if __name__ == "__main__":
    env = HalfCheetahEnv()
    dynamics = DynamicsModelPETS()
    controller = ControllerModel(
        env.observation_space.shape[0], env.action_space.shape[0]
    )
    evaluate_cheetah(env, controller, nr_steps=10)