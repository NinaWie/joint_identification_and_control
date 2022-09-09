import time
import numpy as np
import torch

from mbrl.env.pets_halfcheetah import HalfCheetahEnv
from neural_control.mujoco_utils import DynamicsModelPETS, ControllerModel


def loss_fn(next_obs, act, conversion_factor=1):
    # TODO: rewrite with torch
    loss_ctrl = 0.1 * np.sum(act**2, axis=1)
    loss_move = conversion_factor - np.sum(next_obs[:, 0], axis=0)
    return loss_ctrl + loss_move


def evaluate_cheetah(
    env, controller, nr_steps=100, return_obs=False, render=False
):
    collect_obs = []
    collect_rewards = []

    num_neg_rewards = 0

    # act_shape = env.action_space.shape
    # obs_shape = env.observation_space.shape

    # create real environment
    obs = env.reset()

    for step in range(nr_steps):
        # print()

        obs_reshaped = torch.from_numpy(np.expand_dims(obs, 0)).float()

        act = controller(obs_reshaped)[0, 0].detach().numpy()

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

        obs, rew, _, _ = env.step(act)
        # render
        if render:
            # print(obs[2])
            # print("act", act)
            # if obs[2] > 3:
            #     print(step)
            #     print("flipped")
            env.render()
            time.sleep(0.05)

        # flipped
        if abs(obs[2]) > 1.5:
            collect_obs.append(obs)
            # collect_rewards.append(-10)
            # print(collect_rewards)
            break
        # logging
        collect_obs.append(obs)
        collect_rewards.append(rew)
        # print(round(float(rew), 2))
        if rew < 0:
            num_neg_rewards += 1
        else:
            num_neg_rewards = 0
        if num_neg_rewards > 10:
            collect_rewards = collect_rewards[:-10]
            break

        # loss = loss_fn(np.expand_dims(obs, axis=0), np.array([act]))
        # test_list.append([rew, loss])
    if return_obs:
        return collect_obs
    else:
        return collect_rewards


def run_eval(env, controller, nr_steps, nr_iters):
    rew, not_falling = [], []
    for _ in range(nr_iters):
        rewards = evaluate_cheetah(env, controller, nr_steps)
        not_falling.append(len(rewards))
        rew.append(np.sum(rewards))  # TODO: any reason to take the sum?
    print(not_falling)
    print(rew)
    print("Average rewards before fall:", np.mean(rew), np.median(rew))
    print(
        "Average nr of steps before fall:", np.mean(not_falling),
        np.median(not_falling)
    )
    return round(np.mean(rew), 3)


class RandomController:

    def __call__(self, obs):
        return torch.from_numpy(
            np.expand_dims(np.random.rand(6) * 2 - 1, axis=(0, 1))
        )


if __name__ == "__main__":
    env = HalfCheetahEnv()
    # dynamics = DynamicsModelPETS()

    controller = torch.load(
        "trained_models/mujoco/cheetah_model_petsdyn_final"
    )
    controller.eval()
    # controller = RandomController()

    # # Evaluate with plotting
    # evaluate_cheetah(env, controller, nr_steps=10000, render=True)

    # Evaluate systematically
    avg_rewards = run_eval(env, controller, 1000, 50)
    # print("with run eval", avg_rewards)