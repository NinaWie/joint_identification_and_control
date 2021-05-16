import json
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from neural_control.environments.cartpole_env import CartPoleEnv
from neural_control.environments.rl_envs import (
    CartPoleEnvRL, WingEnvRL, QuadEnvRL
)
from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics
from neural_control.dynamics.cartpole_dynamics import CartpoleDynamics
from neural_control.dynamics.fixed_wing_dynamics import FixedWingDynamics
from neural_control.models.hutter_model import Net
from neural_control.plotting import plot_wing_pos_3d
from neural_control.trajectory.q_funcs import project_to_line

# PARAMS
fixed_wing_dt = 0.05
cartpole_dt = 0.05
quad_dt = 0.1
quad_speed = 0.2
quad_horizon = 10

# ------------------ CartPole -----------------------


def train_cartpole(save_name):
    dyn = CartpoleDynamics()
    env = CartPoleEnvRL(dyn, dt=cartpole_dt)
    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=60000)
    model.save(save_name)


def finetune_cartpole(save_name, modified_params):
    dyn = CartpoleDynamics(modified_params=modified_params)
    env = CartPoleEnvRL(dyn, dt=cartpole_dt)
    model = PPO.load(save_name, env=env)

    for i in range(50):
        model.learn(total_timesteps=1000)
        evaluate_cartpole(model, env, nr_iters=10)
    model.save(save_name + "_finetuned")


def evaluate_cartpole(model, env, max_steps=250, nr_iters=1, render=0):
    states, actions = [], []
    for j in range(nr_iters):
        obs = env.reset()
        for i in range(max_steps):
            action, _states = model.predict(obs)
            actions.append(action)
            obs, rewards, done, info = env.step(action)
            states.append(obs)
            if render:
                env.render()
            if done:
                break

    states = np.array(states)
    actions = np.array(actions)
    print("Average velocity:", np.mean(np.absolute(states[:, 1])))
    # plt.hist(actions)
    # plt.show()


def test_cartpole(save_name, modified_params={}, max_steps=500):
    dyn = CartpoleDynamics(modified_params=modified_params)
    env = CartPoleEnvRL(dyn, dt=cartpole_dt)
    model = PPO.load(save_name)
    evaluate_cartpole(model, env, max_steps, render=1)


# ------------------ Fixed wing drone -----------------------


def evaluate_wing(model=None, env=None, max_steps=1000, nr_iters=1, render=0):
    # TODO: merge evaluate functions
    if env is None:
        dyn = FixedWingDynamics()
        env = WingEnvRL(dyn, dt=0.05)

    div_target = []
    np.set_printoptions(precision=3, suppress=1)
    for j in range(nr_iters):
        obs = env.reset(x_dist=50, x_std=5)
        if render:
            print(f"iter {j}:", env.target_point)
        trajectory = []
        for i in range(max_steps):
            if model is not None:
                # OURS
                if isinstance(model, Net):
                    obs_state, obs_ref = env.prepare_obs()
                    with torch.no_grad():
                        suggested_action = model(obs_state, obs_ref)
                        suggested_action = torch.sigmoid(suggested_action)[0]
                        suggested_action = torch.reshape(
                            suggested_action, (10, 4)
                        )
                        action = suggested_action[0].numpy()
                else:
                    # RL
                    action, _states = model.predict(obs)
                # print(action)
            else:
                action_prior = np.array([.25, .5, .5, .5])
                sampled_action = np.random.normal(scale=.15, size=4)
                action = np.clip(sampled_action + action_prior, 0, 1)

            obs, rewards, done, info = env.step(action)
            # print(env.state[:3], env.get_divergence())
            # print()
            trajectory.append(env.state)

            if render:
                env.render()
            if done:
                if env.state[0] < 20:
                    div_target.append(
                        np.linalg.norm(env.state[:3] - env.target_point)
                    )
                else:
                    target_on_traj = project_to_line(
                        trajectory[-2][:3], env.state[:3], env.target_point
                    )
                    div_target.append(
                        np.linalg.norm(target_on_traj - env.target_point)
                    )
                if render:
                    print("last state", env.state[:3], "div", div_target[-1])
                break

    print(
        "Average error: %3.2f (%3.2f)" %
        (np.mean(div_target), np.std(div_target))
    )
    return np.array(trajectory), np.mean(div_target), np.std(div_target)


def train_wing(
    save_name, load_model=None, modified_params={}, steps_per_bunch=10000
):
    dyn = FixedWingDynamics(modified_params=modified_params)
    env = WingEnvRL(dyn, fixed_wing_dt)
    if load_model is None:
        model = PPO('MlpPolicy', env, verbose=1)
    else:
        model = PPO.load(load_model, env=env)

    try:
        res_dict = defaultdict(list)
        for k in range(50):
            print(f"------------- Samples: {k*steps_per_bunch}---------------")
            _, meandiv, stddiv = evaluate_wing(model, env, nr_iters=30)
            res_dict["mean_div"].append(meandiv)
            res_dict["std_div"].append(stddiv)
            res_dict["samples"].append(k * steps_per_bunch)
            model.learn(total_timesteps=steps_per_bunch)
    except KeyboardInterrupt:
        pass
    with open(save_name + "_res.json", "w") as outfile:
        json.dump(res_dict, outfile)
    model.save(save_name)


def test_wing(save_name, modified_params={}, max_steps=1000):
    dyn = FixedWingDynamics(modified_params=modified_params)
    env = WingEnvRL(dyn, fixed_wing_dt)
    model = PPO.load(save_name)
    trajectory, _, _ = evaluate_wing(
        model, env, max_steps, nr_iters=40, render=0
    )
    # plot
    plot_wing_pos_3d(
        trajectory, [env.target_point], save_path=save_name + "_plot.png"
    )


def test_ours_wing(model_path, modified_params={}, max_steps=1000):
    model = torch.load(os.path.join(model_path, "model_wing"))
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as outfile:
        param_dict = json.load(outfile)
    dyn = FixedWingDynamics(modified_params=modified_params)
    env = WingEnvRL(dyn, **param_dict)
    evaluate_wing(model, env, max_steps, nr_iters=40, render=0)


# ------------------ Quadrotor -----------------------


def evaluate_quad(model, env, max_steps=400, nr_iters=1, render=0):
    divergences = []
    num_steps = []
    np.set_printoptions(precision=3, suppress=1)
    for j in range(nr_iters):
        obs = env.reset()
        drone_trajectory = []
        for i in range(max_steps):
            if isinstance(model, Net):  # DP
                obs_state, obs_ref = env.prepare_obs()
                with torch.no_grad():
                    suggested_action = model(obs_state, obs_ref)
                    suggested_action = torch.sigmoid(suggested_action)[0]
                    suggested_action = torch.reshape(suggested_action, (10, 4))
                    action = suggested_action[0].numpy()
            else:  # RL
                action, _states = model.predict(obs)

            obs, rewards, done, info = env.step(action)
            if render:
                env.render()
            divergences.append(env.get_divergence())
            if done:
                num_steps.append(len(drone_trajectory))
                break
            drone_trajectory.append(env.state)
    print(
        "Tracking error: %3.2f (%3.2f)" %
        (np.mean(divergences), np.std(divergences))
    )
    print(
        "Number steps: %3.2f (%3.2f)" %
        (np.mean(num_steps), np.std(num_steps))
    )
    return drone_trajectory, np.mean(divergences), np.std(divergences)


def test_rl_quad(save_name, modified_params={}, max_steps=1000):
    dyn = FlightmareDynamics(modified_params=modified_params)
    env = QuadEnvRL(dyn, quad_dt)
    model = PPO.load(save_name)
    _ = evaluate_quad(model, env, max_steps, nr_iters=40, render=0)


def test_ours_quad(model_path, modified_params={}, max_steps=500):
    model = torch.load(os.path.join(model_path, "model_quad"))
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as outfile:
        param_dict = json.load(outfile)
    dyn = FlightmareDynamics(modified_params=modified_params)
    env = QuadEnvRL(dyn, **param_dict)
    evaluate_quad(model, env, max_steps, nr_iters=1, render=1)


def train_quad(
    save_name, load_model=None, modified_params={}, steps_per_bunch=10000
):
    dyn = FlightmareDynamics(modified_params=modified_params)
    env = QuadEnvRL(
        dyn, quad_dt, speed_factor=quad_speed, nr_actions=quad_horizon
    )
    if load_model is None:
        model = PPO('MlpPolicy', env, verbose=1)
    else:
        model = PPO.load(load_model, env=env)

    try:
        res_dict = defaultdict(list)
        for k in range(50):
            print(f"------------- Samples: {k*steps_per_bunch}---------------")
            # print(model.total_timesteps)
            _, meandiv, stddiv = evaluate_quad(model, env, nr_iters=30)
            res_dict["mean_div"].append(meandiv)
            res_dict["std_div"].append(stddiv)
            res_dict["samples"].append(k * steps_per_bunch)
            model.learn(total_timesteps=steps_per_bunch)
    except KeyboardInterrupt:
        pass
    with open(save_name + "_res.json", "w") as outfile:
        json.dump(res_dict, outfile)
    model.save(save_name)


if __name__ == "__main__":
    # ------------------ CartPole -----------------------
    # save_name = "trained_models/cartpole/reinforcement_learning/ppo2_smallact"
    # train_cartpole(save_name)
    # finetune_cartpole(save_name, modified_params={"wind": .5})
    # test_cartpole(save_name, modified_params={"wind": .5})

    # ------------------ Fixed wing drone -----------------------
    load_name = "trained_models/wing/reinforcement_learning/ppo_50"
    save_name = "trained_models/wing/reinforcement_learning/ppo_finetuned"
    scenario = {"vel_drag_factor": .3}
    # train_wing(
    #     save_name,
    #     load_model=load_name,
    #     modified_params=scenario,
    #     steps_per_bunch=5000
    # )
    # finetune_wing(save_name, modified_params={})
    # test_ours_wing(
    #     "trained_models/wing/current_model",
    #     modified_params=scenario
    # )
    # test_wing(save_name, modified_params=scenario)
    # evaluate_wing(render=1)

    # ------------------ Quadrotor -----------------------
    # save_name = "trained_models/quad/optimizer_04_model/"
    # save_name = "trained_models/quad/reinforcement_learning/ppo_test"

    scenario = {}  # {"translational_drag": np.array([.3, .3, .3])}
    # test_ours_quad(save_name, modified_params=scenario)
    train_quad(save_name)
