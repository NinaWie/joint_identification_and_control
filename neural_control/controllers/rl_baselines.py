import json
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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


class EvalCallback(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps
    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """

    def __init__(
        self,
        eval_func,  # function to evaluate model
        eval_env,
        eval_freq: int,
        save_path: str,
        nr_iters=10,
        # eval_key="mean_div",
        # eval_up_down=-1,
        verbose=0
    ):
        super(EvalCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_func = eval_func
        self.eval_env = eval_env
        self.save_path = save_path
        self.nr_iters = nr_iters

        # self.best_perf = 0 if eval_up_down == 1 else np.inf
        self.res_dict = defaultdict(list)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # evaluate
            _, res_step = self.eval_func(
                self.model, self.eval_env, nr_iters=self.nr_iters
            )
            for key in res_step.keys():
                self.res_dict[key].append(res_step[key])
            self.res_dict["samples"].append(self.num_timesteps)

            # save every time (TODO: change to saving best?)
            path = self.save_path + '_{}_steps'.format(self.num_timesteps)
            self.model.save(path)
            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))
        return True


def train_main(
    model_path,
    env,
    evaluate_func,
    load_model=None,
    total_timesteps=50000,
    eval_freq=10000
):
    # make directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    save_name = os.path.join(model_path, "rl")

    if load_model is None:
        model = PPO('MlpPolicy', env, verbose=1)
    else:
        model = PPO.load(load_model, env=env)

    eval_callback = EvalCallback(
        evaluate_func,
        env,
        eval_freq=eval_freq,
        save_path=save_name,
        nr_iters=10
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    except KeyboardInterrupt:
        pass
    with open(save_name + "_res.json", "w") as outfile:
        json.dump(eval_callback.res_dict, outfile)
    model.save(save_name + "_final")


# ------------------ CartPole -----------------------


def train_cartpole(model_path, load_model=None, modified_params={}):
    dyn = CartpoleDynamics(modified_params=modified_params)
    env = CartPoleEnvRL(dyn, dt=cartpole_dt)
    train_main(
        model_path,
        env,
        evaluate_cartpole,
        load_model=load_model,
        total_timesteps=50000,
        eval_freq=1000
    )


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


def test_rl_cartpole(save_name, modified_params={}, max_steps=500):
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
    return np.array(trajectory), {
        "mean_div": np.mean(div_target),
        "std_div": np.std(div_target)
    }


def train_wing(model_path, load_model=None, modified_params={}):
    dyn = FixedWingDynamics(modified_params=modified_params)
    env = WingEnvRL(dyn, fixed_wing_dt)

    train_main(
        model_path,
        env,
        evaluate_wing,
        load_model=load_model,
        total_timesteps=500000,
        eval_freq=10000
    )


def test_rl_wing(save_name, modified_params={}, max_steps=1000):
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


def evaluate_quad(model, env, max_steps=500, nr_iters=1, render=0):
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
    res_step = {
        "mean_div": np.mean(divergences),
        "std_div": np.std(divergences),
        "mean_steps": np.mean(num_steps),
        "std_steps": np.std(num_steps)
    }
    return drone_trajectory, res_step


def test_rl_quad(save_name, modified_params={}, max_steps=1000):
    dyn = FlightmareDynamics(modified_params=modified_params)
    env = QuadEnvRL(dyn, quad_dt)
    model = PPO.load(save_name)
    _ = evaluate_quad(model, env, max_steps, nr_iters=1, render=1)


def test_ours_quad(model_path, modified_params={}, max_steps=500):
    model = torch.load(os.path.join(model_path, "model_quad"))
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as outfile:
        param_dict = json.load(outfile)
    dyn = FlightmareDynamics(modified_params=modified_params)
    env = QuadEnvRL(dyn, **param_dict)
    evaluate_quad(model, env, max_steps, nr_iters=1, render=1)


def train_quad(model_path, load_model=None, modified_params={}):
    dyn = FlightmareDynamics(modified_params=modified_params)
    env = QuadEnvRL(
        dyn, quad_dt, speed_factor=quad_speed, nr_actions=quad_horizon
    )
    train_main(
        model_path,
        env,
        evaluate_quad,
        load_model=load_model,
        total_timesteps=500000,
        eval_freq=10000
    )


if __name__ == "__main__":
    # ------------------ CartPole -----------------------
    # save_name = "trained_models/cartpole/reinforcement_learning/ppo2_smallact"
    # train_cartpole(save_name)
    # finetune_cartpole(save_name, modified_params={"wind": .5})
    # test_cartpole(save_name, modified_params={"wind": .5})

    # ------------------ Fixed wing drone -----------------------
    load_name = "trained_models/wing/reinforcement_learning/final/ppo_50"
    save_name = "trained_models/wing/reinforcement_learning/ppo_finetuned_2"
    scenario = {"vel_drag_factor": .3}
    train_wing(save_name, load_model=load_name, modified_params=scenario)
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
    # save_name = "trained_models/quad/reinforcement_learning/test_with_dir"

    # scenario = {}  # {"translational_drag": np.array([.3, .3, .3])}
    # test_ours_quad(save_name, modified_params=scenario)
    # train_quad(save_name)
    # test_rl_quad(save_name)
