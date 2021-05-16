import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from neural_control.environments.cartpole_env import CartPoleEnv
from neural_control.environments.rl_envs import CartPoleEnvRL, WingEnvRL
from neural_control.dynamics.cartpole_dynamics import CartpoleDynamics
from neural_control.dynamics.fixed_wing_dynamics import FixedWingDynamics
from neural_control.models.hutter_model import Net
from neural_control.plotting import plot_wing_pos_3d

# PARAMS
fixed_wing_dt = 0.05
cartpole_dt = 0.05


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


def evaluate_wing(model=None, env=None, max_steps=1000, nr_iters=1, render=0):
    # TODO: merge evaluate functions
    if env is None:
        dyn = FixedWingDynamics()
        env = WingEnvRL(dyn, dt=0.05)

    np.set_printoptions(precision=3, suppress=1)
    for j in range(nr_iters):
        obs = env.reset(x_dist=50, x_std=3)
        print(env.target_point)
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
                print(env.state[:3])
                break

    return np.array(trajectory)


def train_wing(save_name):
    dyn = FixedWingDynamics()
    env = WingEnvRL(dyn, fixed_wing_dt)
    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=500000)
    model.save(save_name)


def test_wing(save_name, modified_params={}, max_steps=1000):
    dyn = FixedWingDynamics(modified_params=modified_params)
    env = WingEnvRL(dyn, fixed_wing_dt)
    model = PPO.load(save_name)
    trajectory = evaluate_wing(model, env, max_steps, render=1)
    # plot
    plot_wing_pos_3d(
        trajectory, [env.target_point], save_path=save_name + "_plot.png"
    )


def test_ours_wing(load_name, max_steps=1000):
    dyn = FixedWingDynamics()
    env = WingEnvRL(dyn, fixed_wing_dt)
    model = torch.load(load_name)
    evaluate_wing(model, env, max_steps, render=1)


if __name__ == "__main__":
    # save_name = "trained_models/cartpole/reinforcement_learning/ppo2_smallact"
    # train_cartpole(save_name)
    # finetune_cartpole(save_name, modified_params={"wind": .5})
    # test_cartpole(save_name, modified_params={"wind": .5})

    save_name = "trained_models/wing/reinforcement_learning/ppo_test"
    # train_wing(save_name)
    # finetune_wing(save_name, modified_params={})
    # test_ours_wing("trained_models/wing/current_model/model_wing")
    test_wing(save_name, modified_params={})
    # evaluate_wing(render=1)
