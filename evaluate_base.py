import os
import json
import numpy as np
import torch

from neural_control.plotting import plot_success
from neural_control.dynamics.fixed_wing_dynamics import (
    FixedWingDynamics, SequenceFixedWingDynamics
)
from neural_control.dynamics.cartpole_dynamics import CartpoleDynamics
from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics
from neural_control.dataset import WingSequenceDataset, QuadSequenceDataset


def load_model_params(model_path, name="model_quad", epoch=""):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        print("Load old config..")
        config_path = os.path.join(model_path, "param_dict.json")
    with open(config_path, "r") as outfile:
        param_dict = json.load(outfile)

    net = torch.load(os.path.join(model_path, name + epoch))
    net.eval()
    return net, param_dict


global last_actions
last_actions = np.zeros((10, 4))


def average_action(action, step, do_avg_act=0):
    # weight = 9 - np.arange(10)
    weight = np.ones(10)
    # weight_first = np.array(np.ones(9).tolist() + [0])
    # weighting = do_avg_act * weigh_equal + (1 - do_avg_act) * weight_first
    weight = np.expand_dims(weight, 1)
    # print(weight)

    global last_actions
    if do_avg_act:
        # make average action
        if step == 0:
            last_actions = action.copy()
        else:
            last_actions = np.roll(last_actions, -1, axis=0)
            # rolling mean

            last_actions = (last_actions * weight + action) / (weight + 1)
        # print("actions", action)
        # print("last actions", last_actions)\
        use_action = last_actions[0]
    else:
        use_action = action[0]
    return use_action


def increase_param(default_val, inc):
    # first case: param is array
    if isinstance(default_val, list):
        new_val = (np.array(default_val) * inc).astype(float)
        # if all zero, add instead
        if not np.any(new_val):
            new_val += (inc - 1)
    else:
        new_val = float(default_val * inc)
        if new_val == 0:
            new_val += (inc - 1)
    return new_val


def run_mpc_analysis(
    evaluator, system="fixed_wing", out_path="../presentations/analysis"
):
    """
    Run eval function with mpc multiple times and plot the results
    Args:
        evaluator (Evaluator): fully initialized environment with controller
    """
    with open(f"neural_control/dynamics/config_{system}.json", "r") as inf:
        parameters = json.load(inf)

    increase_factors = np.arange(1, 2, .1)
    for key, default_val in parameters.items():
        # for key in ["mass"]:
        #     default_val = parameters[key]

        if key == "g" or key == "gravity":
            # gravity won't change ;)
            continue
        default_val = parameters[key]

        print(
            f"\n-------------{key} (with default {default_val}) ------------"
        )
        mean_list, std_list = [], []
        for inc in increase_factors:
            new_val = increase_param(default_val, inc)

            modified_params = {key: new_val}
            print("\n ", round(inc, 2), "modified:", modified_params)

            if system == "fixed_wing":
                evaluator.eval_env.dynamics = FixedWingDynamics(
                    modified_params=modified_params
                )
            elif system == "quad":
                evaluator.eval_env.dynamics = FlightmareDynamics(
                    modified_params=modified_params
                )

            mean_dist, std_dist = evaluator.run_eval(nr_test=20)
            mean_list.append(mean_dist)
            std_list.append(std_dist)
        x = np.array(increase_factors)
        plot_success(
            x, mean_list, std_list, os.path.join(out_path, key + "_mpc.jpg")
        )


def dyn_comparison_cartpole(
    dyn_trained, np_state, np_action, history, timestamp, dt=0.05
):
    action_torch = torch.tensor([np_action.tolist()]).float()
    state_torch = torch.tensor([np_state.tolist()]).float()
    # history_torch = torch.from_numpy(np.expand_dims(history, 0)).float()
    # print(state_torch.size(), action_torch.size(), inp_history.size())

    # DYN EVALUATION
    dyn_bl = CartpoleDynamics()
    dyn_mod = CartpoleDynamics({"contact": 1})
    dyn_mod.timestamp = timestamp
    # pass through dynamics
    state_bl = dyn_bl(state_torch, action_torch, dt=dt)
    state_mod = dyn_mod(state_torch, action_torch, dt=dt)
    with torch.no_grad():
        state_trained = dyn_trained(state_torch, history, action_torch, dt=dt)
    # np.set_printoptions(suppress=1, precision=3)
    # print("bl")
    # print(state_bl.numpy())
    # print("mod")
    # print(state_mod.numpy())
    # print("trained")
    # print(state_trained.numpy())
    # print()
    actual_delta = torch.sqrt(torch.sum(((state_mod - state_bl) / dt)**2))
    trained_delta = torch.sqrt(
        torch.sum(((state_mod - state_trained) / dt)**2)
    )
    return [actual_delta.item(), trained_delta.item()]


def dyn_comparison_wing(
    dyn_trained, np_state, np_action, history, timestamp, dt=0.05
):
    action_torch = torch.tensor([np_action.tolist()]).float()
    state_torch = torch.tensor([np_state.tolist()]).float()
    history_torch = torch.from_numpy(np.expand_dims(history, 0)).float()

    helper_dataset = WingSequenceDataset(1)
    inp_history = helper_dataset.prepare_history(history_torch)
    # print(state_torch.size(), action_torch.size(), inp_history.size())

    # DYN EVALUATION
    dyn_bl = FixedWingDynamics()
    dyn_mod = FixedWingDynamics({"wind": 2})
    dyn_mod.timestamp = timestamp
    # pass through dynamics
    state_bl = dyn_bl(state_torch, action_torch, dt=dt)
    state_mod = dyn_mod(state_torch, action_torch, dt=dt)
    with torch.no_grad():
        state_trained = dyn_trained(
            state_torch, inp_history, action_torch, dt=dt
        )
    # np.set_printoptions(suppress=1, precision=3)
    # print("bl")
    # print(state_bl.numpy())
    # print("mod")
    # print(state_mod.numpy())
    # print("trained")
    # print(state_trained.numpy())
    # print()
    actual_delta = torch.sqrt(torch.sum(((state_mod - state_bl) / dt)**2))
    trained_delta = torch.sqrt(
        torch.sum(((state_mod - state_trained) / dt)**2)
    )
    return [actual_delta.item(), trained_delta.item()]


def dyn_comparison_quad(
    dyn_trained, np_state, np_action, history, timestamp, dt=0.05
):
    action_torch = torch.tensor([np_action.tolist()]).float()
    state_torch = torch.tensor([np_state.tolist()]).float()
    history_torch = torch.from_numpy(np.expand_dims(history, 0)).float()

    helper_dataset = QuadSequenceDataset(1)
    inp_history = helper_dataset.prepare_history(history_torch)
    # print(state_torch.size(), action_torch.size(), inp_history.size())

    # DYN EVALUATION
    dyn_bl = FlightmareDynamics()
    dyn_mod = FlightmareDynamics(
        {"wind": 2}
        # {'translational_drag': np.array([0.3, 0.3, 0.3])}
    )
    dyn_mod.timestamp = timestamp
    # pass through dynamics
    state_bl = dyn_bl(state_torch, action_torch, dt=dt)
    state_mod = dyn_mod(state_torch, action_torch, dt=dt)
    with torch.no_grad():
        state_trained = dyn_trained(
            state_torch, inp_history, action_torch, dt=dt
        )
    # np.set_printoptions(suppress=1, precision=3)
    # print("bl")
    # print(state_bl.numpy())
    # print("mod")
    # print(state_mod.numpy())
    # print("trained")
    # print(state_trained.numpy())
    # print()
    actual_delta = torch.sqrt(torch.sum(((state_mod - state_bl) / dt)**2))
    trained_delta = torch.sqrt(
        torch.sum(((state_mod - state_trained) / dt)**2)
    )
    return [actual_delta.item(), trained_delta.item()]
