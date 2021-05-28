import os
import time
import argparse
import json
import numpy as np
import torch

from neural_control.environments.wing_env import SimpleWingEnv, run_wing_flight
from neural_control.plotting import plot_wing_pos_3d, plot_success
from neural_control.dataset import WingDataset, WingSequenceDataset
from neural_control.controllers.network_wrapper import FixedWingNetWrapper
from neural_control.controllers.mpc import MPC
from neural_control.dynamics.fixed_wing_dynamics import (
    FixedWingDynamics, SequenceFixedWingDynamics
)
from neural_control.trajectory.q_funcs import project_to_line
from evaluate_base import (
    run_mpc_analysis, load_model_params, average_action, dyn_comparison_wing
)


class FixedWingEvaluator:
    """
    Evaluate performance of the fixed wing drone
    """

    def __init__(
        self,
        controller,
        env,
        dt=0.01,
        horizon=1,
        render=0,
        thresh_div=10,
        thresh_stable=0.8,
        test_time=0,
        buffer_len=3,
        waypoint_metric=1,
        eval_dyn=None,
        is_seq=False,
        **kwargs
    ):
        # sequence dynamics?
        self.is_seq = is_seq
        self.controller = controller
        self.dt = dt
        self.horizon = horizon
        self.eval_dyn = eval_dyn
        self.render = render
        self.thresh_div = thresh_div
        self.thresh_stable = thresh_stable
        self.eval_env = env
        self.des_speed = 11.5
        self.test_time = test_time
        self.waypoint_metric = waypoint_metric
        self.use_random_actions = 0
        self.use_mpc = None

        self.dyn_eval_test = []

        self.state_action_history = np.zeros((buffer_len, 12 + 4))

    def fly_to_point(
        self, target_points, max_steps=1000, do_avg_act=0, return_traj=False
    ):
        self.eval_env.zero_reset()
        if self.render:
            self.eval_env.drone_render_object.set_target(target_points)

        # first target
        current_target_ind = 0
        # target trajectory
        line_start = self.eval_env._state[:3]

        state = self.eval_env._state
        stable = True

        self.state_action_history[:, :12] = state
        self.state_action_history[:, 12:] = np.array([.5 for _ in range(4)])

        drone_traj, div_to_linear, div_target = [], [], []
        step = 0
        while len(drone_traj) < max_steps:
            current_target = target_points[current_target_ind]
            if self.is_seq:
                action = self.controller.predict_actions(
                    self.state_action_history,
                    np.array(current_target),
                    timestamp=self.eval_env.dynamics.timestamp
                )
            else:
                action = self.controller.predict_actions(state, current_target)
            use_action = average_action(action, step, do_avg_act=do_avg_act)

            # REPLACE ACTION BY RANDOM
            if self.use_random_actions:
                action_prior = np.array([.25, .5, .5, .5])
                sampled_action = np.random.normal(scale=.15, size=(4))
                use_action = np.clip(sampled_action + action_prior, 0, 1)

            # REPLACE BY MPC
            if self.use_mpc:
                use_action = self.use_mpc.predict_actions(
                    state, current_target
                )[0]

            step += 1

            # if self.render:
            #     np.set_printoptions(suppress=1, precision=3)
            #     print(action[0])
            #     print()
            if step % 10 == 0 and self.eval_dyn is not None:
                self.dyn_eval_test.append(
                    dyn_comparison_wing(
                        self.eval_dyn,
                        state,
                        use_action,
                        self.state_action_history,
                        self.eval_env.dynamics.timestamp,
                        dt=self.dt
                    )
                )
            state, stable = self.eval_env.step(
                use_action, thresh_stable=self.thresh_stable
            )
            if self.render:
                self.eval_env.render()
                time.sleep(.05)

            # update history
            self.state_action_history = np.roll(
                self.state_action_history, 1, axis=0
            )
            self.state_action_history[0, :12] = state
            self.state_action_history[0, 12:] = use_action

            # project drone onto line and compute divergence
            drone_on_line = project_to_line(
                line_start, current_target, state[:3]
            )
            div = np.linalg.norm(drone_on_line - state[:3])
            div_to_linear.append(div)
            drone_traj.append(np.concatenate((state, action[0])))

            # set next target if we have passed one
            if state[0] > current_target[0]:
                # project target onto line
                target_on_traj = project_to_line(
                    drone_traj[-2][:3], state[:3], current_target
                )
                div_target.append(
                    np.linalg.norm(target_on_traj - current_target)
                )
                if self.render:
                    np.set_printoptions(suppress=1, precision=3)
                    print(
                        "target:", current_target, "pos:", state[:3],
                        "div to target", div_target[-1]
                    )
                if current_target_ind < len(target_points) - 1:
                    current_target_ind += 1
                    line_start = state[:3]
                else:
                    break

            if not stable or div > self.thresh_div:
                div_target.append(self.thresh_div)
                if self.test_time:
                    print("diverged", div, "stable", stable)
                    break
                else:
                    reset_state = np.zeros(12)
                    reset_state[:3] = drone_on_line
                    vec = current_target - drone_on_line
                    reset_state[3:6
                                ] = vec / np.linalg.norm(vec) * self.des_speed
                    self.eval_env._state = reset_state
        if len(drone_traj) == max_steps:
            print("Reached max steps")
            div_target.append(self.thresh_div)
        if return_traj:
            return np.array(drone_traj)
        else:
            return np.array(div_target), np.array(div_to_linear)

    def run_eval(self, nr_test, return_dists=False, x_dist=50, x_std=5):
        self.dyn_eval_test = []
        mean_div_target, mean_div_linear = [], []
        for i in range(nr_test):
            self.eval_env.dynamics.reset_wind()
            # important! reset after every run
            if isinstance(self.controller, MPC):
                self.controller._initDynamics()
            # set target point
            rand_y, rand_z = tuple((np.random.rand(2) - .5) * 2 * x_std)
            target_point = np.array([[x_dist, rand_y, rand_z]])
            # for overfitting
            # target_point = np.array([[x_dist, -3, 3]])
            # self.eval_env.dynamics.timestamp = np.pi / 2
            div_target, div_linear = self.fly_to_point(target_point)

            mean_div_target.append(np.mean(div_target))
            mean_div_linear.append(np.mean(div_linear))
            # np.set_printoptions(suppress=True, precision=3)
            # print(
            #     self.eval_env.dynamics.wind_direction, target_point,
            #     div_target, np.mean(div_linear)
            # )
            # not_div_time.append(not_diverged)

        res_eval = {
            "mean_div_target": np.mean(mean_div_target),
            "mean_div_linear": np.mean(mean_div_linear),
            "std_div_target": np.std(mean_div_target),
            "std_div_linear": np.std(mean_div_linear),
            "median_div_target": np.median(mean_div_target)
        }
        # print(
        #     "Time not diverged: %3.2f (%3.2f)" %
        #     (np.mean(not_div_time), np.std(not_div_time))
        # )
        if self.eval_dyn is not None:
            actual_delta = np.array(self.dyn_eval_test)[:, 0]
            # [elem[0] for elem in self.dyn_eval_test])
            trained_delta = np.array(self.dyn_eval_test)[:, 1]
            # np.array([elem[0] for elem in self.dyn_eval_test])
            res_eval["mean_delta"] = np.mean(actual_delta)
            res_eval["std_delta"] = np.std(actual_delta)
            res_eval["mean_trained_delta"] = np.mean(trained_delta)
            res_eval["std_trained_delta"] = np.std(trained_delta)
            print("--- Dynamics eval")
            print(
                "Average delta: %3.2f (%3.2f)" %
                (res_eval["mean_delta"], res_eval["std_delta"])
            )
            print(
                "Average trained delta: %3.2f (%3.2f)" % (
                    res_eval["mean_trained_delta"],
                    res_eval["std_trained_delta"]
                )
            )

        print("--- controller eval")
        print(
            "Average target error: %3.2f (%3.2f)" %
            (res_eval["mean_div_target"], res_eval["std_div_target"])
        )
        print("Median target error: %3.2f" % (res_eval["median_div_target"]))
        print(
            "Average linear error: %3.2f (%3.2f)" %
            (res_eval["mean_div_linear"], res_eval["std_div_linear"])
        )
        # print("Median linear error", round(np.median(mean_div_linear), 2))
        if return_dists and self.waypoint_metric:
            return np.array(mean_div_target)
        if return_dists and not self.waypoint_metric:
            return np.array(mean_div_linear)
        return res_eval


def load_model(model_path, epoch="", **kwargs):
    """
    Load model and corresponding parameters
    """
    net, param_dict = load_model_params(model_path, "model_wing", epoch=epoch)
    if "seq" in model_path:
        dataset = WingSequenceDataset(1, 0)
    else:
        dataset = WingDataset(0, **param_dict)

    controller = FixedWingNetWrapper(net, dataset, **param_dict)
    return controller


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
        "-u", "--unity", action='store_true', help="unity rendering"
    )
    parser.add_argument(
        "-a", "--eval", type=int, default=0, help="number eval runs"
    )
    args = parser.parse_args()

    # parameters
    params = {
        "render": 1,
        "dt": 0.05,
        "horizon": 10,
        "thresh_stable": 3,
        "thresh_div": 10
    }

    # load model
    model_name = args.model
    model_path = os.path.join("trained_models", "wing", model_name)

    if model_name != "mpc":
        controller = load_model(
            model_path, epoch=args.epoch, name="model_wing", **params
        )
    else:
        load_dynamics = None
        # 'trained_models/wing/train_vel_drag_5/dynamics_model'
        controller = MPC(
            horizon=20,
            dt=0.1,
            dynamics="fixed_wing_3D",
            load_dynamics=load_dynamics
        )

    modified_params = {"wind": 2}
    # {"residual_factor": 0.0001}
    # {"vel_drag_factor": 0.3}
    # {
    #     "CL0": 0.3,  # 0.39
    #     "CD0": 0.02,  #  0.0765,
    #     "CY0": 0.02,  # 0.0,
    #     "Cl0": -0.01,  # 0.0,
    #     "Cm0": 0.01,  # 0.02,
    #     "Cn0": 0.0,
    # }
    # modified_params = {"rho": 1.6}

    dyn_trained = None
    dyn_trained = SequenceFixedWingDynamics()
    dyn_trained.load_state_dict(
        torch.load(
            os.path.join(
                "trained_models/wing/finetune_seq_dyn_random",
                "dynamics_model_1800"
            )
        )
    )

    is_seq = "seq" in model_name
    dynamics = FixedWingDynamics(modified_params=modified_params)
    eval_env = SimpleWingEnv(dynamics, params["dt"])
    evaluator = FixedWingEvaluator(
        controller,
        eval_env,
        eval_dyn=dyn_trained,
        is_seq=is_seq,
        test_time=1,
        **params
    )
    # evaluator.use_random_actions = True

    # only run evaluation without render
    if args.eval > 0:
        # tic = time.time()
        out_path = "../presentations/final_res/wing_seq_con_comparison_target/"
        evaluator.render = 0
        # evaluator.waypoint_metric = False
        dists_from_target = evaluator.run_eval(
            nr_test=args.eval, return_dists=True
        )
        np.save(os.path.join(out_path, f"{model_name}.npy"), dists_from_target)
        # print("time for 100 trajectories", time.time() - tic)
        # run_mpc_analysis(evaluator)
        exit()

    target_point = [[50, -3, -3]]  # , [100, 3, 3]]

    # RUN
    drone_traj = evaluator.fly_to_point(
        target_point, max_steps=600, return_traj=True
    )

    np.set_printoptions(suppress=True, precision=3)
    print("\n final state", drone_traj[-1])
    print(drone_traj.shape)
    # np.save(os.path.join(model_path, "drone_traj.npy"), drone_traj)

    evaluator.eval_env.close()

    # EVAL
    plot_wing_pos_3d(
        drone_traj,
        target_point,
        save_path=os.path.join(model_path, "coords.png")
    )
