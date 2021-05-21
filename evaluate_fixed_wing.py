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
from neural_control.dynamics.fixed_wing_dynamics import FixedWingDynamics
from neural_control.trajectory.q_funcs import project_to_line
from evaluate_base import run_mpc_analysis, load_model_params, average_action


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
        is_seq=False,
        **kwargs
    ):
        # sequence dynamics?
        self.is_seq = is_seq
        self.controller = controller
        self.dt = dt
        self.horizon = horizon
        self.render = render
        self.thresh_div = thresh_div
        self.thresh_stable = thresh_stable
        self.eval_env = env
        self.des_speed = 11.5
        self.test_time = test_time
        self.waypoint_metric = waypoint_metric

        self.state_action_history = np.zeros((buffer_len, 12 + 4))

    def fly_to_point(self, target_points, max_steps=1000, do_avg_act=0):
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
            step += 1

            # if self.render:
            #     np.set_printoptions(suppress=1, precision=3)
            #     print(action[0])
            #     print()
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
        if not self.waypoint_metric:
            return np.array(drone_traj), np.array(div_to_linear)
        return np.array(drone_traj), np.array(div_target)

    def run_eval(self, nr_test, return_dists=False):
        mean_div, not_div_time = [], []
        for i in range(nr_test):
            # important! reset after every run
            if isinstance(self.controller, MPC):
                self.controller._initDynamics()
            target_point = [
                np.random.rand(3) * np.array([70, 10, 10]) +
                np.array([20, -5, -5])
            ]
            # traj_test = run_wing_flight(
            #     self.eval_env, traj_len=300, dt=self.dt, render=0
            # )
            # target_point = [traj_test[-1, :3]]
            drone_traj, divergences = self.fly_to_point(target_point)
            # last_x_points = drone_traj[-20:, :3]
            # last_x_dists = [
            #     np.linalg.norm(target_point - p) for p in last_x_points
            # ]
            # min_dists.append(np.min(last_x_dists))
            # not_diverged = np.sum(divergences < self.thresh_div)
            mean_div.append(np.mean(divergences))
            # not_div_time.append(not_diverged)
        mean_err = np.mean(mean_div)
        std_err = np.std(mean_div)
        # print(
        #     "Time not diverged: %3.2f (%3.2f)" %
        #     (np.mean(not_div_time), np.std(not_div_time))
        # )
        print("Average error: %3.2f (%3.2f)" % (mean_err, std_err))
        if return_dists:
            return np.array(mean_div)
        return mean_err, std_err


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

    is_seq = "seq" in model_name
    dynamics = FixedWingDynamics(modified_params=modified_params)
    eval_env = SimpleWingEnv(dynamics, params["dt"])
    evaluator = FixedWingEvaluator(
        controller, eval_env, is_seq=is_seq, test_time=1, **params
    )

    # only run evaluation without render
    if args.eval > 0:
        # tic = time.time()
        out_path = "../presentations/analysis"
        evaluator.render = 0
        dists_from_target = evaluator.run_eval(
            nr_test=args.eval, return_dists=True
        )
        # np.save(
        #     os.path.join(
        #         out_path,
        #         f"{model_name}_{'_'.join(modified_params.keys())}.npy"
        #     ), dists_from_target
        # )
        # print("time for 100 trajectories", time.time() - tic)
        # run_mpc_analysis(evaluator)
        exit()

    target_point = [[50, -3, -3]]  # , [100, 3, 3]]

    # RUN
    drone_traj, _ = evaluator.fly_to_point(target_point, max_steps=600)

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
