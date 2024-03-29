import os
import time
import argparse
import json
import numpy as np
import torch

from neural_control.environments.drone_env import QuadRotorEnvBase
from neural_control.plotting import (
    plot_state_variables, plot_trajectory, plot_position, plot_suc_by_dist,
    plot_drone_ref_coords
)
from neural_control.trajectory.straight import Hover, Straight
from neural_control.trajectory.circle import Circle
from neural_control.trajectory.polynomial import Polynomial
from neural_control.trajectory.random_traj import Random
from neural_control.dataset import QuadDataset, QuadSequenceDataset
from neural_control.controllers.network_wrapper import NetworkWrapper
from neural_control.controllers.mpc import MPC
from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics
from neural_control.dynamics.quad_dynamics_trained import (
    SequenceQuadDynamics, LearntQuadDynamics
)
from neural_control.dynamics.quad_dynamics_simple import SimpleDynamics
from evaluate_base import (
    run_mpc_analysis, load_model_params, average_action, dyn_comparison_quad
)
try:
    from neural_control.flightmare import FlightmareWrapper
except ModuleNotFoundError:
    pass

ROLL_OUT = 1

# Use cuda if available
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuadEvaluator():

    def __init__(
        self,
        controller,
        environment,
        horizon=5,
        max_drone_dist=0.1,
        render=0,
        dt=0.02,
        eval_dyn=None,
        is_seq=False,
        test_time=0,
        speed_factor=.6,
        buffer_len=3,
        **kwargs
    ):
        self.is_seq = is_seq
        self.controller = controller
        self.eval_env = environment
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist
        self.render = render
        self.dt = dt
        self.eval_dyn = eval_dyn
        self.dyn_eval_test = []
        self.action_counter = 0
        self.test_time = test_time
        self.speed_factor = speed_factor
        self.state_action_history = np.zeros((buffer_len, 12 + 4))
        self.use_random_actions = False
        self.use_mpc = None

    def help_render(self, sleep=.05):
        """
        Helper function to make rendering prettier
        """
        if self.render:
            # print([round(s, 2) for s in current_np_state])
            current_np_state = self.eval_env._state.as_np
            self.eval_env._state.set_position(
                current_np_state[:3] + np.array([0, 0, 1])
            )
            self.eval_env.render()
            self.eval_env._state.set_position(current_np_state[:3])
            time.sleep(sleep)

    def follow_trajectory(
        self,
        traj_type,
        max_nr_steps=200,
        thresh_stable=.4,
        thresh_div=3,
        do_avg_act=0,
        **traj_args
    ):
        """
        Follow a trajectory with the drone environment
        Argument trajectory: Can be any of
                straight
                circle
                hover
                poly
        """
        # reset action counter for new trajectory
        self.action_counter = 0

        # reset drone state
        init_state = [0, 0, 3]
        self.eval_env.zero_reset(*tuple(init_state))

        states = None  # np.load("id_5.npy")
        # Option to load data
        if states is not None:
            self.eval_env._state.from_np(states[0])

        # get current state
        current_np_state = self.eval_env._state.as_np

        # Get right trajectory object:
        object_dict = {
            "hover": Hover,
            "straight": Straight,
            "circle": Circle,
            "poly": Polynomial,
            "rand": Random
        }
        reference = object_dict[traj_type](
            current_np_state.copy(),
            self.render,
            self.eval_env.renderer,
            max_drone_dist=self.max_drone_dist,
            speed_factor=self.speed_factor,
            horizon=self.horizon,
            dt=self.dt,
            test_time=self.test_time,
            **traj_args
        )
        if traj_type == "rand":
            # self.eval_env._state.from_np(reference.initial_state)
            current_np_state = self.eval_env.zero_reset(
                *tuple(reference.initial_pos)
            )

        self.state_action_history[:, :12] = current_np_state
        self.state_action_history[:, 12:] = np.array([.5 for _ in range(4)])

        self.help_render()

        (reference_trajectory, drone_trajectory,
         divergences) = [], [current_np_state], []
        for i in range(max_nr_steps):
            # acc = self.eval_env.get_acceleration()
            trajectory = reference.get_ref_traj(current_np_state, 0)
            if self.is_seq:
                action = self.controller.predict_actions(
                    self.state_action_history,
                    trajectory.copy(),
                    is_seq=1,
                    hist_conv=1,
                    timestamp=self.eval_env.dynamics.timestamp
                )
            else:
                action = self.controller.predict_actions(
                    current_np_state, trajectory.copy()
                )

            # possible average with previous actions
            use_action = average_action(action, i, do_avg_act=do_avg_act)
            # REPLACE BY RANDOM OR MPC
            if self.use_random_actions:
                if i == 0:
                    print("Attention: use random action")
                use_action = np.random.rand(4)

            if self.use_mpc:
                if i == 0:
                    print("Attention: use MPC action")
                use_action = self.use_mpc.predict_actions(
                    current_np_state, trajectory.copy()
                )[0]

            if i % 10 == 0 and self.eval_dyn is not None:
                self.dyn_eval_test.append(
                    dyn_comparison_quad(
                        self.eval_dyn,
                        current_np_state,
                        use_action,
                        self.state_action_history,
                        self.eval_env.dynamics.timestamp,
                        dt=self.dt
                    )
                )
            current_np_state, stable = self.eval_env.step(
                use_action, thresh=thresh_stable
            )
            # np.set_printoptions(suppress=1, precision=3)
            # print(current_np_state[:3], trajectory[0, :3])
            if states is not None:
                self.eval_env._state.from_np(states[i])
                current_np_state = states[i]

            self.help_render(sleep=0)

            # update history
            self.state_action_history = np.roll(
                self.state_action_history, 1, axis=0
            )
            self.state_action_history[0, :12] = current_np_state
            self.state_action_history[0, 12:] = use_action

            drone_pos = current_np_state[:3]
            drone_trajectory.append(current_np_state)

            # project to trajectory and check divergence
            drone_on_line = reference.project_on_ref(drone_pos)
            reference_trajectory.append(drone_on_line)
            div = np.linalg.norm(drone_on_line - drone_pos)
            divergences.append(div)

            # reset the state to the reference
            if div > thresh_div or not stable:
                if self.test_time:
                    # TODO: must always be down for flightmare train
                    # print("diverged at", len(drone_trajectory))
                    break
                current_np_state = reference.get_current_full_state()
                self.eval_env._state.from_np(current_np_state)

            if i >= reference.ref_len:
                break
        if self.render:
            self.eval_env.close()
        # return trajectorie and divergences
        return (
            np.array(reference_trajectory), np.array(drone_trajectory),
            divergences
        )

    def compute_speed(self, drone_traj):
        """
        Compute speed, given a trajectory of drone positions
        """
        if len(drone_traj) == 0:
            return [0]
        dist = []
        for j in range(len(drone_traj) - 1):
            dist.append(
                (np.linalg.norm(drone_traj[j, :3] - drone_traj[j + 1, :3])) /
                self.dt
            )
        return [round(d, 2) for d in dist]

    def sample_circle(self):
        possible_planes = [[0, 1], [0, 2], [1, 2]]
        plane = possible_planes[np.random.randint(0, 3, 1)[0]]
        radius = np.random.rand() * 3 + 2
        direct = np.random.choice([-1, 1])
        circle_args = {"plane": plane, "radius": radius, "direction": direct}
        return circle_args

    def run_mpc_ref(
        self,
        reference: str,
        nr_test: int = 10,
        max_steps: int = 200,
        thresh_div=2,
        thresh_stable=2,
        **kwargs
    ):
        for _ in range(nr_test):
            _ = self.follow_trajectory(
                reference,
                max_nr_steps=max_steps,
                thresh_div=thresh_div,
                thresh_stable=thresh_stable,
                use_mpc_every=1
                # **circle_args
            )

    def run_eval(
        self,
        reference: str = "rand",
        nr_test: int = 10,
        max_steps: int = 200,
        thresh_div=1,
        thresh_stable=1,
        return_div=0,
        **kwargs
    ):
        """
        Function to evaluate a trajectory multiple times
        """
        self.dyn_eval_test = []
        if nr_test == 0:
            return 0, 0
        div, stable = [], []
        for _ in range(nr_test):
            if isinstance(self.controller, MPC):
                self.controller._initDynamics()
            # circle_args = self.sample_circle()
            _, drone_traj, divergences = self.follow_trajectory(
                reference,
                max_nr_steps=max_steps,
                thresh_div=thresh_div,
                thresh_stable=thresh_stable
                # **circle_args
            )
            div.append(np.mean(divergences))
            # before take over
            no_large_div = np.sum(np.array(divergences) < thresh_div)
            # print(np.mean(divergences), no_large_div)
            # no_large_div = np.where(np.array(divergences) > thresh_div)[0][0]
            stable.append(no_large_div)
            # stable.append(len(drone_traj))

        # Output results
        stable = np.array(stable)
        div_of_full_runs = np.array(div)[stable == np.max(stable)]
        print("----- control eval")
        print(
            "%s: Average div of full runs: %3.2f (%3.2f)" %
            (reference, np.mean(div_of_full_runs), np.std(div_of_full_runs))
        )
        print(
            "%s: Average div total: %3.2f (%3.2f)" %
            (reference, np.mean(div), np.std(div))
        )
        print(
            "%s: Steps until divergence: %3.2f (%3.2f)" %
            (reference, np.mean(stable), np.std(stable))
        )
        res_eval = {
            "mean_stable": np.mean(stable),
            "std_stable": np.std(stable),
            "mean_div": np.mean(div),
            "std_div": np.std(div)
        }
        if self.eval_dyn is not None and len(self.dyn_eval_test) > 0:
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
        return res_eval

    def collect_training_data(self, outpath="data/jan_2021.npy"):
        """
        Run evaluation but collect and save states as training data
        """
        data = []
        for _ in range(80):
            _, drone_traj = self.straight_traj(max_nr_steps=100)
            data.extend(drone_traj)
        for _ in range(20):
            # vary plane and radius
            possible_planes = [[0, 1], [0, 2], [1, 2]]
            plane = possible_planes[np.random.randint(0, 3, 1)[0]]
            radius = np.random.rand() + .5
            # run
            _, drone_traj = self.circle_traj(
                max_nr_steps=500, radius=radius, plane=plane
            )
            data.extend(drone_traj)
        data = np.array(data)
        print(data.shape)
        np.save(outpath, data)


def load_model(model_path, epoch=""):
    """
    Load model and corresponding parameters
    """
    # load std or other parameters from json
    net, param_dict = load_model_params(model_path, "model_quad", epoch=epoch)
    param_dict["self_play"] = 0
    if "seq" in model_path:
        dataset = QuadSequenceDataset(1, 0)
    else:
        dataset = QuadDataset(1, **param_dict)

    controller = NetworkWrapper(net, dataset, **param_dict)

    return controller, param_dict


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
        "-r", "--ref", type=str, default="rand", help="which trajectory"
    )
    parser.add_argument(
        "-a", "--eval", type=int, default=0, help="run evaluation for steps"
    )
    parser.add_argument(
        '-p',
        '--points',
        type=str,
        default=None,
        help="use predefined reference"
    )
    parser.add_argument(
        "-u", "--unity", action='store_true', help="unity rendering"
    )
    parser.add_argument(
        "-f", "--flightmare", action='store_true', help="Flightmare"
    )
    parser.add_argument(
        "-save_data",
        action="store_true",
        help="save the episode as training data"
    )
    args = parser.parse_args()

    DYNAMICS = "flightmare"
    RENDER = 1

    # CONTROLLER - define and load controller
    model_path = os.path.join("trained_models", "quad", args.model)
    # MPC
    if model_path.split(os.sep)[-1] == "mpc":
        # mpc parameters:
        params = {"horizon": 10, "dt": .1}
        load_dynamics = None
        # load_dynamics = 'trained_models/quad/dyn_tanh_woparams/dynamics_model'
        controller = MPC(
            dynamics=DYNAMICS, load_dynamics=load_dynamics, **params
        )
    # Neural controller
    else:
        controller, params = load_model(model_path, epoch=args.epoch)

    # PARAMETERS
    params["render"] = RENDER if not args.unity else 0
    is_seq = "seq" in args.model
    # params["dt"] = .05
    # params["max_drone_dist"] = 1
    params["speed_factor"] = .4
    modified_params = {}
    # {"wind": 2}
    # {"rotational_drag": np.array([.1, .1, .1])}
    # {"mass": 1}
    # {"translational_drag": np.array([.3, .3, .3])}
    # {
    #     "mass": 1,
    #     "frame_inertia": np.array([2, 2, 3]),
    #     "kinv_ang_vel_tau": np.array([21, 21, 3.0])
    # }
    print("MODIFIED: ", modified_params)

    # DEFINE ENVIRONMENT
    if args.flightmare:
        environment = FlightmareWrapper(params["dt"], args.unity)
    else:
        # DYNAMICS
        dynamics = (
            FlightmareDynamics(modified_params=modified_params)
            if DYNAMICS == "flightmare" else SimpleDynamics()
        )
        environment = QuadRotorEnvBase(dynamics, params["dt"])

    dyn_trained = None
    # dyn_trained = SequenceQuadDynamics(buffer_length=3)
    # dyn_trained.load_state_dict(
    #     torch.load(
    #         os.path.join(
    #             "trained_models/quad/iterative_seq_dyn_mpc",
    #             "dynamics_model_1000"
    #         )
    #     ),
    #     strict=True
    # )
    # EVALUATOR
    evaluator = QuadEvaluator(
        controller,
        environment,
        test_time=1,
        eval_dyn=dyn_trained,
        is_seq=is_seq,
        **params
    )

    # Specify arguments for the trajectory
    fixed_axis = 1
    traj_args = {
        "plane": [0, 2],
        "radius": 2,
        "direction": 1,
        "thresh_div": 5,
        "thresh_stable": 2,
        "duration": 10
    }
    if args.points is not None:
        from neural_control.trajectory.predefined_trajectories import (
            collected_trajectories
        )
        traj_args["points_to_traverse"] = collected_trajectories[args.points]

    # RUN
    if args.unity:
        evaluator.eval_env.env.connectUnity()

    if args.eval > 0:
        evaluator.render = 0
        # run_mpc_analysis(evaluator, system="quad")
        res_dict = evaluator.run_eval(
            args.ref, nr_test=args.eval, max_steps=500, **traj_args
        )
        # with open(
        #     f"../presentations/final_res/quad_seq_plot/{args.model}.json", "w"
        # ) as outfile:
        #     json.dump(res_dict, outfile)
        exit()

    # evaluator.run_mpc_ref(args.ref)
    reference_traj, drone_traj, divergences = evaluator.follow_trajectory(
        args.ref, max_nr_steps=2000, use_mpc_every=1000, **traj_args
    )

    if args.unity:
        evaluator.eval_env.env.disconnectUnity()

    # EVAL
    speed = evaluator.compute_speed(drone_traj[:, :3])
    print(
        "Speed: max:", round(np.max(speed), 2), ", mean:",
        round(np.mean(speed), 2), "stopped at", len(drone_traj),
        "avg tracking error", np.mean(divergences), "max", np.max(divergences)
    )
    # print(speed)
    plot_trajectory(
        reference_traj,
        drone_traj,
        os.path.join(model_path, args.ref + "_traj.png"),
        fixed_axis=fixed_axis
    )
    plot_drone_ref_coords(
        drone_traj[1:, :3], reference_traj,
        os.path.join(model_path, args.ref + "_coords.png")
    )
