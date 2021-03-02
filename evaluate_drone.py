import os
import time
import argparse
import json
import numpy as np
import torch

from neural_control.environments.drone_env import (
    QuadRotorEnvBase, trajectory_training_data
)
from neural_control.utils.plotting import (
    plot_state_variables, plot_trajectory, plot_position, plot_suc_by_dist,
    plot_drone_ref_coords, print_state_ref_div
)
from neural_control.utils.straight import Hover, Straight
from neural_control.utils.circle import Circle
from neural_control.utils.polynomial import Polynomial
from neural_control.utils.random_traj import Random
from neural_control.dataset import DroneDataset
from neural_control.controllers.network_wrapper import NetworkWrapper
from neural_control.controllers.mpc import MPC
try:
    from neural_control.flightmare import FlightmareWrapper
except ModuleNotFoundError:
    pass

ROLL_OUT = 3

# Use cuda if available
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuadEvaluator():

    def __init__(
        self,
        controller,
        horizon=5,
        max_drone_dist=0.1,
        render=0,
        dt=0.02,
        **kwargs
    ):
        self.controller = controller
        self.eval_env = QuadRotorEnvBase(dt)
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist
        self.render = render
        self.dt = dt
        self.action_counter = 0

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
            horizon=self.horizon,
            dt=self.dt,
            **traj_args
        )
        if traj_type == "rand":
            self.eval_env._state.from_np(reference.reference[0])
            current_np_state = self.eval_env._state.as_np

        self.help_render()
        # start = input("start")

        (reference_trajectory, drone_trajectory,
         divergences) = [], [current_np_state], []
        for _ in range(max_nr_steps):
            acc = self.eval_env.get_acceleration()
            trajectory = reference.get_ref_traj(current_np_state, acc)
            numpy_action_seq = self.controller.predict_actions(
                current_np_state, trajectory
            )
            # only use first action (as in mpc)
            # np_ref = trajectory
            # np_state = np.zeros((ROLL_OUT, 12))
            # TODO: if roll out>0 then need to increase currend_ind von ref
            for k in range(ROLL_OUT):
                action = numpy_action_seq[k]
                current_np_state, stable = self.eval_env.step(
                    action, thresh=thresh_stable
                )
                self.help_render(sleep=0)
                # np_state[k] = current_np_state.copy()
                # np.set_printoptions(suppress=1, precision=3)
                # print(action)
                # print(current_np_state[:3], trajectory[k, :3])
                if k > 0:
                    reference.current_ind += 1
            # print(self.eval_env._state.as_np)
            # print("action", numpy_action_seq)
            # print_state_ref_div(np_ref, np_state)
            # exit()

            if states is not None:
                self.eval_env._state.from_np(states[i])
                current_np_state = states[i]
                stable = i < (len(states) - 1)
            if not stable:
                if self.render:
                    np.set_printoptions(precision=3, suppress=True)
                    print("unstable")
                    # print(self.eval_env._state.as_np)
                break

            drone_pos = current_np_state[:3]
            drone_trajectory.append(current_np_state)

            # project to trajectory and check divergence
            drone_on_line = reference.project_on_ref(drone_pos)
            reference_trajectory.append(drone_on_line)
            div = np.linalg.norm(drone_on_line - drone_pos)
            divergences.append(div)
            if div > thresh_div:
                if self.render:
                    np.set_printoptions(precision=3, suppress=True)
                    print("state")
                    print([round(s, 2) for s in current_np_state])
                    print("trajectory:")
                    print(np.around(trajectory, 2))
                break
        if self.render:
            self.eval_env.close()
        # return trajectorie and divergences
        return (
            np.array(reference_trajectory), np.array(drone_trajectory),
            divergences
        )

    def compute_speed(self, drone_traj):
        dist = 0
        for j in range(len(drone_traj) - 1):
            dist += np.linalg.norm(drone_traj[j, :3] - drone_traj[j + 1, :3])

        time_passed = len(drone_traj) * self.dt
        speed = dist / time_passed
        return speed

    def sample_circle(self):
        possible_planes = [[0, 1], [0, 2], [1, 2]]
        plane = possible_planes[np.random.randint(0, 3, 1)[0]]
        radius = np.random.rand() * 3 + 2
        direct = np.random.choice([-1, 1])
        circle_args = {"plane": plane, "radius": radius, "direction": direct}
        return circle_args

    def eval_ref(
        self,
        reference: str,
        nr_test: int = 10,
        max_steps: int = 200,
        thresh_div=1,
        thresh_stable=1
    ):
        """
        Function to evaluate a trajectory multiple times
        """
        if nr_test == 0:
            return 0, 0
        div, stable = [], []
        for _ in range(nr_test):
            circle_args = self.sample_circle()
            _, drone_traj, divergences = self.follow_trajectory(
                reference,
                max_nr_steps=max_steps,
                thresh_div=thresh_div,
                thresh_stable=thresh_stable,
                **circle_args
            )
            div.append(np.mean(divergences))
            stable.append(len(drone_traj))

        # Output results
        print("Speed (last):", self.compute_speed(drone_traj))
        print(
            "%s: Average divergence: %3.2f (%3.2f)" %
            (reference, np.mean(div), np.std(div))
        )
        print(
            "%s: Steps until divergence: %3.2f (%3.2f)" %
            (reference, np.mean(stable), np.std(stable))
        )
        return np.mean(stable), np.std(stable)

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


def load_model_params(model_path, name="model_quad", epoch=""):
    with open(os.path.join(model_path, "param_dict.json"), "r") as outfile:
        param_dict = json.load(outfile)

    net = torch.load(os.path.join(model_path, name + epoch))
    net = net.to(device)
    net.eval()
    return net, param_dict


def load_model(model_path, epoch="", horizon=10, dt=0.05, **kwargs):
    """
    Load model and corresponding parameters
    """
    if "mpc" not in model_path:
        # load std or other parameters from json
        net, param_dict = load_model_params(
            model_path, "model_quad", epoch=epoch
        )
        dataset = DroneDataset(1, 1, **param_dict)

        controller = NetworkWrapper(net, dataset, **param_dict)
    else:
        controller = MPC(horizon, dt, dynamics="simple_quad")
    return controller


if __name__ == "__main__":
    # make as args:
    parser = argparse.ArgumentParser("Model directory as argument")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="test_model",
        help="Directory of model"
    )
    parser.add_argument(
        "-e", "--epoch", type=str, default="", help="Saved epoch"
    )
    parser.add_argument(
        "-r", "--ref", type=str, default="rand", help="which trajectory"
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

    params = {"render": 1, "dt": 0.05, "horizon": 3, "max_drone_dist": .5}

    # rendering
    if args.unity:
        params["render"] = 0

    # load model
    model_path = os.path.join("trained_models", "drone", args.model)
    controller = load_model(model_path, epoch=args.epoch, **params)

    # define evaluation environment
    evaluator = QuadEvaluator(controller, **params)

    # FLIGHTMARE
    if args.flightmare:
        evaluator.eval_env = FlightmareWrapper(params["dt"], args.unity)

    # Specify arguments for the trajectory
    fixed_axis = 1
    traj_args = {
        "plane": [0, 2],
        "radius": 2,
        "direction": 1,
        "thresh_div": np.inf,
        "thresh_stable": np.inf
    }
    if args.points is not None:
        from neural_control.utils.predefined_trajectories import (
            collected_trajectories
        )
        traj_args["points_to_traverse"] = collected_trajectories[args.points]

    # RUN
    if args.unity:
        evaluator.eval_env.env.connectUnity()

    reference_traj, drone_traj, _ = evaluator.follow_trajectory(
        args.ref, max_nr_steps=500, **traj_args
    )

    if args.unity:
        evaluator.eval_env.env.disconnectUnity()

    # EVAL
    print("Speed:", evaluator.compute_speed(drone_traj[100:300, :3]))
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
