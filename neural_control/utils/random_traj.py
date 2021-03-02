import numpy as np
from .generate_trajectory import generate_trajectory


class Random:

    def __init__(
        self,
        drone_state,
        render=False,
        renderer=None,
        max_drone_dist=0.25,
        horizon=10,
        duration=10,
        dt=0.05,
        **kwargs
    ):
        """
        Create random trajectory
        """
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist
        self.dt = dt
        # make variable whether we are already finished with the trajectory
        self.finished = False
        if render and renderer is None:
            raise ValueError("if render is true, need to input renderer")

        points_3d = generate_trajectory(duration, dt)

        # subtract current position to start there
        # points_3d[:, :3] = points_3d[:, :3] - points_3d[
        #     0, :3] + drone_state[:3]

        self.reference = points_3d
        self.ref_len = len(self.reference)
        self.target_ind = 0
        self.current_ind = 0

        # draw trajectory on renderer
        if render:
            renderer.add_object(PolyObject(self.reference))

    def get_ref_traj(self, drone_state, drone_acc):
        """
        Given the current position, compute a min snap trajectory to the next
        target
        """
        # if already at end, return zero velocities and accelerations
        if self.current_ind >= len(self.reference) - self.horizon:
            zero_ref = np.zeros(
                (
                    self.horizon - (self.ref_len - self.current_ind),
                    self.reference.shape[1]
                )
            )
            zero_ref[:, :3] = self.reference[-1, :3]
            left_over_ref = self.reference[self.current_ind:]
            return np.vstack((left_over_ref, zero_ref))
        out_ref = self.reference[self.current_ind + 1:self.current_ind +
                                 self.horizon + 1]
        self.current_ind += 1
        return out_ref

    def project_on_ref(self, drone_state):
        """
        Project drone state onto the trajectory
        """
        return self.reference[self.current_ind, :3]


class PolyObject():

    def __init__(self, reference_arr):
        self.points = np.array(
            [
                reference_arr[i] for i in range(len(reference_arr))
                if i % 20 == 0
            ]
        )
        self.points[:, 2] += 1

    def draw(self, renderer):
        for p in range(len(self.points) - 1):
            renderer.draw_line_3d(
                self.points[p], self.points[p + 1], color=(1, 0, 0)
            )
