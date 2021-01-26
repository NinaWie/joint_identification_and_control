import torch
from environments.drone_dynamics import simulate_quadrotor, device
from environments.mpc_drone_env import Quadrotor_v0
torch.autograd.set_detect_anomaly(True)
zero_tensor = torch.zeros(3).to(device)


def mpc_loss(self, states, target_states, action, target_end_state):
    gt_action = torch.tensor([9.81, 0, 0, 0]).to(device)

    state_error = states - target_states  # shape 20 x 10
    state_loss = torch.sum(
        torch.mm(torch.mm(state_error, self._Q_gap), state_error.t())
    )
    # print(state_error.size(), state_loss.size())
    target_error = states[-1] - target_end_state  # shape 20 x 10
    inner_mv = torch.mv(self._Q_goal, target_error)
    target_loss = torch.sum(torch.dot(inner_mv, target_error.t()))
    # add up action losses
    action_loss = 0
    for k in range(self.horizon):
        action_error = actions[k] - self.action_prior[k]
        action_loss += torch.dot(
            torch.mv(self._Q_u, action_error), action_error.t()
        )
    return torch.sum(state_loss + target_loss) * 0.00001


def drone_loss_function(current_state, start_state=None, printout=0):
    """
    Computes loss of the current state (target is assumed to be zero-state)
    Arguments:
        current_state: array with x entries describing attitude and velocity
        start_state: same format as current_state, but start position (for
            normalization)
    Returns:
        loss (torch float scalar)
    """
    # weighting
    angle_factor = 1
    angvel_factor = 2e-2
    pos_factor = .5

    # attittude and att velocity loss
    angle_error = torch.sum(current_state[:, 3:6]**2, axis=1)
    ang_vel_error = torch.sum(current_state[:, 13:16]**2, axis=1)

    # position loss
    div, prog = pos_traj_loss(start_state[:, :3], current_state[:, :3])
    position_loss = div + 3 * prog  # added 3 only
    # torch.sum(
    #     current_state[:, :3]**2 * torch.tensor([.5, 2, 2]), dim=1
    # )

    # angle_factor = torch.relu(angle_factor - position_loss)

    # together
    loss = (
        angle_factor * angle_error + angvel_factor * ang_vel_error +
        pos_factor * position_loss
    )

    if printout:
        print()
        print("attitude loss", (angle_factor * angle_error)[0])
        print("att vel loss", (angvel_factor * ang_vel_error)[0])
        print("position loss", (pos_factor * position_loss)[0])
    return torch.sum(loss)


def reference_loss(states, ref_states, actions, printout=0, delta_t=0.1):
    """
    Compute loss with respect to reference trajectory
    """
    # TODO: add loss on actions with quaternion formulation
    # (9.81, 0,0,0)
    # TODO: include attitude in reference
    # angle_factor = 0.1
    # angvel_factor = 2e-2
    # vel_factor = 0.01
    # pos_factor = 0.5
    # yaw_factor = 10
    action_factor = .1

    # import numpy as np
    # np.set_printoptions(precision=3, suppress=True)
    # print(states[0])
    # print(ref_states[0])
    # print(actions[0])

    # compute euler angles:
    # angle_error = 0
    # for i in range(states.size()[0]):
    #     for j in range(states.size()[1]):
    #         euler_angles = Quadrotor_v0._quatToEuler(states[i, j, 3:7])
    #         # print("euler_angles", euler_angles)
    #         angle_error += torch.sum(euler_angles**2)

    # position_loss = torch.sum((states[:, :, :3] - ref_states[:, :, :3])**2)
    # velocity_loss = torch.sum((states[:, :, 7:] - ref_states[:, :, 3:6])**2)

    # penalize actions - should optimally be all zero
    action_loss = torch.sum(actions**2)
    # print("position_loss", position_loss)
    # print("velocity_loss", velocity_loss)
    # exit()
    # loss = angle_error * .01

    # angle_error = 0
    # for k in range(states.size()[1] - 2):
    #     # approximate acceleration
    #     acc = (states[:, k + 1, 7:] - states[:, k, 7:]) / delta_t
    #     acc_ref = ref_states[:, k, 6:] * delta_t
    #     # subtract from desired acceleration
    #     angle_error += torch.sum((acc_ref - acc)**2)

    # OLD VERSION
    angle_factor = 0.01
    angvel_factor = 2e-2
    vel_factor = 0.1
    pos_factor = 1
    yaw_factor = 10

    position_loss = torch.sum((states[:, :, :3] - ref_states[:, :, :3])**2)
    velocity_loss = torch.sum((states[:, :, 7:] - ref_states[:, :, 3:6])**2)
    # print(states[0, :, 7:])
    # print(ref_states[0, :, 3:6])
    # print()

    angle_error = 0
    # for k in range(states.size()[1] - 2):
    #     # approximate acceleration
    #     acc = (states[:, k + 1, 7:] - states[:, k, 7:]) / delta_t
    #     acc_ref = ref_states[:, k, 6:] * delta_t
    #     # subtract from desired acceleration
    #     angle_error += torch.sum((acc_ref - acc)**2)

    loss = (
        angle_factor * angle_error + pos_factor * position_loss +
        vel_factor * velocity_loss + action_factor * action_loss
    )

    if printout:
        print()
        print("attitude loss", (angle_factor * angle_error).item())
        # print("att vel loss", (angvel_factor * ang_vel_error).item())
        print("velocity loss", (velocity_loss * vel_factor).item())
        print("position loss", (pos_factor * position_loss).item())
        print("action loss", (action_factor * action_loss).item())
    return loss


def project_to_line(a_on_line, b_on_line, p):
    """
    Project a point p to a line from a to b
    Arguments:
        All inputs are 2D tensors of shape (BATCH_SIZE, n)
        a_on_line: First point on the line
        b_on_line: Second point on the line
        p: point to be projected onto the line
    Returns: Tensor of shape (BATCH_SIZE, n) which is the orthogonal projection
            of p on the line
    """
    ap = torch.unsqueeze(p - a_on_line, 2)
    ab = b_on_line - a_on_line
    # normalize
    norm = torch.sum(ab**2, axis=1)
    v = torch.unsqueeze(ab, 2)
    # vvT * (p-a)
    dot = torch.matmul(v, torch.transpose(v, 1, 2))
    product = torch.squeeze(torch.matmul(dot, ap))
    # add a to move away from origin again
    projected = a_on_line + (product.t() / norm).t()

    return projected


def pos_traj_loss(start_state, drone_state):
    """
    Compute position loss based on the projection of the drone state on the
    target trajectory (from start_state to zero)
    Arguments: (Shape BATCH_SIZE, 3)
        start_state: Drone position before action
        drone_state: Position after applying action to start_state
    Returns:
        Two tensors, each of shape (BATCH_SIZE, 1)
        divergence_loss: divergence from the target trajectory
        progress_loss: 1 - how far did the drone progress towards the target
    """
    # distance from start to target
    total_distance = torch.sum(start_state**2, 1)
    # project to trajectory
    projected_state = project_to_line(start_state, zero_tensor, drone_state)
    # losses
    divergence_loss = torch.sum(
        (projected_state - drone_state)**2, 1
    ) / total_distance
    progress_loss = torch.sum(projected_state**2, 1) / total_distance
    return divergence_loss, progress_loss


def trajectory_loss(
    state,
    target_state,
    drone_state,
    loss_weights=None,
    mask=None,
    printout=0
):
    """
    Trajectory loss for position and attitude (in contrast to pos_traj_loss)
    Input states must be normalized!
    """
    if mask is None:
        mask = torch.ones(state.size()[1])
    else:
        state = state * mask

    # multiply losses by 0 or weights
    mask = mask * loss_weights

    # normalize by distance between states
    total_distance = torch.sum((state - target_state)**2, 1)

    projected_state = project_to_line(state, target_state, drone_state)

    # divergence from the desired route
    divergence_loss_all = (projected_state - drone_state)**2 * mask
    divergence_loss = torch.sum(divergence_loss_all, 1)

    # minimize remaining distance to target (normalized on total distance)
    progress_loss_all = (projected_state - target_state)**2 * mask
    progress_loss = torch.sum(progress_loss_all, 1) / total_distance
    if printout:
        print(
            total_distance.size(), progress_loss_all.size(),
            progress_loss.size()
        )
        print("state", state[0])
        print("target", target_state)
        print("drone", drone_state[0])
        print("divergence all", divergence_loss_all[0])
        print("progress all", progress_loss_all[0])
        print("divergence", divergence_loss[0].item())
        print("progress_loss", progress_loss[0].item())
        print("final", progress_loss + .1 * divergence_loss)
        exit()
    return torch.sum(progress_loss + .1 * divergence_loss)
