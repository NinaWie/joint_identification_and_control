import numpy as np
import torch
import matplotlib.pyplot as plt

img_width, img_height, radius = (8, 8, 0)
image_bounds = np.array([img_width, img_height])

ball_x_options = [-1, 0, 1]
ball_y_options = [-1, 0, 1]
knight_x_options = [0, 2, 1, -1, -2, -2, -1, 1, 2]
knight_y_options = [0, 1, 2, 2, 1, -1, -2, -2, -1]


def number_moves_knight(x, y):
    # switch if necessary
    if x < y:
        temp = x
        x = y
        y = temp

    # 2 corner cases
    if x == 1 and y == 0:
        return 3
    if x == 2 and y == 2:
        return 4

    delta = x - y

    if y > delta:
        return delta - 2 * np.floor((delta - y) / 3)
    else:
        return delta - 2 * np.floor((delta - y) / 4)


def number_moves_ball(x, y):
    return max([x, y])


def dice_loss(pred, target):
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    return 1 - (numerator + 1) / (denominator + 1)


def one_hot_to_cmd_ball(one_hot):
    idx = np.argmax(one_hot)
    return (ball_x_options[idx // 3], ball_y_options[idx % 3])


def one_hot_to_cmd_knight(one_hot):
    idx = np.argmax(one_hot)
    return (knight_x_options[idx], knight_y_options[idx])


def get_img(img_width, img_height, center, radius):
    img = np.zeros((img_width, img_height))
    img[center[0] - radius:center[0] + radius + 1,
        center[1] - radius:center[1] + radius + 1] = 1
    return img


def get_torch_img(np_img):
    return torch.tensor([np_img.tolist()]).float()


def round_diff(x, slope=10):
    e = torch.exp(slope * (x - .5))
    return e / (e + 1)


def add_grid(ax):
    ax.set_xticks(np.arange(-.5, img_width, 1 + 2 * radius))
    ax.set_yticks(np.arange(-.5, img_height, 1 + 2 * radius))
    ax.grid(color='black', linestyle='-', linewidth=1)
    plt.grid(True)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')


def plot_chess(center, center_king):
    img = np.zeros((img_width, img_height))
    # img[center[0]-radius:center[0]+radius+1, center[1]-radius:center[1]+radius+1] = 1
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    bbox = fig.axes[0].get_position().bounds
    print(bbox)
    each_cell_x, each_cell_y = bbox[2] / img_width, bbox[3] / img_height

    pos_horse_x = bbox[0] + each_cell_x * (center[1] - radius)
    pos_horse_y = bbox[1] + each_cell_y * (img_width - center[0] - radius - 1)

    newax = fig.add_axes(
        [
            pos_horse_x, pos_horse_y, each_cell_x *
            (1 + 2 * radius), each_cell_y * (1 + 2 * radius)
        ],
        zorder=1
    )
    newax.imshow(horse)
    newax.axis('off')

    pos_king_x = bbox[0] + each_cell_x * (center_king[1] - radius)
    pos_king_y = bbox[
        1] + each_cell_y * (img_width - center_king[0] - radius - 1)

    newax = fig.add_axes(
        [
            pos_king_x, pos_king_y, each_cell_x *
            (1 + 2 * radius), each_cell_y * (1 + 2 * radius)
        ],
        zorder=1
    )
    newax.imshow(king)
    newax.axis('off')

    # Gridlines based on minor ticks
    ax.set_xticks(np.arange(-.5, 8, 1 + 2 * radius), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1 + 2 * radius), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    # ax.axis("off")
    plt.grid(True)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.show()


def test_qualitatively(model_save_path, dynamics_path, nr_actions):
    """
    ATTENTION: currently in receding horizon!

    Args:
        model_save_path ([type]): [description]
        dynamics_path ([type]): [description]
        nr_actions ([type]): [description]
    """
    con = torch.load(model_save_path)
    dyn = torch.load(dynamics_path)
    dyn.trainable = False

    state = (4, 5)
    target = (2, 4)

    x, y = state
    test_img = torch.zeros(1, 8, 8)
    start_x = max([0, x - radius])
    end_x = min([x + radius + 1, img_width])
    start_y = max([0, y - radius])
    end_y = min([y + radius + 1, img_height])
    test_img[0, start_x:end_x, start_y:end_y] = 1

    x, y = target
    test_img_out = torch.zeros(1, 8, 8)
    start_x = max([0, x - radius])
    end_x = min([x + radius + 1, img_width])
    start_y = max([0, y - radius])
    end_y = min([y + radius + 1, img_height])
    test_img_out[0, start_x:end_x, start_y:end_y] = 1

    current_state = test_img.clone().float()
    # apply all at once
    # pred_cmd = con(test_img, test_img_out)
    # print("predicted command", pred_cmd)
    # for action_ind in range(nr_actions):
    #     current_state = dyn(current_state, pred_cmd[:, action_ind])

    plt.figure(figsize=(10, 10))
    # Apply in receding horizon
    for i in range(nr_actions):
        input_center = torch.argmax(current_state).item()
        input_center_x = input_center // img_height
        input_center_y = input_center % img_height

        pred_cmd = con(current_state, test_img_out)
        current_state_before = current_state.clone()

        # np_cmd = one_hot_to_cmd_knight(pred_cmd[0, 0].detach().numpy())
        raw_cmd = pred_cmd[0, 0].detach().numpy()
        np_cmd = [round(raw_cmd[0]), round(raw_cmd[1])]
        np.set_printoptions(suppress=1, precision=3)
        print("raw command", raw_cmd)
        print("predicted command", np_cmd)

        transform_cmd = pred_cmd[:, 0]
        # transform_cmd = torch.zeros(pred_cmd[:, 0].size())
        # transform_cmd[0, np.argmax(pred_cmd[:, 0].detach().numpy())] = 1
        # print(current_state)
        # print(transform_cmd)
        current_state = dyn(current_state, transform_cmd)

        ax1 = plt.subplot(nr_actions, 3, 1 + i * 3)
        ax1.imshow(current_state_before[0].detach().numpy(), cmap="Greys")
        add_grid(ax1)
        ax1.set_title(
            "Input 1:\nCurrent state " + str((input_center_x, input_center_y))
        )
        ax2 = plt.subplot(nr_actions, 3, 2 + i * 3)
        ax2.imshow(test_img_out[0].detach().numpy(), cmap="Greys")
        add_grid(ax2)
        ax2.set_title("Input 2:\nTarget state " + str(target))
        ax3 = plt.subplot(nr_actions, 3, 3 + i * 3)
        ax3.imshow(current_state[0].detach().numpy(), cmap="Greys")
        add_grid(ax3)
        ax3.set_title("Applying predicted\n command " + str(np_cmd))
    plt.tight_layout()
    plt.savefig("../neurips/figures/horse_chess_secondpart.pdf")
    plt.show()
