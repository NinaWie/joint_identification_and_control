import numpy as np
import matplotlib.pyplot as plt

from neural_control.environments.rl_envs import (
    CartPoleEnvRL, WingEnvRL, QuadEnvRL
)
from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics
from neural_control.dynamics.cartpole_dynamics import CartpoleDynamics
from neural_control.dynamics.fixed_wing_dynamics import FixedWingDynamics

num_required = 100000

# dt = 0.05
# dyn = CartpoleDynamics()
# env = CartPoleEnvRL(dyn, dt=dt)
# random_actions_1 = np.random.rand(num_required, 1) * 2 - 1
# random_actions_2 = np.random.rand(num_required, 1) * 2 - 1

dt = 0.05
dyn = FixedWingDynamics()
env = WingEnvRL(dyn, dt)
env.thresh_div = 4
env.thresh_stable = .5
# action between 0 and 1
random_actions_1 = np.random.rand(num_required, 4)
random_actions_2 = np.random.rand(num_required, 4)

# dt = 0.05
# dyn = FlightmareDynamics()
# env = QuadEnvRL(dyn, dt)
# random_actions_1 = np.random.rand(num_required, 4) * 2 - 1
# random_actions_2 = np.random.rand(num_required, 4) * 2 - 1

np.set_printoptions(suppress=1, precision=3)
lipschitz_factors, state_action_tuples = [], []
counter = 0
for i in range(500):
    # print("------ reset ------")
    env.reset()
    done = False
    prev_counter = counter
    while not done:
        prev_state = env.state.copy()

        # first action
        action1 = random_actions_1[counter]
        _, _, done, _ = env.step(action1)
        state1 = env.state.copy()
        # second action
        #         set_state(env, prev_state)

        env.set_state(prev_state.copy())
        action2 = random_actions_2[counter]
        _, rew, done, _ = env.step(action2)
        state2 = env.state.copy()
        # second action executed
        state_diff_1 = (state1 - prev_state) / dt
        state_diff_2 = (state2 - prev_state) / dt

        state_diff = np.linalg.norm(state_diff_2 - state_diff_1)
        action_diff = np.linalg.norm(action2 - action1)

        # print(prev_state)
        # print(action1, state1)
        # print(action2, state2)
        # print(state_diff_1, state_diff_2)
        # print(state_diff, action_diff)
        # print(state_diff / action_diff)
        # print(env._state)
        # print(np.any(np.absolute(env._state[6:8]) >= env.thresh_stable))
        # print()
        lipschitz_factors.append(state_diff / action_diff)
        state_action_tuples.append(
            (prev_state, action1, action2, state1, state2)
        )
        counter += 1

print(
    "Maximum is", np.max(lipschitz_factors), "times dt",
    np.max(lipschitz_factors) * dt
)

import pickle
with open(
    '../presentations/final_res/lipschitz/wing_lipschitz.pickle', 'wb'
) as f:
    pickle.dump((lipschitz_factors, state_action_tuples), f)

plt.hist(lipschitz_factors)
plt.show()
