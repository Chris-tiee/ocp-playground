import casadi as ca
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(local_path, ".."))

import models.bicycleXYModel as bicycleXYModel


@dataclass
class GoalReachingCtrlConfig:
    max_delta: float = math.radians(15.0)
    max_V: float = 3.0
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 1.0]))
    Q_e: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 10.0]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1]))
    n_hrzn = 50
    num_agents = 2


sampling_time = 0.05
num_agents = 2
ctrl_cfg = GoalReachingCtrlConfig()

# initial and goal (example from original open_loop)
x_init = np.array([0.0, 1.5, 0., 1.5, 0.0, np.pi*0.5, ])
goal = np.array([3.2, 1.5, 0., 1.5, 3.0, np.pi*0.5, ])

# Model
model = bicycleXYModel.BicycleXYModel(sampling_time)
nx = model.model_config.nx
nu = model.model_config.nu
n_hrzn = ctrl_cfg.n_hrzn

# Decision variables
x_opt = ca.MX.sym('x', nx * num_agents, n_hrzn + 1)
u_opt = ca.MX.sym('u', nu * num_agents, n_hrzn)

# Parameters (initial state and goal)
x_0_param = ca.MX.sym('x_0', nx * num_agents)
goal_param = ca.MX.sym('goal', nx * num_agents)

# Constraints and objective
g = []
lbg = []
ubg = []

# Input constraints per timestep
for k in range(n_hrzn):
    # steering limits (first input of each agent)
    g.append(u_opt[0::nu, k])
    lbg.append(-ctrl_cfg.max_delta * np.ones(num_agents))
    ubg.append(ctrl_cfg.max_delta * np.ones(num_agents))
    # velocity limits (second input of each agent)
    g.append(u_opt[1::nu, k])
    lbg.append(np.zeros(num_agents,))
    ubg.append(ctrl_cfg.max_V * np.ones(num_agents))

# Initial state equality
g.append(x_opt[:nx * num_agents, 0] - x_0_param)
lbg.append(np.zeros(nx * num_agents,))
ubg.append(np.zeros(nx * num_agents,))

# Dynamics
for i_agent in range(num_agents):
    for k in range(n_hrzn):
        x_slice = x_opt[i_agent * nx:(i_agent + 1) * nx, k]
        u_slice = u_opt[i_agent * nu:(i_agent + 1) * nu, k]
        g.append(x_opt[i_agent * nx:(i_agent + 1) * nx, k + 1] - model.f_disc(x_slice, u_slice))
        lbg.append(np.zeros(nx,))
        ubg.append(np.zeros(nx,))

# Collision avoidance (pairwise) -- ensure agents stay apart
for i_agent in range(num_agents):
    for j_agent in range(i_agent + 1, num_agents):
        for k in range(1, n_hrzn + 1):
            pos_i = x_opt[i_agent * nx:i_agent * nx + 2, k]
            pos_j = x_opt[j_agent * nx:j_agent * nx + 2, k]
            g.append(ca.sumsqr(pos_i - pos_j))
            lbg.append((model.model_config.safety_radius * 2) ** 2)
            ubg.append(ca.inf)

# Objective
J = 0.0
for i_agent in range(num_agents):
    for k in range(n_hrzn):
        x_diff = x_opt[i_agent * nx:(i_agent + 1) * nx, k] - goal_param[i_agent * nx:(i_agent + 1) * nx]
        J += x_diff.T @ ctrl_cfg.Q @ x_diff
        J += u_opt[i_agent * nu:(i_agent + 1) * nu, k].T @ ctrl_cfg.R @ u_opt[i_agent * nu:(i_agent + 1) * nu, k]
    # terminal cost
    x_diff = x_opt[i_agent * nx:(i_agent + 1) * nx, -1] - goal_param[i_agent * nx:(i_agent + 1) * nx]
    J += x_diff.T @ ctrl_cfg.Q_e @ x_diff

# Build NLP
ocp = {
    'x': ca.veccat(x_opt, u_opt),
    'p': ca.vertcat(x_0_param, goal_param),
    'g': ca.vertcat(*g),
    'f': J
}
solver = ca.nlpsol('solver', 'ipopt', ocp)

# Initial guesses
x0_guess = np.tile(x_init.flatten()[:, np.newaxis], (1, n_hrzn + 1))
u0_guess = np.zeros((nu * num_agents, n_hrzn))

# Solve
sol = solver(
    x0=ca.veccat(x0_guess, u0_guess),
    p=ca.vertcat(x_init.flatten(), goal.flatten()),
    lbg=ca.vertcat(*lbg),
    ubg=ca.vertcat(*ubg)
)

sol_vec = sol['x'].full().flatten()
x_sol = sol_vec[:(n_hrzn + 1) * nx * num_agents].reshape((nx * num_agents, n_hrzn + 1), order='F')
u_sol = sol_vec[(n_hrzn + 1) * nx * num_agents:].reshape((nu * num_agents, n_hrzn), order='F')

print(f"Optimal cost: {sol['f'].full().flatten()[0]}")

# Visualize using model helper (same as original)
# plot states and controls
fig, axs = model.plotSimulation(x_sol, u_sol, num_agents=num_agents)
# plt.show()
additional_lines_or_scatters = {"Goal": {"type": "scatter", "data": [[goal[0], goal[3]], [goal[1], goal[4]]], "color": "tab:orange", "s": 100, "marker": "x"}}
# To save the animation as GIF, uncomment these lines:
# save_gif_path = os.path.join(local_path, "bicycle_multi.gif")
# model.animateSimulation(x_sol, u_sol, num_agents=num_agents, additional_lines_or_scatters=additional_lines_or_scatters, save_path=save_gif_path)
model.animateSimulation(x_sol, u_sol, num_agents=num_agents, additional_lines_or_scatters=additional_lines_or_scatters)