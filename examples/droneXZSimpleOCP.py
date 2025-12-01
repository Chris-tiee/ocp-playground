import casadi as ca
from dataclasses import dataclass, field
import numpy as np
import os
import sys

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(local_path, ".."))

import models.droneXZModel as droneXZModel
import matplotlib.pyplot as plt


@dataclass
class GoalReachingCtrlConfig:
    max_fl: float = 15.0
    max_fr: float = 15.0

    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.1, 0.1, 1.0, 0.1]))
    Q_e: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 1.0, 1.0, 10.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01]))
    n_hrzn = 40


# --- Linear/script style open-loop OCP (no MPC) ---
sampling_time = 0.05
cfg = GoalReachingCtrlConfig()
# increase horizon to 30 as requested

# initial and goal (from original main)
x_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
goal = np.array([1.0, 1.0])

model = droneXZModel.DroneXZModel(sampling_time)
nx = model.model_config.nx
nu = model.model_config.nu
N = cfg.n_hrzn

# Decision variables
x_opt = ca.MX.sym('x', nx, N+1)
u_opt = ca.MX.sym('u', nu, N)

# Parameters
x_0_param = ca.MX.sym('x_0', nx)
goal_param = ca.MX.sym('goal', 2)

# Constraints and objective
g = []
lbg = []
ubg = []

# Input constraints
for k in range(N):
    g.append(u_opt[0, k])
    lbg.append(0.0)
    ubg.append(cfg.max_fl)
    g.append(u_opt[1, k])
    lbg.append(0.0)
    ubg.append(cfg.max_fr)

# initial condition equality
g.append(x_opt[:, 0] - x_0_param)
lbg.append(np.zeros(nx,))
ubg.append(np.zeros(nx,))

# dynamics equality
for k in range(N):
    g.append(x_opt[:, k+1] - model.f_disc(x_opt[:, k], u_opt[:, k]))
    lbg.append(np.zeros(nx,))
    ubg.append(np.zeros(nx,))

# objective
J = 0.0
x_goal = ca.veccat(goal_param, ca.DM.zeros(4, 1))
u_equilibrium = 0.5 * model.model_config.mass * model.model_config.gravity * ca.DM.ones(2, 1)
for k in range(N):
    J += (x_opt[:, k] - x_goal).T @ cfg.Q @ (x_opt[:, k] - x_goal)
    J += (u_opt[:, k] - u_equilibrium).T @ cfg.R @ (u_opt[:, k] - u_equilibrium)
# terminal cost
J += (x_opt[:, -1] - x_goal).T @ cfg.Q_e @ (x_opt[:, -1] - x_goal)

# Build NLP
ocp = {
    'x': ca.veccat(x_opt, u_opt),
    'p': ca.vertcat(x_0_param, goal_param),
    'g': ca.vertcat(*g),
    'f': J
}
solver = ca.nlpsol('solver', 'ipopt', ocp)

# Initial guesses
x0_guess = np.tile(x_init.flatten()[:, np.newaxis], (1, N+1))
u0_guess = np.zeros((nu, N))

# Solve open-loop OCP
solution = solver(
    x0=ca.veccat(x0_guess, u0_guess),
    p=ca.vertcat(x_init.flatten(), goal.flatten()),
    lbg=ca.vertcat(*lbg),
    ubg=ca.vertcat(*ubg)
)

sol_vec = solution['x'].full().flatten()
x_sol = sol_vec[:(N+1) * nx].reshape((nx, N+1), order='F')
u_sol = sol_vec[(N+1) * nx:].reshape((nu, N), order='F')

print(f"Optimal cost: {solution['f'].full().flatten()[0]}")

# plot states and controls (returns fig, axs). Caller may call plt.show() if desired.
fig, axs = model.plotSimulation(x_sol, u_sol)
# plt.show()
additional_lines_or_scatters = {"Goal": {"type": "scatter", "data": [[goal[0]], [goal[1]]], "color": "tab:orange", "s": 100, "marker":"x"}}
    # To save the animation as GIF, uncomment these lines:
    # save_gif_path = os.path.join(local_path, "drone.gif")
    # model.animateSimulation(x_sol, u_sol, additional_lines_or_scatters=additional_lines_or_scatters, save_path=save_gif_path)
model.animateSimulation(x_sol, u_sol, additional_lines_or_scatters=additional_lines_or_scatters)