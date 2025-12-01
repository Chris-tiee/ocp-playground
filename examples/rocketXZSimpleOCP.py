import casadi as ca
from dataclasses import dataclass, field
import numpy as np
import math
import os
import sys

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(local_path, ".."))

import models.rocketXZModel as rocketXZModel


@dataclass
class RocketOCPConfig:
    n_hrzn: int = 50
    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 1.0, 1.0, 50.0, 1.0]))
    Q_e: np.ndarray = field(default_factory=lambda: np.diag([50.0, 50.0, 1.0, 1.0, 100.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01]))
    max_T: float = 15
    min_T: float = 0
    max_delta: float = 0.5


sampling_time = 0.05
cfg = RocketOCPConfig()

model = rocketXZModel.RocketXZModel(sampling_time)
nx = model.model_config.nx
nu = model.model_config.nu
N = cfg.n_hrzn

# initial state: px, pz, vx, vz, pitch, vpitch
x_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# goal: some forward and up position with nose up
goal = np.array([3.0, 3.0, 0.0, 0.0, 0.0, 0.0])

# Decision variables
x_opt = ca.MX.sym('x', nx, N+1)
u_opt = ca.MX.sym('u', nu, N)

# Parameters
x_0_param = ca.MX.sym('x_0', nx)
goal_param = ca.MX.sym('goal', nx)

# Constraints and objective
g = []
lbg = []
ubg = []

# input constraints (simple bound enforcement via inequality constraints)
for k in range(N):
    g.append(u_opt[0, k])
    lbg.append(cfg.min_T)
    ubg.append(cfg.max_T)
    g.append(u_opt[1, k])
    lbg.append(-cfg.max_delta)
    ubg.append(cfg.max_delta)

# initial condition equality
g.append(x_opt[:, 0] - x_0_param)
lbg.append(np.zeros(nx,))
ubg.append(np.zeros(nx,))

# dynamics equality constraints
for k in range(N):
    x_next = model.f_disc(x_opt[:, k], u_opt[:, k])
    g.append(x_opt[:, k+1] - x_next)
    lbg.append(np.zeros(nx,))
    ubg.append(np.zeros(nx,))

# objective
J = 0.0
for k in range(N):
    x_err = x_opt[:, k] - goal_param
    J += ca.mtimes([x_err.T, cfg.Q, x_err])
    u = u_opt[:, k]
    J += ca.mtimes([u.T, cfg.R, u])
# terminal cost
x_err = x_opt[:, -1] - goal_param
J += ca.mtimes([x_err.T, cfg.Q_e, x_err])

# Build NLP
ocp = {
    'x': ca.veccat(x_opt, u_opt),
    'p': ca.vertcat(x_0_param, goal_param),
    'g': ca.vertcat(*g),
    'f': J
}

# Create solver without IPOPT options
solver = ca.nlpsol('solver', 'ipopt', ocp)

# Initial guesses
x0_guess = np.tile(x_init.flatten()[:, np.newaxis], (1, N+1))
u0_guess = np.zeros((nu, N))

# Solve
solution = solver(
    x0=ca.veccat(x0_guess, u0_guess),
    p=ca.vertcat(x_init.flatten(), goal.flatten()),
    lbg=ca.vertcat(*lbg),
    ubg=ca.vertcat(*ubg)
)

sol_vec = solution['x'].full().flatten()
x_len = (N+1) * nx
x_sol = sol_vec[:x_len].reshape((nx, N+1), order='F')
u_sol = sol_vec[x_len:].reshape((nu, N), order='F')

print(f"Optimal cost: {solution['f'].full().flatten()[0]}")

additional_lines_or_scatters = {"Goal": {"type": "scatter", "data": [[goal[0]], [goal[1]]], "color": "tab:orange", "s": 100, "marker":"x"}}
model.animateSimulation(x_sol, u_sol, additional_lines_or_scatters=additional_lines_or_scatters)
