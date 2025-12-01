import casadi as ca
from dataclasses import dataclass, field
import numpy as np
import os
import sys

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(local_path, ".."))

import models.omniBotXYModel as omniBotXYModel
import matplotlib.pyplot as plt


@dataclass
class RefTrackingCtrlConfig:
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.01, 0.01]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01]))
    n_hrzn = 50
    num_agents = 2


sampling_time = 0.05
num_agents = 2
cfg = RefTrackingCtrlConfig()

# scenario values (from original open_loop)
p1_target_val = np.array([3.5, 2.5])
x1_init_val = np.array([0.0, 0.0, 0.0, 0.0])
p2_target_val = np.array([0., 2.5])
x2_init_val = np.array([2.5, 0.7, 0.0, 0.0])

model = omniBotXYModel.OmniBotXYModel(sampling_time)
nx = model.model_config.nx
nu = model.model_config.nu
n_hrzn = cfg.n_hrzn

# Decision variables
x_opt = ca.MX.sym('x', nx * num_agents, n_hrzn + 1)
x_ref = ca.MX.sym('x_ref', nx * num_agents, n_hrzn + 1)
u_opt = ca.MX.sym('u', nu * num_agents, n_hrzn)

# Parameters
x_0_param = ca.MX.sym('x_0', nx * num_agents)

# Constraints and objective
g = []
lbg = []
ubg = []

# system dynamics: initial equality
g.append(x_opt[:nx * num_agents, 0] - x_0_param)
lbg.append(np.zeros(nx * num_agents,))
ubg.append(np.zeros(nx * num_agents,))

# dynamics per agent and time
for i_agent in range(num_agents):
    for k in range(n_hrzn):
        x_slice = x_opt[i_agent*nx:(i_agent+1)*nx, k]
        u_slice = u_opt[i_agent*nu:(i_agent+1)*nu, k]
        g.append(x_opt[i_agent*nx:(i_agent+1)*nx, k+1] - model.f_disc(x_slice, u_slice))
        lbg.append(np.zeros(nx,))
        ubg.append(np.zeros(nx,))

# Collision Avoidance Constraints
for i_agent in range(num_agents):
    for j_agent in range(i_agent+1, num_agents):
        for k in range(1, n_hrzn+1):
            g.append(ca.sumsqr(x_opt[i_agent*nx:i_agent*nx+2, k] - x_opt[j_agent*nx:j_agent*nx+2, k]))
            lbg.append((model.model_config.safety_radius * 2)**2)
            ubg.append(ca.inf)

# Objective
J = 0.0
for i_agent in range(num_agents):
    for k in range(n_hrzn):
        x_dev = x_opt[i_agent*nx:(i_agent+1)*nx, k] - x_ref[i_agent*nx:(i_agent+1)*nx, k]
        J += x_dev.T @ cfg.Q @ x_dev
        u = u_opt[i_agent*nu:(i_agent+1)*nu, k]
        J += u.T @ cfg.R @ u
    # terminal cost
    x_dev = x_opt[i_agent*nx:(i_agent+1)*nx, -1] - x_ref[i_agent*nx:(i_agent+1)*nx, -1]
    J += x_dev.T @ cfg.Q @ x_dev

# Build NLP
ocp = {
    'x': ca.veccat(x_opt, u_opt),
    'p': ca.veccat(x_0_param, x_ref),
    'g': ca.vertcat(*g),
    'f': J
}
solver = ca.nlpsol('solver', 'ipopt', ocp)

# build reference trajectories (same as original open_loop)
p_lin = np.linspace(0.0, 1.0, n_hrzn+1, endpoint=True)[np.newaxis, :]
p1_ref_val = (p1_target_val[:, np.newaxis] - x1_init_val[:2, np.newaxis]) @ p_lin + x1_init_val[:2, np.newaxis]
v1_ref_val = np.diff(p1_ref_val, axis=1) / sampling_time
v1_ref_val = np.hstack((v1_ref_val, np.zeros((2, 1))))

p2_ref_val = (p2_target_val[:, np.newaxis] - x2_init_val[:2, np.newaxis]) @ p_lin + x2_init_val[:2, np.newaxis]
v2_ref_val = np.diff(p2_ref_val, axis=1) / sampling_time
v2_ref_val = np.hstack((v2_ref_val, np.zeros((2, 1))))

x_ref_val = np.vstack((p1_ref_val, v1_ref_val, p2_ref_val, v2_ref_val))

# Initial conditions and parameter vector
x0_val = np.hstack((x1_init_val, x2_init_val)).flatten()
p_val = np.hstack((x0_val, x_ref_val.flatten(order='F')))

# initial guesses
x0_guess = np.tile(x0_val[:, np.newaxis], (1, n_hrzn+1))
u0_guess = np.zeros((nu * num_agents, n_hrzn))

# Solve
sol = solver(
    x0=ca.veccat(x0_guess, u0_guess),
    p=p_val,
    lbg=ca.vertcat(*lbg),
    ubg=ca.vertcat(*ubg)
)

sol_vec = sol['x'].full().flatten()
x_sol = sol_vec[:(n_hrzn+1) * nx * num_agents].reshape((nx * num_agents, n_hrzn+1), order='F')
u_sol = sol_vec[(n_hrzn+1) * nx * num_agents:].reshape((nu * num_agents, n_hrzn), order='F')

additional_lines_or_scatters = {"Ref1": {"type": "line", "data": [p1_ref_val[0, :], p1_ref_val[1, :]], "color": "tab:orange", "s": 100, "marker":"x"},
                                "Ref2": {"type": "line", "data": [p2_ref_val[0, :], p2_ref_val[1, :]], "color": "tab:orange", "s": 100, "marker":"x"}}
# plot states and controls
fig, axs = model.plotSimulation(x_sol, u_sol, num_agents=num_agents)
# plt.show()
model.animateSimulation(x_sol, u_sol, num_agents=num_agents, additional_lines_or_scatters=additional_lines_or_scatters)