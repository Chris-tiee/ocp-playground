import casadi as ca
from dataclasses import dataclass, field
import numpy as np
import os
import sys

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(local_path, ".."))

import models.droneXZModel as droneXZModel
import matplotlib.pyplot as plt

#basic configurations
@dataclass
class GoalReachingCtrlConfig:
    max_fl: float = 15.0
    max_fr: float = 15.0

    # Weight matrix for state
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.1, 0.1, 1.0, 0.1]))
    # Weight matrix for terminal state 
    # Q_e: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 1.0, 1.0, 10.0, 1.0]))
    Q_e: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.1, 0.1, 1.0, 0.1]))
    # Weight matrix for input control
    R: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01]))
    #  horizon length
    n_hrzn = 100

# --- Linear/script style open-loop OCP (no MPC) ---
sampling_time = 0.05
cfg = GoalReachingCtrlConfig()
# increase horizon to 30 as requested

# initial and goal (from original main)
x_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
goal = np.array([1.0, 1.0])

model = droneXZModel.DroneXZModel(sampling_time)
nx = model.model_config.nx #state size
nu = model.model_config.nu #control size (left/right)
N = cfg.n_hrzn # horizon

# Decision variables
x_opt = ca.MX.sym('x', nx, N+1) # [p_x, p_z, v_x, v_z, phi, phi_dot]
u_opt = ca.MX.sym('u', nu, N) # [ul, ur] for N steps

# Parameters (predefined known values)
x_0_param = ca.MX.sym('x_0', nx)
goal_param = ca.MX.sym('goal', 2)

# Defining the path to follow:
def build_reference_trajectory(N, nx, goal):
    # Defining the path to follow: figure-8 around the goal position
    x_ref = np.zeros((N+1, nx))

    # Phase length fractions
    frac1 = 0.3   # 30% of horizon: departure from origin
    frac2 = 0.5   # 50% of horizon: circular loop
    # remaining 20%: go to goal and stay there

    N1 = max(1, int(frac1 * N))
    N2 = max(2, int(frac2 * N))
    N3 = N - N1 - N2
    if N3 < 1:
        N3 = 1
        N2 = N - N1 - N3

    # ---- Phase 1: straight line from (0,0) to (2.25, 1.5) ----
    px_start = 0.0
    pz_start = 0.0
    px_phase1_end = 2.25
    pz_phase1_end = 1.5

    for k in range(N1 + 1):
        alpha = k / max(1, N1)
        px_ref = (1 - alpha) * px_start + alpha * px_phase1_end
        pz_ref = (1 - alpha) * pz_start + alpha * pz_phase1_end

        x_ref[k, 0] = px_ref
        x_ref[k, 1] = pz_ref
        x_ref[k, 2:] = 0.0   # v_x, v_z, pitch, vpitch = 0

    # ---- Phase 2: circular loop in positive quadrant (bigger) ----
    center_x = 1.5
    center_z = 1.5
    radius = 0.75

    for i in range(N2 + 1):
        k = N1 + i
        if k > N:
            break
        theta = 2.0 * np.pi * i / max(1, N2)   # one full loop
        px_ref = center_x + radius * np.cos(theta)
        pz_ref = center_z + radius * np.sin(theta)

        x_ref[k, 0] = px_ref
        x_ref[k, 1] = pz_ref
        x_ref[k, 2:] = 0.0

    # index & position where Phase 2 ends
    last_phase2_index = min(N1 + N2, N)
    px_last = x_ref[last_phase2_index, 0]
    pz_last = x_ref[last_phase2_index, 1]

    # ---- Phase 3: go to (3,3) and stay there ----
    for i in range(1, N3 + 1):
        k = last_phase2_index + i
        if k > N:
            break
        alpha = i / max(1, N3)
        px_ref = (1 - alpha) * px_last + alpha * goal[0]
        pz_ref = (1 - alpha) * pz_last + alpha * goal[1]

        x_ref[k, 0] = px_ref
        x_ref[k, 1] = pz_ref
        x_ref[k, 2:] = 0.0

    # fill any remaining steps with exact goal
    for k in range(last_phase2_index + N3 + 1, N+1):
        x_ref[k, 0] = goal[0]
        x_ref[k, 1] = goal[1]
        x_ref[k, 2:] = 0.0

    return x_ref

def build_trajectory_easy(N, nx, goal):
    x_ref = np.zeros((N+1,nx))
    N1 = max(1, int(0.9 * N))
    N2 = N-N1
    A = 0.5 #Wave Height
    for k in range(N1+1):
        alpha = k/N1
        x_ref[k,1] = alpha * goal[1] + 0.5 * np.sin(np.pi*alpha)
        x_ref[k,0] = alpha * goal[0]
        x_ref[k, 2:] = 0.0   # v_x, v_z, pitch, vpitch = 0
    
    for i in range(1, N2+1):
        k = N1 + i
        if k > N:
            break
        x_ref[k,1] = goal[1]
        x_ref[k,0] = goal[0]
        x_ref[k, 2:] = 0.0   # v_x, v_z, pitch, vpitch = 0
    
    return x_ref

# x_ref = build_reference_trajectory(N, nx, goal)
x_ref = build_trajectory_easy(N, nx, goal)
# convert to CasADi matrix with shape (nx, N+1)
x_ref_DM = ca.DM(x_ref.T)   # now column k is x_ref at step k

# Constraints and objective
g = []
lbg = []
ubg = []

# Input constraints
# Limited by thrusters power
for k in range(N):
    g.append(u_opt[0, k])
    lbg.append(0.0)
    ubg.append(cfg.max_fl)
    g.append(u_opt[1, k])
    lbg.append(0.0)
    ubg.append(cfg.max_fr)

# initial condition equality
# Start at initial condition
g.append(x_opt[:, 0] - x_0_param)
lbg.append(np.zeros(nx,))
ubg.append(np.zeros(nx,))

# dynamics equality
for k in range(N):
    g.append(x_opt[:, k+1] - model.f_disc(x_opt[:, k], u_opt[:, k]))
    lbg.append(np.zeros(nx,))
    ubg.append(np.zeros(nx,))

# hardcoded obstacle
obs_c = np.array([x_ref[(N-2*(N//3)),0], x_ref[(N-2*(N//3)),1]])
obs_R = 0.09
for k in range(N+1):
    px_k = x_opt[0,k]
    pz_k = x_opt[1,k]
    h_k = obs_R**2 - ((px_k - obs_c[0])**2 + (pz_k - obs_c[1])**2)

    g.append(h_k)
    lbg.append(-ca.inf)
    ubg.append(0.0)
#to draw it
theta = np.linspace(0, 2 * np.pi, 80)
obs_x = obs_c[0] + obs_R * np.cos(theta)
obs_z = obs_c[1] + obs_R * np.sin(theta)

# objective
J = 0.0
u_equilibrium = 0.5 * model.model_config.mass * model.model_config.gravity * ca.DM.ones(2, 1)
for k in range(N):
    J += (x_opt[:, k] - x_ref_DM[:, k]).T @ cfg.Q @ (x_opt[:, k] - x_ref_DM[:, k])  #state
# terminal cost E(x_N)
J += (u_opt[:, N-1] - u_equilibrium).T @ cfg.R @ (u_opt[:, N-1] - u_equilibrium)    #control
J += (x_opt[:, -1] - x_ref_DM[:, -1]).T @ cfg.Q_e @ (x_opt[:, -1] - x_ref_DM[:, -1])

# Build NLP
ocp = {
    'x': ca.veccat(x_opt, u_opt), #decision variable, unknowns to be optimized
    'p': x_0_param, #parameters, known during optimization
    #solver receives them as external inputs
    'g': ca.vertcat(*g),
    'f': J #objective function, cost to minimize (=L+E terminal cost)
}
solver = ca.nlpsol('solver', 'ipopt', ocp) #minimize f(x,p) 

# Initial guesses
x0_guess = np.tile(x_init.flatten()[:, np.newaxis], (1, N+1))
u0_guess = np.zeros((nu, N))

# Solve open-loop OCP
solution = solver(
    x0=ca.veccat(x0_guess, u0_guess),
    p=x_init.flatten(),
    lbg=ca.vertcat(*lbg),
    ubg=ca.vertcat(*ubg)
)

sol_vec = solution['x'].full().flatten()
x_sol = sol_vec[:(N+1) * nx].reshape((nx, N+1), order='F')
u_sol = sol_vec[(N+1) * nx:].reshape((nu, N), order='F')

print(f"Optimal cost: {solution['f'].full().flatten()[0]}")

# plot states and controls (returns fig, axs). Caller may call plt.show() if desired.
# fig, axs = model.plotSimulation(x_sol, u_sol)
additional_lines_or_scatters = {
    "Reference path": {
        "type": "line",
        "data": [x_ref[:, 0], x_ref[:, 1]],   # px_ref vs pz_ref
        "color": "tab:red"
    },
    "Goal": {
        "type": "scatter",
        "data": [[goal[0]], [goal[1]]],
        "color": "tab:orange",
        "s": 100,
        "marker": "x"
    },
    "Obstacle": {
        "type": "line",
        "data": [obs_x,obs_z],
        "color": "tab:purple"
    }
}

# additional_lines_or_scatters = {
#     "Reference path": {
#         "type": "line",
#         "data": [x_ref[:, 0], x_ref[:, 1]],   # px_ref vs pz_ref
#         "color": "tab:red"
#     },
#     "Goal": {
#         "type": "scatter",
#         "data": [[goal[0]], [goal[1]]],
#         "color": "tab:orange",
#         "s": 100,
#         "marker": "x"
#     }
# }

# To save the animation as GIF, uncomment these lines:
# save_gif_path = os.path.join(local_path, "drone.gif")
# model.animateSimulation(x_sol, u_sol, additional_lines_or_scatters=additional_lines_or_scatters, save_path=save_gif_path)
model.animateSimulation(x_sol, u_sol, additional_lines_or_scatters=additional_lines_or_scatters)


# plt.show()