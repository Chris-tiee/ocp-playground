import casadi as ca
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
import os
import sys


@dataclass
class OCPConfig:
    nx: int = 4
    nu: int = 2
    n_hrzn: int = 50
    sampling_time = 0.05
    Q = np.diag([1.0, 1.0, 0.01, 0.01])
    R = np.diag([1, 1])


ocp_config = OCPConfig()

# %% Define the OCP

# Parameter values
radius_obj = 0.3
radius_obs_1 = 0.35
radius_obs_2 = 0.8
lb_p = -1.0
ub_p = 2.0
x_init = ca.DM([-0.5, -0.5, 0.0, 0.0])
x_target = ca.DM([1.5, 1.5, 0.0, 0.0])
p_obs_1= np.array([1.5, 0.0])
p_obs_2 = np.array([0.0, 1.0])

# optimization variables
X_shooting = ca.SX.sym('x', ocp_config.nx, ocp_config.n_hrzn + 1)
U_shooting = ca.SX.sym('u', ocp_config.nu, ocp_config.n_hrzn)

# objective function
J = 0.
for i in range(ocp_config.n_hrzn):
    J += U_shooting[:, i].T @ ocp_config.R @ U_shooting[:, i]

# constraints
g = []
lbg, ubg = [], []

for i in range(ocp_config.n_hrzn+1):
    # obstacle 1 avoidance constraints
    g.append(ca.sumsqr(X_shooting[0:2, i] - p_obs_1))
    lbg.append((radius_obj + radius_obs_1)**2)
    ubg.append(ca.inf)

    # obstacle 2 avoidance constraints
    g.append(ca.sumsqr(X_shooting[0:2, i] - p_obs_2))
    lbg.append((radius_obj + radius_obs_2)**2)
    ubg.append(ca.inf)

# initial and final constraints
g.append(X_shooting[:, 0] - x_init)
lbg.append(ca.DM.zeros(ocp_config.nx))
ubg.append(ca.DM.zeros(ocp_config.nx))

g.append(X_shooting[:, ocp_config.n_hrzn] - x_target)
lbg.append(ca.DM.zeros(ocp_config.nx))
ubg.append(ca.DM.zeros(ocp_config.nx))

# discretized dynamics
for i in range(ocp_config.n_hrzn):
    g.append(X_shooting[0:2, i + 1] - (X_shooting[0:2, i] + ocp_config.sampling_time * X_shooting[2:4, i] + 0.5 * ocp_config.sampling_time ** 2 * U_shooting[:, i]))
    g.append(X_shooting[2:4, i + 1] - (X_shooting[2:4, i] + ocp_config.sampling_time * U_shooting[:, i]))
    lbg.append(ca.DM.zeros(ocp_config.nx))
    ubg.append(ca.DM.zeros(ocp_config.nx))

ocp = { 'x': ca.veccat(X_shooting, U_shooting),
        'g': ca.veccat(*g),
        'f': J
    }
opts = {'ipopt': {'print_level': 5, 'max_iter': 1000}, "print_time": False}
solver = ca.nlpsol('solver', 'ipopt', ocp, opts)


x_init = np.array([-0.5, -0.5, 1.0, 1.0])
solution = solver(
        x0=ca.veccat(np.tile(x_init, (1, ocp_config.n_hrzn+1)), np.zeros((ocp_config.nu, ocp_config.n_hrzn))),
        lbg=ca.vertcat(*lbg),
        ubg=ca.vertcat(*ubg),
    )
sol = solution['x'].full().flatten()
print(f"The optimal cost is: {solution['f'].full().flatten()}")
x_sol = sol[:(ocp_config.n_hrzn+1) * ocp_config.nx].reshape((ocp_config.nx, ocp_config.n_hrzn+1), order='F')


# %% Make nice plots
local_path = os.path.dirname(os.path.realpath(__file__))
images_folder = os.path.join(local_path, "ocp_images")
if not(os.path.exists(images_folder) and os.path.isdir(images_folder)):
    os.makedirs(images_folder)

fig = plt.figure(figsize=(8,8))
for i in range(ocp_config.n_hrzn + 1):
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-1.1, 2.1)
    ax.set_ylim(-1.1, 2.1)
    # ax.hlines([lb_x, ub_x], lb_x, ub_x, color='tab:brown', linestyle='--', linewidth=2)
    # ax.vlines([lb_x, ub_x], lb_x, ub_x, color='tab:brown', linestyle='--', linewidth=2)
    ax.set_xlabel('$p_x$ (m)')
    ax.set_ylabel('$p_y$ (m)')
    ax.set_aspect('equal')
    ax.add_patch(patches.Circle((p_obs_1[0], p_obs_1[1]), radius_obs_1, linewidth=0.0, color='tab:orange', alpha=0.5))
    ax.add_patch(patches.Circle((p_obs_2[0], p_obs_2[1]), radius_obs_2, linewidth=0.0, color='tab:orange', alpha=0.5))
    ax.add_patch(patches.Circle((x_sol[0, i], x_sol[1, i]), radius_obj, linewidth=0.0, color='tab:blue'))
    ax.scatter(p_obs_1[0], p_obs_1[1], color='tab:orange', s=50, label='Obstacle 1')
    ax.scatter(p_obs_2[0], p_obs_2[1], color='tab:orange', s=50, label='Obstacle 2')
    ax.scatter(x_sol[0, i], x_sol[1, i], color='black', s=50, label="OBJECT", zorder=10)
    ax.plot(x_target[0, :], x_target[1, :], color='tab:gray', linewidth=3, label='Ref. traj.')
    # fig.subplots_adjust(right=0.6)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.pause(0.05)
    # plt.savefig(os.path.join(images_folder, "step_{:02d}.png".format(i)), dpi=300)
    # plt.close()
plt.show()


## TO Generate Video:
## $ ffmpeg -framerate 10 -i step_%02d.png -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4