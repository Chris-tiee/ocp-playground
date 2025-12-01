import casadi as ca
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np

 

"""
x = (px, py, vx, vy)
a = (ax, ay)
"""

@dataclass
class OmniBotXYConfig:
    nx: int = 4
    nu: int = 2
    safety_radius: float = 0.8


class OmniBotXYModel:
    def __init__(self, sampling_time):
        self._sampling_time = sampling_time
        self.model_name = "OmniBotXYModel"
        self.model_config = OmniBotXYConfig()

        x = ca.MX.sym('x', self.model_config.nx)
        u = ca.MX.sym('u', self.model_config.nu)

        x_dot = ca.vertcat(
            x[2],  # \dot{px}
            x[3],  # \dot{py}
            u[0],  # \dot{vx}
            u[1],  # \dot{vy}
        )
        # set up integrator for discrete dynamics
        # multiply xdot by sampling time (time scaling), since casadi integrator integrates over [0,1] by default
        dae = {'x': x, 'p': u, 'ode': self._sampling_time * x_dot}
        self.I = ca.integrator('I', 'rk', dae)

                # create continuous and discrete dynamics
        self.f_cont = ca.Function('f_cont', [x, u], [x_dot])
        self.f_disc = ca.Function('f_disc', [x, u], [self.I(x0=x, p=u)['xf']])

    def animateSimulation(self, x_trajectory, u_trajectory, num_agents:int=1, additional_lines_or_scatters=None):

        fontsize = 16
        params = {
            'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{cmbright}",
            'axes.labelsize': fontsize,
            'axes.titlesize': fontsize,
            'legend.fontsize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            "mathtext.fontset": "stixsans",
            "axes.unicode_minus": False,
        }
        matplotlib.rcParams.update(params)

        sim_length = u_trajectory.shape[1]
        fig, ax = plt.subplots()

        nx = self.model_config.nx
        nu = self.model_config.nu

        for i in range(sim_length+1):
            ax.set_aspect('equal')
            ax.set_xlim(-1, 5)
            ax.set_ylim(-1, 5)
            ax.set_xlabel(r'$p_{\mathrm{x}}$ in m', fontsize=14)
            ax.set_ylabel(r'$p_{\mathrm{y}}$ in m', fontsize=14)

            for i_agent in range(num_agents):
                ax.scatter(x_trajectory[i_agent*nx, i], x_trajectory[i_agent*nx+1, i], color="tab:gray", s=50, zorder=2)
                safety_circle = patches.Circle((x_trajectory[i_agent*nx, i], x_trajectory[i_agent*nx+1, i]), self.model_config.safety_radius, color="tab:orange", alpha=0.3)
                ax.add_patch(safety_circle)
                vel_arrow = patches.Arrow(x_trajectory[i_agent*nx, i], x_trajectory[i_agent*nx+1, i], x_trajectory[i_agent*nx+2, i], x_trajectory[i_agent*nx+3, i], color="tab:blue")
                ax.add_patch(vel_arrow)
            if additional_lines_or_scatters is not None:
                for key, value in additional_lines_or_scatters.items():
                    if value["type"] == "scatter":
                        ax.scatter(value["data"][0], value["data"][1], color=value["color"], s=value["s"], label=key, marker=value["marker"])
                    elif value["type"] == "line":
                        ax.plot(value["data"][0], value["data"][1], color=value["color"], linewidth=2, label=key)
            ax.set_title(f"OmniBot Simulation: Step {i+1}")
            fig.subplots_adjust(bottom=0.15)
            if i < sim_length:
                plt.show(block=False)
                plt.pause(0.2)
            else:
                plt.show(block=True)
            ax.clear()
        return

    def plotSimulation(self, x_trajectory: np.ndarray, u_trajectory: np.ndarray, num_agents:int=1, figsize=(8,10)):
        """Plot positions, velocities and controls for OmniBot agents.

        - Positions px, py (one trace per agent)
        - Velocities vx, vy (one trace per agent)
        - Controls ax, ay (step signals per agent)
        """
        dt = self._sampling_time
        nx = self.model_config.nx
        nu = self.model_config.nu
        sim_length = u_trajectory.shape[1]

        t_x = np.arange(sim_length + 1) * dt

        # create 4 rows x num_agents columns so each agent gets its own column
        fig, axes = plt.subplots(4, num_agents, figsize=(figsize[0] * max(1, num_agents), figsize[1]), constrained_layout=True)
        axes = np.atleast_2d(axes)
        if axes.shape[0] != 4:
            axes = axes.reshape(4, num_agents)

        # Positions per agent
        for i_agent in range(num_agents):
            ax = axes[0, i_agent]
            ax.plot(t_x, x_trajectory[i_agent*nx, :], label=r'$p_x$')
            ax.plot(t_x, x_trajectory[i_agent*nx+1, :], label=r'$p_y$')
            ax.set_ylabel(r'position $p$ in m')
            ax.set_title(f'Agent {i_agent}')
            ax.legend()

        # Velocities per agent
        for i_agent in range(num_agents):
            ax = axes[1, i_agent]
            ax.plot(t_x, x_trajectory[i_agent*nx+2, :], label=r'$v_x$')
            ax.plot(t_x, x_trajectory[i_agent*nx+3, :], label=r'$v_y$')
            ax.set_ylabel(r'velocity $v$ in m/s')
            ax.legend()

        # Controls ax
        for i_agent in range(num_agents):
            ax = axes[2, i_agent]
            ax_plot = np.concatenate((np.asarray(u_trajectory[i_agent*nu, :]).flatten(), [np.nan]))
            ax.step(t_x, ax_plot, where='post')
            ax.set_ylabel(r'accel $a_x$ in m/s$^2$)')

        # Controls ay
        for i_agent in range(num_agents):
            ax = axes[3, i_agent]
            ay_plot = np.concatenate((np.asarray(u_trajectory[i_agent*nu+1, :]).flatten(), [np.nan]))
            ax.step(t_x, ay_plot, where='post')
            ax.set_ylabel(r'accel $a_y$ in m/s$^2$)')
            ax.set_xlabel('time (s)')

        # note: additional reference lines/scatters removed from plotSimulation; use animateSimulation to show markers in animation
        fig.suptitle('OmniBot states and controls')
        x_min = t_x[0]
        x_max = t_x[-1]
        for row in range(4):
            for col in range(num_agents):
                axes[row, col].set_xlim(x_min, x_max)

        return fig, axes