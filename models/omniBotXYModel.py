import casadi as ca
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np

# BaseModel removed: models should initialize sampling time directly


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
        # initialize sampling time directly (previously handled by BaseModel)
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
            ax.set_xlabel(r'$p_{\mathrm{x}}$ (m)', fontsize=14)
            ax.set_ylabel(r'$p_{\mathrm{y}}$ (m)', fontsize=14)

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