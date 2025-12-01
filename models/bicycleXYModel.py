import casadi as ca
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np



"""
Consider the kinematic model only:
x = (px, py, yaw)
a = (delta, V)
"""


@dataclass
class BicycleXYConfig:
    nx: int = 3
    nu: int = 2
    lf: float = 0.5
    lr: float = 0.5
    safety_radius: float = 0.8

class BicycleXYModel:
    def __init__(self, sampling_time):
        self._sampling_time = sampling_time
        self.model_name = "BicycleXYModel"
        self.model_config = BicycleXYConfig()

        x = ca.MX.sym('x', self.model_config.nx)
        u = ca.MX.sym('u', self.model_config.nu)

        beta = ca.arctan( self.model_config.lr * ca.tan(u[0]) / (self.model_config.lr + self.model_config.lf) )

        x_dot = ca.vertcat(
            u[1] * ca.cos(x[2] + beta),  # \dot{px}
            u[1] * ca.sin(x[2] + beta),  # \dot{py}
            u[1] * ca.sin(beta) / self.model_config.lr,  # \dot{yaw}
        )
        
        # set up integrator for discrete dynamics
        # multiply xdot by sampling time (time scaling), since casadi integrator integrates over [0,1] by default
        dae = {'x': x, 'p': u, 'ode': self._sampling_time * x_dot}
        self.I = ca.integrator('I', 'rk', dae)

        # create continuous and discrete dynamics
        self.f_cont = ca.Function('f_cont', [x, u], [x_dot])
        self.f_disc = ca.Function('f_disc', [x, u], [self.I(x0=x, p=u)['xf']])


    def animateSimulation(self, x_trajectory, u_trajectory, num_agents:int=1, additional_lines_or_scatters=None):
        wheel_long_axis = 0.4
        wheel_short_axis = 0.1

        sim_length = u_trajectory.shape[1]
        _, ax = plt.subplots()

        nx = self.model_config.nx
        nu = self.model_config.nu

        for i in range(sim_length+1):
            ax.set_aspect('equal')
            ax.set_xlim(-1, 5)
            ax.set_ylim(-1, 5)
            ax.set_xlabel(r'$p_{\mathrm{x}}$ in m', fontsize=14)
            ax.set_ylabel(r'$p_{\mathrm{y}}$ in m', fontsize=14)

            for i_agent in range(num_agents):
                front_x = x_trajectory[i_agent*nx, i] + self.model_config.lf * ca.cos(x_trajectory[i_agent*nx+2, i])
                front_y = x_trajectory[i_agent*nx+1, i] + self.model_config.lf * ca.sin(x_trajectory[i_agent*nx+2, i])
                rear_x = x_trajectory[i_agent*nx, i] - self.model_config.lr * ca.cos(x_trajectory[i_agent*nx+2, i])
                rear_y = x_trajectory[i_agent*nx+1, i] - self.model_config.lr * ca.sin(x_trajectory[i_agent*nx+2, i])
                ax.plot(x_trajectory[i_agent*nx, :i+1], x_trajectory[i_agent*nx+1, :i+1], color="tab:gray", linewidth=2, zorder=0)
                ax.scatter(x_trajectory[i_agent*nx, i], x_trajectory[i_agent*nx+1, i], color="tab:gray", s=50, zorder=2)
                ax.plot([front_x, rear_x], [front_y, rear_y], color="tab:blue", linewidth=3, zorder=1)
                if i < sim_length:
                    wheel_f = patches.Ellipse((front_x, front_y), wheel_long_axis, wheel_short_axis, angle=math.degrees(x_trajectory[i_agent*nx+2, i] + u_trajectory[0, i]), color="tab:green", label="Wheels" if i_agent == 0 else None)
                    wheel_r = patches.Ellipse((rear_x, rear_y), wheel_long_axis, wheel_short_axis, angle=math.degrees(x_trajectory[i_agent*nx+2, i]), color="tab:green")
                    ax.add_patch(wheel_f)
                    ax.add_patch(wheel_r)
                safety_circle = patches.Circle((x_trajectory[i_agent*nx, i], x_trajectory[i_agent*nx+1, i]), self.model_config.safety_radius, color="tab:orange", alpha=0.3)
                ax.add_patch(safety_circle)
            if additional_lines_or_scatters is not None:
                for key, value in additional_lines_or_scatters.items():
                    if value["type"] == "scatter":
                        ax.scatter(value["data"][0], value["data"][1], color=value["color"], s=value["s"], label=key, marker=value["marker"])
                    elif value["type"] == "line":
                        ax.plot(value["data"][0], value["data"][1], color=value["color"], linewidth=2, label=key)
            ax.set_title(f"Bicycle Simulation: Step {i+1}")
            ax.legend()
            if i < sim_length:
                plt.show(block=False)
                plt.pause(0.2)
            else:
                plt.show(block=True)
            ax.clear()
        return
    
    def plotSimulation(self, x_trajectory: np.ndarray, u_trajectory: np.ndarray, num_agents:int=1, figsize=(8, 10)):
        """Plot positions, yaw and controls for one or multiple bicycle agents.

        - Positions: px, py (one trace per agent)
        - Yaw: yaw (one trace per agent)
        - Controls: steering delta and speed V (plotted as step signals per agent)
        """
        dt = self._sampling_time
        nx = self.model_config.nx
        nu = self.model_config.nu
        sim_length = u_trajectory.shape[1]

        t_x = np.arange(sim_length + 1) * dt

        # Create 4 rows x num_agents columns so each agent has its own column
        fig, axes = plt.subplots(4, num_agents, figsize=(figsize[0] * max(1, num_agents), figsize[1]), constrained_layout=True)
        # Ensure axes is 2D array with shape (4, num_agents)
        axes = np.atleast_2d(axes)
        if axes.shape[0] != 4:
            axes = axes.reshape(4, num_agents)

        # Positions: each agent in its own column
        for i_agent in range(num_agents):
            ax = axes[0, i_agent]
            ax.plot(t_x, x_trajectory[i_agent*nx, :], label=r'$p_x$')
            ax.plot(t_x, x_trajectory[i_agent*nx+1, :], label=r'$p_y$')
            ax.set_ylabel(r'position $p$ in m')
            ax.set_title(f'Agent {i_agent}')
            ax.legend()

        # Yaw per agent (single trace per subplot -> no legend)
        for i_agent in range(num_agents):
            ax = axes[1, i_agent]
            ax.plot(t_x, x_trajectory[i_agent*nx+2, :])
            ax.set_ylabel(r'yaw $\theta$ in rad')

        # Controls: steering delta (step) per agent
        for i_agent in range(num_agents):
            ax = axes[2, i_agent]
            delta = u_trajectory[i_agent*nu, :]
            delta_plot = np.concatenate((np.asarray(delta).flatten(), [np.nan]))
            ax.step(t_x, delta_plot, where='post')
            ax.set_ylabel(r'steering $\delta$ in rad')

        # Controls: speed V per agent
        for i_agent in range(num_agents):
            ax = axes[3, i_agent]
            V = u_trajectory[i_agent*nu+1, :]
            V_plot = np.concatenate((np.asarray(V).flatten(), [np.nan]))
            ax.step(t_x, V_plot, where='post')
            ax.set_ylabel(r'speed $V$ in m/s')
            ax.set_xlabel('time in s')

        # note: additional reference lines/scatters removed from plotSimulation; use animateSimulation to show markers in animation

        fig.suptitle('Bicycle states and controls')

        x_min = t_x[0]
        x_max = t_x[-1]
        # set x-limits for all axes
        for row in range(4):
            for col in range(num_agents):
                axes[row, col].set_xlim(x_min, x_max)

        return fig, axes
    