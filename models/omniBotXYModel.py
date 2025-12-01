import casadi as ca
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import matplotlib.animation as animation
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

    def animateSimulation(self, x_trajectory, u_trajectory, num_agents: int = 1, additional_lines_or_scatters=None, save_path: str = None):

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
        ax.set_aspect('equal')
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_xlabel(r'$p_{\mathrm{x}}$ in m', fontsize=14)
        ax.set_ylabel(r'$p_{\mathrm{y}}$ in m', fontsize=14)

        nx = self.model_config.nx
        nu = self.model_config.nu

        path_lines = []
        center_scatters = []
        safety_circles = []
        # arrows will be created/removed each frame to match drone style
        arrows = [None] * num_agents

        for a in range(num_agents):
            ln, = ax.plot([], [], color="tab:gray", linewidth=2, zorder=0)
            sc = ax.scatter([], [], color="tab:gray", s=50, zorder=2)
            scircle = patches.Circle((0, 0), self.model_config.safety_radius, color="tab:orange", alpha=0.3)
            ax.add_patch(scircle)
            path_lines.append(ln)
            center_scatters.append(sc)
            safety_circles.append(scircle)

        added_artists = []
        if additional_lines_or_scatters is not None:
            for key, value in additional_lines_or_scatters.items():
                if value["type"] == "scatter":
                    sc = ax.scatter(value["data"][0], value["data"][1], color=value.get("color", "k"), s=value.get("s", 20), label=key, marker=value.get("marker", ","))
                    added_artists.append(sc)
                elif value["type"] == "line":
                    ln, = ax.plot(value["data"][0], value["data"][1], color=value.get("color", "k"), linewidth=2, label=key)
                    added_artists.append(ln)

        # do not show legend for references by default
        fig.subplots_adjust(bottom=0.15)

        interval = max(50, int(self._sampling_time * 1000 * 2.0))

        def init():
            for ln in path_lines:
                ln.set_data([], [])
            for sc in center_scatters:
                sc.set_offsets(np.empty((0, 2)))
            for sc in safety_circles:
                sc.set_center((0, 0))
            return path_lines + center_scatters + safety_circles + added_artists

        def update(i):
            # scale down velocity arrows so they remain inside plot
            vel_scale = 0.2
            for a in range(num_agents):
                base = a * nx
                xs = x_trajectory[base, :i+1]
                ys = x_trajectory[base+1, :i+1]
                path_lines[a].set_data(xs, ys)
                cx = float(x_trajectory[base, i])
                cy = float(x_trajectory[base+1, i])
                center_scatters[a].set_offsets(np.array([[cx, cy]]))
                safety_circles[a].set_center((cx, cy))
                vx = float(x_trajectory[base+2, i])
                vy = float(x_trajectory[base+3, i])
                # remove previous arrow if present
                if arrows[a] is not None:
                    try:
                        arrows[a].remove()
                    except Exception:
                        pass
                    arrows[a] = None
                # create a small Arrow patch similar to drone thrust arrows
                dx = vx * vel_scale
                dy = vy * vel_scale
                arrows[a] = patches.Arrow(cx, cy, dx, dy, color="tab:blue", width=0.08)
                ax.add_patch(arrows[a])

            if additional_lines_or_scatters is not None:
                for key, value in additional_lines_or_scatters.items():
                    pass

            ax.set_title(f"OmniBot Simulation: Step {i+1}")
            return path_lines + center_scatters + safety_circles + added_artists

        anim = animation.FuncAnimation(fig, update, frames=range(sim_length + 1), init_func=init, interval=interval, blit=False, repeat=True)

        if save_path is not None:
            try:
                fps = max(1, int(round(1000.0 / float(interval))))
                print(f"Saving omnibot animation to {save_path} (fps={fps})")
                writer = animation.PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            except Exception as e:
                print(f"Failed to save omnibot animation to {save_path}: {e}")

        plt.show()
        return anim, fig

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