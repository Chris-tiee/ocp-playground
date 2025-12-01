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


    def animateSimulation(self, x_trajectory, u_trajectory, num_agents: int = 1, additional_lines_or_scatters=None, save_path: str = None):
        wheel_long_axis = 0.4
        wheel_short_axis = 0.1

        sim_length = u_trajectory.shape[1]
        nx = self.model_config.nx
        nu = self.model_config.nu

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_xlabel(r'$p_{\mathrm{x}}$ in m', fontsize=14)
        ax.set_ylabel(r'$p_{\mathrm{y}}$ in m', fontsize=14)

        # per-agent artists
        path_lines = []
        center_scatters = []
        body_lines = []
        wheel_front = []
        wheel_rear = []
        safety_circles = []

        for a in range(num_agents):
            ln, = ax.plot([], [], color="tab:gray", linewidth=2, zorder=0)
            sc = ax.scatter([], [], color="tab:gray", s=50, zorder=2)
            body_ln, = ax.plot([], [], color="tab:blue", linewidth=3, zorder=1)
            wf = patches.Ellipse((0, 0), wheel_long_axis, wheel_short_axis, angle=0.0, color="tab:green")
            wr = patches.Ellipse((0, 0), wheel_long_axis, wheel_short_axis, angle=0.0, color="tab:green")
            scircle = patches.Circle((0, 0), self.model_config.safety_radius, color="tab:orange", alpha=0.3)
            ax.add_patch(wf)
            ax.add_patch(wr)
            ax.add_patch(scircle)
            path_lines.append(ln)
            center_scatters.append(sc)
            body_lines.append(body_ln)
            wheel_front.append(wf)
            wheel_rear.append(wr)
            safety_circles.append(scircle)

        # additional static artists
        added_artists = []
        if additional_lines_or_scatters is not None:
            for key, value in additional_lines_or_scatters.items():
                if value["type"] == "scatter":
                    sc = ax.scatter(value["data"][0], value["data"][1], color=value.get("color", "k"), s=value.get("s", 20), label=key, marker=value.get("marker", ","))
                    added_artists.append(sc)
                elif value["type"] == "line":
                    ln, = ax.plot(value["data"][0], value["data"][1], color=value.get("color", "k"), linewidth=2, label=key)
                    added_artists.append(ln)

        if added_artists:
            ax.legend()
        fig.subplots_adjust(bottom=0.15)

        interval = max(50, int(self._sampling_time * 1000 * 2.0))

        def init():
            for ln in path_lines:
                ln.set_data([], [])
            for sc in center_scatters:
                sc.set_offsets(np.empty((0, 2)))
            for bl in body_lines:
                bl.set_data([], [])
            for wf, wr, sc in zip(wheel_front, wheel_rear, safety_circles):
                wf.set_center((0, 0))
                wr.set_center((0, 0))
                wf.angle = 0.0
                wr.angle = 0.0
                sc.set_center((0, 0))
            return path_lines + center_scatters + body_lines + wheel_front + wheel_rear + safety_circles + added_artists

        def update(frame):
            for a in range(num_agents):
                base = a * nx
                xs = x_trajectory[base, :frame+1]
                ys = x_trajectory[base+1, :frame+1]
                path_lines[a].set_data(xs, ys)
                cx = float(x_trajectory[base, frame])
                cy = float(x_trajectory[base+1, frame])
                center_scatters[a].set_offsets(np.array([[cx, cy]]))
                yaw = float(x_trajectory[base+2, frame])
                front_x = cx + self.model_config.lf * np.cos(yaw)
                front_y = cy + self.model_config.lf * np.sin(yaw)
                rear_x = cx - self.model_config.lr * np.cos(yaw)
                rear_y = cy - self.model_config.lr * np.sin(yaw)
                body_lines[a].set_data([front_x, rear_x], [front_y, rear_y])
                # wheels: front wheel rotated by steering (u_trajectory[0, frame])
                if frame < sim_length:
                    steer = float(u_trajectory[0, frame])
                else:
                    steer = 0.0
                wheel_front[a].set_center((front_x, front_y))
                wheel_rear[a].set_center((rear_x, rear_y))
                wheel_front[a].angle = math.degrees(yaw + steer)
                wheel_rear[a].angle = math.degrees(yaw)
                safety_circles[a].set_center((cx, cy))
            ax.set_title(f"Bicycle Simulation: Step {frame+1}")
            return path_lines + center_scatters + body_lines + wheel_front + wheel_rear + safety_circles + added_artists

        anim = animation.FuncAnimation(fig, update, frames=range(sim_length + 1), init_func=init, interval=interval, blit=False, repeat=True)

        if save_path is not None:
            try:
                fps = max(1, int(round(1000.0 / float(interval))))
                print(f"Saving bicycle animation to {save_path} (fps={fps})")
                writer = animation.PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            except Exception as e:
                print(f"Failed to save bicycle animation to {save_path}: {e}")

        plt.show()
        return anim, fig
    
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
    