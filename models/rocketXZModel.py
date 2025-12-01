import casadi as ca
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import matplotlib.animation as animation
import numpy as np

 
"""
x = (px, pz, vx, vz, pitch, vpitch)
a = (T, delta)
"""

@dataclass
class RocketXZConfig:
    nx: int = 6
    nu: int = 2
    d: float = 0.8
    mass: float = 0.7
    inertia: float = 0.3
    gravity: float = 9.81


class RocketXZModel:
    def __init__(self, sampling_time):
        self._sampling_time = sampling_time
        self.model_name = "RocketXZModel"
        self.model_config = RocketXZConfig()

        x = ca.MX.sym('x', self.model_config.nx)
        u = ca.MX.sym('u', self.model_config.nu)
        x_dot = ca.vertcat(
            x[2],  # \dot{px}
            x[3],  # \dot{pz}
            - u[0] * ca.sin(x[4] + u[1]) / self.model_config.mass,  # \dot{vx}
            -self.model_config.gravity + u[0] * ca.cos(x[4] + u[1]) / self.model_config.mass,  # \dot{vz}
            x[5],  # \dot{pitch}
            -u[0] * ca.sin(u[1]) * self.model_config.d / self.model_config.inertia  # \dot{vpitch}
        )

        # set up integrator for discrete dynamics
        # multiply xdot by sampling time (time scaling), since casadi integrator integrates over [0,1] by default
        dae = {'x': x, 'p': u, 'ode': self._sampling_time * x_dot}
        self.I = ca.integrator('I', 'rk', dae)

        self.A_func = ca.Function('A_func', [x, u], [ca.jacobian(x_dot, x)])
        self.B_func = ca.Function('B_func', [x, u], [ca.jacobian(x_dot, u)])

        self.A_disc_func = ca.Function('A_disc_func', [x, u], [ca.jacobian(self.I(x0=x, p=u)['xf'], x)])
        self.B_disc_func = ca.Function('B_disc_func', [x, u], [ca.jacobian(self.I(x0=x, p=u)['xf'], u)])

        # create continuous and discrete dynamics
        self.f_cont = ca.Function('f_cont', [x, u], [x_dot])
        self.f_disc = ca.Function('f_disc', [x, u], [self.I(x0=x, p=u)['xf']])

    def linearizeContinuousDynamics(self, x, u):
        A = self.A_func(x, u).full()
        B = self.B_func(x, u).full()
        return A, B


    def linearizeDiscreteDynamics(self, x, u):
        A = self.A_disc_func(x, u).full()
        B = self.B_disc_func(x, u).full()
        return A, B


    def animateSimulation(self, x_trajectory, u_trajectory, additional_lines_or_scatters=None, save_path: str = None):
        def compute_rocket_vertices(x: np.ndarray):
            rocket_width = 0.2
            rot_matrix = np.array([[np.cos(x[4]), -np.sin(x[4])], [np.sin(x[4]), np.cos(x[4])]])
            vertices = np.array([[rocket_width, -rocket_width, -rocket_width, rocket_width, rocket_width], [self.model_config.d, self.model_config.d, -self.model_config.d, -self.model_config.d, self.model_config.d], ])
            vertices = np.concatenate((vertices, np.array([[0.], [self.model_config.d * 1.3]]), np.array([[-rocket_width], [self.model_config.d]])), axis=1)
            vertices = rot_matrix @ vertices + x[:2].reshape(2, 1)
            return vertices.T

        def compute_thrust_vertices(x: np.ndarray, u: np.ndarray):
            thrust_width = 0.1
            thrust_scaling = 0.05
            vertices = np.array([[0.0, thrust_width, -thrust_width], [0.0, -u[0], -u[0]]])
            vertices[1, :] *= thrust_scaling
            rot_matrix_1 = np.array([[np.cos(u[1]), -np.sin(u[1])], [np.sin(u[1]), np.cos(u[1])]])
            vertices = rot_matrix_1 @ vertices + np.array([[0.], [-self.model_config.d]])
            rot_matrix_2 = np.array([[np.cos(x[4]), -np.sin(x[4])], [np.sin(x[4]), np.cos(x[4])]])
            vertices = rot_matrix_2 @ vertices + x[:2].reshape(2, 1)
            return vertices.T

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
        ax.set_ylabel(r'$p_{\mathrm{z}}$ in m', fontsize=14)

        # static artists
        center_scatter = ax.scatter([], [], color="tab:gray", s=50, zorder=2)
        rocket_poly = patches.Polygon(np.zeros((5, 2)), closed=True, color="tab:blue", linewidth=2, zorder=1)
        ax.add_patch(rocket_poly)
        thrust_poly = patches.Polygon(np.zeros((3, 2)), closed=True, color="tab:red", alpha=0.5, zorder=0)
        ax.add_patch(thrust_poly)

        # additional static lines or scatters (these are rendered once)
        added_artists = []
        if additional_lines_or_scatters is not None:
            for key, value in additional_lines_or_scatters.items():
                if value["type"] == "scatter":
                    sc = ax.scatter(value["data"][0], value["data"][1], color=value.get("color", "k"), s=value.get("s", 20), label=key, marker=value.get("marker", ","), zorder=3)
                    added_artists.append(sc)
                elif value["type"] == "line":
                    ln, = ax.plot(value["data"][0], value["data"][1], color=value.get("color", "k"), linewidth=2, label=key)
                    added_artists.append(ln)

        ax.set_title(f"Rocket XZ Simulation: Step 1")
        if added_artists:
            ax.legend(loc="lower right")
        fig.subplots_adjust(bottom=0.15)

        # hardcoded slowdown: make animation run ~2x slower than real-time
        interval = max(50, int(self._sampling_time * 1000 * 2.0))

        def init():
            center_scatter.set_offsets(np.empty((0, 2)))
            rocket_poly.set_xy(np.zeros((5, 2)))
            thrust_poly.set_xy(np.zeros((3, 2)))
            return [center_scatter, rocket_poly, thrust_poly] + added_artists

        def update(i):
            # current state
            xi = x_trajectory[:, i]
            center = np.array([xi[0], xi[1]])
            center_scatter.set_offsets(center.reshape(1, 2))

            rocket_vertices = compute_rocket_vertices(xi)
            rocket_poly.set_xy(rocket_vertices)

            if i < sim_length:
                ui = u_trajectory[:, i]
                thrust_vertices = compute_thrust_vertices(xi, ui)
                thrust_poly.set_xy(thrust_vertices)
                thrust_poly.set_visible(True)
            else:
                thrust_poly.set_visible(False)

            ax.set_title(f"Rocket XZ Simulation: Step {i+1}")
            return [center_scatter, rocket_poly, thrust_poly] + added_artists

        anim = animation.FuncAnimation(fig, update, frames=range(sim_length + 1), init_func=init, interval=interval, blit=False, repeat=True)

        # If a save path is provided, attempt to save using PillowWriter (GIF)
        if save_path is not None:
            try:
                fps = max(1, int(round(1000.0 / float(interval))))
                print(f"Saving animation to {save_path} (fps={fps})")
                writer = animation.PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            except Exception as e:
                print(f"Failed to save animation to {save_path}: {e}")

        plt.show()
        return anim, fig

    def plotSimulation(self, x_trajectory: np.ndarray, u_trajectory: np.ndarray, figsize=(8, 10)):
        """Plot states and controls over time.

        States grouping:
          - positions px and pz in one subplot
          - velocities vx and vz in one subplot
          - pitch in one subplot
          - pitch rate in one subplot

        Controls are plotted as piecewise-constant (step) signals with
        `where='post'` so that u_k is extended until t_{k+1}.
        """
        dt = self._sampling_time
        nx = self.model_config.nx
        nu = self.model_config.nu
        sim_length = u_trajectory.shape[1]

        t_x = np.arange(sim_length + 1) * dt
        # we will plot controls on the same grid as states; append NaN to controls
        # so their length matches t_x and step plotting with where='post' works
        # (u_k shown until t_{k+1}).

        fig, axs = plt.subplots(6, 1, figsize=figsize, constrained_layout=True)

        # Positions px, pz
        axs[0].plot(t_x, x_trajectory[0, :], label=r'$p_{\mathrm{x}}$')
        axs[0].plot(t_x, x_trajectory[1, :], label=r'$p_{\mathrm{z}}$')
        axs[0].set_ylabel(r'position $p$ in m')
        axs[0].legend()

        # Velocities vx, vz
        axs[1].plot(t_x, x_trajectory[2, :], label=r'$v_{\mathrm{x}}$')
        axs[1].plot(t_x, x_trajectory[3, :], label=r'$v_{\mathrm{z}}$')
        axs[1].set_ylabel(r'velocity $v$ in m/s')
        axs[1].legend()

        # Pitch
        axs[2].plot(t_x, x_trajectory[4, :], label='pitch')
        axs[2].set_ylabel(r'pitch $\theta$ in rad')

        # Pitch rate
        axs[3].plot(t_x, x_trajectory[5, :], label='pitch rate')
        axs[3].set_ylabel(r'pitch rate $\dot{\theta}$ in rad/s')

        # Controls: thrust T (extend with NaN to align with t_x)
        u_T_plot = np.concatenate((np.asarray(u_trajectory[0, :]).flatten(), [np.nan]))
        axs[4].step(t_x, u_T_plot, where='post', label='T')
        axs[4].set_ylabel(r'thrust $T$ in N')

        # Controls: delta
        # Controls: delta (extend with NaN to align with t_x)
        u_delta_plot = np.concatenate((np.asarray(u_trajectory[1, :]).flatten(), [np.nan]))
        axs[5].step(t_x, u_delta_plot, where='post', label='delta')
        axs[5].set_ylabel(r'gimbal angle $\delta$ in rad')
        axs[5].set_xlabel('time in s')

        fig.suptitle('Rocket states and controls')

        # Set x-axis limits exactly to the plotted time range (no extra whitespace)
        x_min = t_x[0]
        x_max = t_x[-1]
        for ax in axs:
            ax.set_xlim(x_min, x_max)

        return fig, axs
    