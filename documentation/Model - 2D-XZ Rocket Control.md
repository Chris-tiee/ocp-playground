# (Model) 2D-XZ Rocket Control

In this model we consider a rocket in the 2D xz-plane:

<img src="_misc/rocketSketch.svg"/>

## Dynamics
The state and control vector are
$$
\begin{aligned}
x = \begin{bmatrix}
p \\
v \\
\phi \\
\dot{\phi}
\end{bmatrix} = \begin{bmatrix}
p_x \\
p_z \\
v_x \\
v_z \\
\phi \\
\dot{\phi}
\end{bmatrix}\in \mathbb{R}^6, && u = \begin{bmatrix}
T \\
\delta
\end{bmatrix} \in \mathbb{R}^2,
\end{aligned}
$$

with 2D position $p \in \mathbb{R}^2$, velocity $v \in \mathbb{R}^2$, orientation angle $\phi$ relative to the vertical axis and rotational velocity $\dot{\phi}$.
The rocket is controlled using a thrust-vectoring system where the direction of the thrust can be changed using a gimbal.
The controls are the thrust $T$ and the angle $\delta$ with which it is applied.

The following forces act on the rocket:
- The force vector resulting from thrust force $T$:

$$
F_\mathrm{T} = T \begin{bmatrix}  -\sin(\phi + \delta) \\
\cos(\phi + \delta)\end{bmatrix}
$$

- Gravity

$$
F_g = \begin{bmatrix}0 \\
-m g\end{bmatrix}
$$

- Optional: aerodynamic drag force

$$F_D(v, v_\mathrm{wind}) = ?$$

The force from the thruster no only acceleartes the rocket but also creates a torque

$$
M_\mathrm{T} = r \times F_\mathrm{T} =  - T \cdot d \cdot \sin(\delta),
$$
which rotates the rocket around its center of mass.
Here $r$ is the vector of lenght $d$ from the rockets center of mass to the thruster.
The dynamics are then given by:

$$
\begin{aligned}
\begin{bmatrix}
\dot{p} \\
\dot{v} \\
\dot{\phi} \\
\ddot{\phi}
\end{bmatrix} = \dot{x} = f(x,u) =  \begin{bmatrix}
v \\
m^{-1}(F_\mathrm{T} + F_g) \\
\dot{\phi} \\
I^{-1} M_\mathrm{T}
\end{bmatrix}
\end{aligned}
$$


## Details

| State                                     | Symbol               | Unit          |
| ----------------------------------------- | -------------------- | ------------- |
| XZ - position of the rocket                | $p \in \mathbb{R}^2$ | $\mathrm{m}$             |
| XZ - velocity of the rocket                | $v \in \mathbb{R}^2$ | $\mathrm{m}\cdot \mathrm{s}^{-1}$ |
| orientation relative to the vertical axis | $\phi  \in \mathbb{R}$              | $\mathrm{rad}$           |
| angular velocity                          | $\dot{\phi} \in \mathbb{R}$         | $\mathrm{rad}\cdot \mathrm{s}^{-1}$         |

| Control                                     | Symbol               | Unit          |
| ----------------------------------------- | -------------------- | ------------- |
| Thrust force               | $T\in \mathbb{R} $ | $\mathrm{N}$             |
| Thrust angle               | $\delta\in \mathbb{R} $ | $\mathrm{rad}$             |



| Parameter                   | Symbol | Value | Unit                      |
| --------------------------- | ------ | ----- | ------------------------- |
| distance to center of mass to thruster           | $d$    | 0.80     | $\mathrm{m}$             |
| mass                        | $m$    | 0.70   | $\mathrm{kg}$             |
| rotational inertia          | $I$    | 0.30  | $\mathrm{kg}\cdot \mathrm{m}^2$ |
| acceleration due to gravity | $g$    | 9.81  | $\mathrm{m}\cdot \mathrm{s}^{-2}$        |
