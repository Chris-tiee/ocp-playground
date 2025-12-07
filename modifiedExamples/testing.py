import casadi as ca
from dataclasses import dataclass, field
import numpy as np

sampling_time = 0.05
N=40
nx = 6
goal = np.array([3.0, 3.0])

# Defining the path to follow: figure-8 around the goal position
dt = sampling_time
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

# ---- Phase 1: straight line from (0,0) to (0.6, 0.6) ----
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

print(x_ref)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.plot(x_ref[:, 0], x_ref[:, 1], 'bo', markersize=6, label='Reference points')
plt.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Reference trajectory points')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()