import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- 2D ambient space ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')

# Define a 1D manifold (line) in 2D
x = np.linspace(-2, 2, 100)
y = 0.5 * x
ax.plot(x, y, label='Manifold (1D in 2D)', color='black')

# Pick a point on the line
px, py = 1, 0.5
ax.plot(px, py, 'ro')

# Tangent vector (same direction as line)
tangent = np.array([1, 0.5])
tangent = tangent / np.linalg.norm(tangent)
ax.quiver(px, py, tangent[0], tangent[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Tangent (1D)')

# Normal vector (perpendicular to line)
normal = np.array([-0.5, 1])
normal = normal / np.linalg.norm(normal)
ax.quiver(px, py, normal[0], normal[1], angles='xy', scale_units='xy', scale=1, color='green', label='Normal (1D)')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.grid(True)
ax.legend()
ax.set_title("Tangent and Normal in 2D Ambient Space")

plt.show()

# --- 3D ambient space ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Define a 1D manifold (curve) in 3D
theta = np.linspace(-2, 2, 100)
x = theta
y = 0.5 * theta
z = 0.2 * theta
ax.plot(x, y, z, label='Manifold (1D in 3D)', color='black')

# Pick a point
px, py, pz = 1, 0.5, 0.2
ax.scatter(px, py, pz, color='red')

# Tangent vector
tangent = np.array([1, 0.5, 0.2])
tangent = tangent / np.linalg.norm(tangent)
ax.quiver(px, py, pz, tangent[0], tangent[1], tangent[2], color='blue', label='Tangent (1D)')

# Normal plane spanned by two vectors orthogonal to tangent
# Generate one orthogonal vector using Gram-Schmidt
v1 = np.array([0, 0, 1])
v1 = v1 - np.dot(v1, tangent) * tangent
v1 = v1 / np.linalg.norm(v1)

v2 = np.cross(tangent, v1)

# Generate normal plane points for visualization
plane_range = np.linspace(-0.7, 0.7, 2)
xx, yy = np.meshgrid(plane_range, plane_range)
plane_points = px + v1[0]*xx + v2[0]*yy, py + v1[1]*xx + v2[1]*yy, pz + v1[2]*xx + v2[2]*yy
ax.plot_surface(*plane_points, alpha=0.3, color='green')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.set_title("Tangent (1D) and Normal (2D) in 3D Ambient Space")
ax.legend()

plt.show()
