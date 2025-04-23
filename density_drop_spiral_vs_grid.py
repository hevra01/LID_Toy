import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal
import imageio

# Set random seed for reproducibility
np.random.seed(42)

# Spiral generator (1D manifold in 2D)
def generate_spiral(num_points=500, rotations=2):
    t = np.linspace(0, rotations * 2 * np.pi, num_points)
    x = t * np.cos(t)
    y = t * np.sin(t)
    return np.stack([x, y], axis=1)

# Line generator (1D manifold in 2D)
def generate_line(num_points_per_axis=900, grid_range=5):
    x = np.linspace(-grid_range, grid_range, num_points_per_axis)
    y = np.zeros_like(x)  # All y-values are 0 → line lies along x-axis
    return np.stack([x, y], axis=1)  # Shape: (N, 2)

# Grid generator (2D manifold in 2D)
def generate_grid(num_points_per_axis=30, grid_range=5):
    x = np.linspace(-grid_range, grid_range, num_points_per_axis)
    y = np.linspace(-grid_range, grid_range, num_points_per_axis)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.ravel(), yv.ravel()], axis=1)

# Estimate density using Gaussian kernel
def estimate_density(x, data, sigma):
    kernel = multivariate_normal(mean=x, cov=sigma**2 * np.eye(2))
    densities = kernel.pdf(data)
    return np.mean(densities)

# Create GIF comparing density drop for spiral vs grid
spiral_data = generate_spiral()
grid_data = generate_grid()
line_data = generate_line()
query_point = np.array([0.0, 0.0])
sigmas = np.linspace(0.05, 2.0, 40)

filenames = []
for i, sigma in enumerate(sigmas):
    noisy_line = line_data + np.random.normal(scale=sigma, size=line_data.shape)
    noisy_grid = grid_data + np.random.normal(scale=sigma, size=grid_data.shape)

    density_line = estimate_density(query_point, noisy_line, sigma)
    density_grid = estimate_density(query_point, noisy_grid, sigma)

    print(density_grid, "density_grid")
    print(density_line, "density_line")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, data, noisy_data, title, density in zip(
        axs,
        [line_data, grid_data],
        [noisy_line, noisy_grid],
        [f"Line (LID=1)", f"Grid (LID=2)"],
        [density_line, density_grid]
    ):
        ax.scatter(data[:, 0], data[:, 1], s=5, alpha=0.3, label='Clean')
        ax.scatter(noisy_data[:, 0], noisy_data[:, 1], s=5, alpha=0.5, label='Noisy')
        ax.scatter(*query_point, color='red', label='Query')
        ax.set_title(f"{title}\nσ={sigma:.2f}, ρ={density:.5f}")
        ax.axis('equal')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.legend()

    filename = f"LID_Toy/data/line_vs_grid_data/frame_line_grid_{i:03d}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)

# Create GIF
gif_path = "LID_Toy/data/line_vs_grid_data/density_drop_line_vs_grid_clear.gif"
with imageio.get_writer(gif_path, mode='I', duration=500.2) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

gif_path
