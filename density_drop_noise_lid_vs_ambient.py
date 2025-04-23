"""
This code creates a GIF to show how the density drops when noise
is added at different dimensions. When the noise is added to the ambient 
space, the density drops faster when compared to when the noise is 
added only in the LID dimension. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import multivariate_normal
import os

# Create a directory to store frames
os.makedirs("frames_1d_vs_2d_noise", exist_ok=True)

# Generate 1D line in 2D space
num_points = 200
x = np.linspace(-5, 5, num_points)
line_data = np.stack([x, np.zeros_like(x)], axis=1)

# Define query point for density estimation
query_point = np.array([0.0, 0.0])

# Define noise levels
sigmas = np.linspace(0.1, 2.0, 20)

# Prepare Gaussian kernel function
def estimate_density(query, data, sigma):
    kernel = multivariate_normal(mean=query, cov=sigma**2 * np.eye(2))
    densities = kernel.pdf(data)
    return np.mean(densities)

# Generate frames for GIF
frames = []
for i, sigma in enumerate(sigmas):
    # Noise only in x-direction (1D noise)
    noise_1d = np.random.normal(scale=sigma, size=(num_points, 1))
    noisy_1d = line_data + np.concatenate([noise_1d, np.zeros_like(noise_1d)], axis=1)

    # Isotropic noise (2D noise)
    noise_2d = np.random.normal(scale=sigma, size=(num_points, 2))
    noisy_2d = line_data + noise_2d

    # Estimate density at query point
    d1 = estimate_density(query_point, noisy_1d, sigma)
    d2 = estimate_density(query_point, noisy_2d, sigma)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(noisy_1d[:, 0], noisy_1d[:, 1], alpha=0.6, label=f"1D Noise\nDensity={d1:.4f}")
    ax[0].scatter(*query_point, color='red')
    ax[0].set_title("Noise in 1D (X only)")
    ax[0].legend()
    ax[0].set_xlim(-10, 10)
    ax[0].set_ylim(-5, 5)
    ax[0].set_aspect('equal')

    ax[1].scatter(noisy_2d[:, 0], noisy_2d[:, 1], alpha=0.6, label=f"2D Noise\nDensity={d2:.4f}")
    ax[1].scatter(*query_point, color='red')
    ax[1].set_title("Isotropic Noise (2D)")
    ax[1].legend()
    ax[1].set_xlim(-10, 10)
    ax[1].set_ylim(-5, 5)
    ax[1].set_aspect('equal')

    fig.suptitle(f"Sigma = {sigma:.2f}")
    fname = f"frames_1d_vs_2d_noise/frame_{i:03d}.png"
    plt.savefig(fname)
    plt.close()
    frames.append(fname)

# Create a GIF from the saved frames
import imageio
gif_path = "./density_drop_1d_vs_2d.gif"
with imageio.get_writer(gif_path, mode='I', duration=1000.8) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

gif_path
