"""
This file will be used for visualizations.
"""
# Add parent directory to Python path
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generate_data import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D projection
from scipy.stats import gaussian_kde


def plot_gaussian_heatmap(samples, bins=100, cmap='viridis'):
    """
    Plot a heatmap (density visualization) for 2D Gaussian samples.

    Args:
        samples (np.ndarray): Array of shape (n_samples, 2).
        bins (int): Number of bins for the 2D histogram.
        cmap (str): Colormap for the heatmap.
    """
    assert samples.shape[1] == 2, "This function only supports 2D Gaussians for visualization."

    x = samples[:, 0]
    y = samples[:, 1]

    # Plot using a 2D histogram with KDE-style appearance
    plt.figure(figsize=(6, 5))
    sns.kdeplot(x=x, y=y, fill=True, cmap=cmap, thresh=0.05, levels=100)
    plt.title("2D Gaussian Heatmap")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_gaussian_2d_surface(samples, grid_size=100):
    """
    Plots a 3D surface of the density of 2D Gaussian samples.

    Args:
        samples (np.ndarray): Shape (n_samples, 2)
        grid_size (int): Resolution of the density grid
    """
    assert samples.shape[1] == 2, "This function expects 2D Gaussian samples."

    x = samples[:, 0]
    y = samples[:, 1]

    # Create a KDE to estimate density
    kde = gaussian_kde(np.vstack([x, y]))

    # Create a meshgrid over a fixed range
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    X, Y = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                       np.linspace(y_min, y_max, grid_size))
    
    # Evaluate the KDE on the grid
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    # Plot the 3D surface
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_title("3D Surface Plot of 2D Gaussian Density")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Density")
    plt.tight_layout()
    plt.show()

def plot_gaussian_3d(samples, title="3D Gaussian Scatter Plot", alpha=0.5):
    """
    Visualizes samples from a 3D Gaussian distribution using a 3D scatter plot.

    Args:
        samples (np.ndarray): Array of shape (n_samples, 3) representing 3D points.
        title (str): Title of the plot.
        alpha (float): Transparency of the scatter points (between 0 and 1).
    """
    # Ensure that the input has the correct shape
    assert samples.shape[1] == 3, "This function is for visualizing 3D Gaussian data only."

    # Extract x, y, z coordinates
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points in 3D space
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=alpha, s=10)

    # Add color bar to show the z-axis scale
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Z-axis Value')

    # Set axis labels and plot title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Show the plot
    plt.tight_layout()
    plt.show()


# Generate 1000 samples from a 2D Gaussian
n_samples = 5000
dim = 2
mean = [0.0, 0.0]
std = [1.0, 3.5]

samples = generate_gaussian(n_samples, dim, mean, std)

# Visualize
plot_gaussian_heatmap(samples)


#plot_gaussian_2d_surface(samples)