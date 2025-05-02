import numpy as np

def add_gaussian_noise(data, noise_std):
    """
    Add isotropic Gaussian noise to input data.

    Parameters:
        data (np.ndarray): Original data of shape (N, D).
        noise_std (float): Standard deviation of Gaussian noise.

    Returns:
        np.ndarray: Noisy data with the same shape as input.
    """
    noise = np.random.normal(loc=0.0, scale=noise_std, size=data.shape)
    return data + noise

# Spiral generator (1D manifold in 2D)
def generate_spiral(num_points=500, rotations=2):
    t = np.linspace(0, rotations * 2 * np.pi, num_points)
    x = t * np.cos(t)
    y = t * np.sin(t)
    return np.stack([x, y], axis=1)

# Line generator (1D manifold in 2D)
def generate_line(num_points_per_axis=300, grid_range=5):
    x = np.linspace(-grid_range, grid_range, num_points_per_axis)
    y = np.zeros_like(x)  # All y-values are 0 → line lies along x-axis
    return np.stack([x, y], axis=1)  # Shape: (N, 2)

# Grid generator (2D manifold in 2D)
def generate_grid(num_points_per_axis=30, grid_range=5):
    x = np.linspace(-grid_range, grid_range, num_points_per_axis)
    y = np.linspace(-grid_range, grid_range, num_points_per_axis)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.ravel(), yv.ravel()], axis=1)


def generate_gaussian(n_samples, dim, mean, std):
    """
    Generate samples from a multivariate Gaussian distribution with a diagonal covariance matrix.

    Args:
        n_samples (int): Number of samples to generate.
        dim (int): Dimension of the Gaussian.
        mean (list or np.ndarray): Mean vector of length `dim`.
        std (list or np.ndarray): Standard deviation vector of length `dim`.

    Returns:
        np.ndarray: Array of shape (n_samples, dim) with generated samples.
    """
    mean = np.array(mean)
    std = np.array(std)

    assert mean.shape[0] == dim, "Mean vector must match the specified dimension."
    assert std.shape[0] == dim, "STD vector must match the specified dimension."

    # Generate standard normal samples and scale them
    samples = np.random.randn(n_samples, dim) * std + mean
    return samples