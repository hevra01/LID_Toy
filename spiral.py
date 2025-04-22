from scipy.stats import multivariate_normal

import numpy as np
import matplotlib.pyplot as plt


def generate_spiral(num_points=1000, rotations=3, noise_std=0.5):
    """
    Generate a 1D spiral manifold in 2D space with optional Gaussian noise.
    
    Parameters:
        num_points (int): Number of points in the spiral.
        rotations (float): Number of spiral turns.
        noise_std (float): Standard deviation of Gaussian noise to add.
        
    Returns:
        spiral_clean (np.ndarray): Original spiral (no noise), shape (num_points, 2).
        spiral_noisy (np.ndarray): Spiral with added Gaussian noise, shape (num_points, 2).
    """
    # Parameter t goes from 0 to rotations * 2pi
    t = np.linspace(0, rotations * 2 * np.pi, num_points)
    t1= np.linspace(0, 10, 20)
    
    # Spiral parametric equations
    x = t * np.cos(t)
    y = t * np.sin(t)
    
    spiral_clean = np.stack([x, y], axis=1)

    # Add Gaussian noise
    noise = np.random.normal(loc=0.0, scale=noise_std, size=spiral_clean.shape)
    spiral_noisy = spiral_clean + noise

    return spiral_clean, spiral_noisy

def generate_2d_grid(num_points_per_axis=50, grid_range=5, noise_std=0.5):
    """
    Generate a 2D uniform grid in 2D space, with optional Gaussian noise.

    Parameters:
        num_points_per_axis (int): Number of points per axis (total = N²).
        grid_range (float): The grid spans from -grid_range to +grid_range.
        noise_std (float): Standard deviation of added Gaussian noise.

    Returns:
        grid_clean (np.ndarray): Clean grid, shape (num_points², 2).
        grid_noisy (np.ndarray): Noisy grid, shape (num_points², 2).
    """
    x = np.linspace(-grid_range, grid_range, num_points_per_axis)
    y = np.linspace(-grid_range, grid_range, num_points_per_axis)
    xv, yv = np.meshgrid(x, y)
    grid_clean = np.stack([xv.ravel(), yv.ravel()], axis=1)

    noise = np.random.normal(loc=0.0, scale=noise_std, size=grid_clean.shape)
    grid_noisy = grid_clean + noise

    return grid_clean, grid_noisy

def plot(clean, noisy=None, save=False, filename="spiral_plot.png"):
    """
    Plot the original and noisy spiral.

    Parameters:
        spiral_clean (np.ndarray): Clean spiral, shape (N, 2).
        spiral_noisy (np.ndarray or None): Noisy spiral to plot, optional.
    """
    plt.figure(figsize=(8, 8))

    plt.scatter(clean[:, 0], clean[:, 1], 
                    s=10, color='blue', alpha=0.6, label=f"Clean Spiral")

    if noisy is not None:
        plt.scatter(noisy[:, 0], noisy[:, 1], 
                    s=10, color='red', alpha=0.6, label=f"Noisy Spiral")

    plt.axis('equal')
    plt.title("1D Spiral Manifold in 2D")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # Save if requested
    if save:
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")

    plt.show()

def estimate_density(x, data, sigma):
    """
    Estimate the density at point x using a Gaussian kernel with std=sigma.
    """
    
    # this is only a gaussian function/object (centered around "x") that can evaluate the value 
    # given a data point  
    kernel = multivariate_normal(mean=x, cov=sigma**2 * np.eye(data.shape[1]))

    # This applies the Gaussian bump to every point in your dataset
    # A score for how close each data point is to the query point.
    densities = kernel.pdf(data)
    
    # Averages all the scores = KDE estimate of density at the query.
    return np.mean(densities)

def compare_density_for_noise_levels(
    start, end, num_steps=5,
    query_point=(2, 3),
    num_points=200,
    rotations=3,
    grid_size=50,
    grid_range=5
):
    """
    Compare density estimates for two noise levels using both grid and spiral datasets.

    Parameters:
        noise1 (float): First noise standard deviation to test.
        noise2 (float): Second noise standard deviation to test.
        query_point (tuple): The point at which to estimate density.
        num_points (int): Number of points in the spiral.
        rotations (float): Number of spiral rotations.
        grid_size (int): Number of grid points per axis.
        grid_range (float): The span of the grid from -range to +range.
    """

    noise_values = np.linspace(start, end, num_steps)

    for noise in noise_values:
        print(f"\n--- Evaluating for noise std = {noise} ---")

        # GRID
        grid_clean, grid_noisy = generate_2d_grid(num_points_per_axis=grid_size, grid_range=grid_range, noise_std=noise)
        d1 = estimate_density(query_point, grid_clean, sigma=noise)
        d2 = estimate_density(query_point, grid_noisy, sigma=noise)
        print(f"Grid Density (Clean): {d1:.5f}")
        print(f"Grid Density (Noisy): {d2:.5f}")
        plot(grid_clean, grid_noisy, save=True, filename=f"grid_noise_{noise:.2f}.png")

        # SPIRAL
        spiral_clean, spiral_noisy = generate_spiral(num_points=num_points, rotations=rotations, noise_std=noise)
        d3 = estimate_density(query_point, spiral_clean, sigma=noise)
        d4 = estimate_density(query_point, spiral_noisy, sigma=noise)
        print(f"Spiral Density (Clean): {d3:.5f}")
        print(f"Spiral Density (Noisy): {d4:.5f}")
        plot(spiral_clean, spiral_noisy, save=True, filename=f"spiral_noise_{noise:.2f}.png")


def compute_lid_from_densities(log_deltas, log_densities):
    """
    Estimate LID by fitting a line to (log δ, log ρ) values.

    Parameters:
        log_deltas (np.ndarray): Logarithm of noise values (log δ).
        log_densities (list or np.ndarray): Corresponding log-density values (log ρ_δ(x)).

    Returns:
        lid (float): Estimated Local Intrinsic Dimensionality.
        coeffs (np.ndarray): Coefficients [slope, intercept] of the fitted line.
    """

    # Fit a line: log_density ≈ slope * log_delta + intercept
    coeffs = np.polyfit(log_deltas, log_densities, deg=1)

    # Extract the slope (first coefficient)
    slope = coeffs[0]

    # LID is defined as the negative of this slope
    lid = -slope

    return lid, coeffs

def plot_lid_curve(log_deltas, log_densities, title="LID Plot"):
    """
    Plot log(ρ) vs log(δ) and fit a line to estimate LID.

    Parameters:
        log_deltas (np.ndarray): log(δ) values.
        log_densities (list): Corresponding log(ρ) values.
        title (str): Title for the plot.

    Returns:
        lid (float): Estimated local intrinsic dimensionality.
    """
    lid, coeffs = compute_lid_from_densities(log_deltas, log_densities)
    fitted_line = np.polyval(coeffs, log_deltas)

    plt.plot(log_deltas, log_densities, 'o-', label='log density')
    plt.plot(log_deltas, fitted_line, '--', label=f'LID ≈ {lid:.2f}')
    plt.xlabel('log(δ)')
    plt.ylabel('log(ρ)')
    plt.title(title)
    plt.grid(True)
    plt.legend()

    return lid

def compute_log_densities(model_fn, delta_values, query_point):
    """
    Compute log-densities at a query point for a dataset generator over different noise scales.

    Parameters:
        model_fn (function): A function like `generate_spiral` or `generate_2d_grid`.
        delta_values (np.ndarray): Array of noise std deviations (delta).
        query_point (tuple): The point at which density is estimated.

    Returns:
        log_densities (list): List of log(ρ_δ(x)) values across noise scales.
    """
    log_densities = []

    for delta in delta_values:
        _, noisy_data = model_fn(noise_std=delta)
        rho = estimate_density(query_point, noisy_data, sigma=delta)
        log_densities.append(np.log(rho))

    return log_densities

def compare_and_compute_lid(
    start, end, num_steps=10,
    query_point=(2, 3),
    num_points=200,
    rotations=3,
    grid_size=50,
    grid_range=5
):
    """
    Compare and compute LID for grid and spiral datasets over a range of noise levels.

    Parameters:
        start, end (float): Start and end of noise interval.
        num_steps (int): Number of noise scales (δ) to evaluate.
        query_point (tuple): The point at which to estimate LID.
        num_points, rotations (int, float): Parameters for the spiral.
        grid_size, grid_range (int, float): Parameters for the 2D grid.

    Returns:
        (lid_grid, lid_spiral): LID values for grid and spiral datasets.
    """
    delta_values = np.linspace(start, end, num_steps)
    log_deltas = np.log(delta_values)

    print(delta_values, log_deltas)
    exit()

    # Define dataset generators as closures with preset parameters
    spiral_fn = lambda noise_std: generate_spiral(num_points=num_points, rotations=rotations, noise_std=noise_std)
    grid_fn = lambda noise_std: generate_2d_grid(num_points_per_axis=grid_size, grid_range=grid_range, noise_std=noise_std)

    # Compute log densities
    spiral_log_densities = compute_log_densities(spiral_fn, delta_values, query_point)
    grid_log_densities = compute_log_densities(grid_fn, delta_values, query_point)

    # Plotting both curves
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    lid_grid = plot_lid_curve(log_deltas, grid_log_densities, title="Grid: log(ρ) vs log(δ)")

    plt.subplot(1, 2, 2)
    lid_spiral = plot_lid_curve(log_deltas, spiral_log_densities, title="Spiral: log(ρ) vs log(δ)")

    plt.tight_layout()
    plt.show()

    return lid_grid, lid_spiral

compare_and_compute_lid(start=0.1, end=1, num_steps=10)
