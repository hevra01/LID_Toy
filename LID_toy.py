"""
This file has the code to generate spiral and grid both clean and noisy. 
It can measure the density and LID. But there is smth wrong with the LID calculation.
slope = d - D => so LID = slope + D
"""

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from generate_data import *
from visualizations.LID_estimation_visualizations import *
from density_drop import estimate_density


def determine_noise(data):
    """
    This function will determine the noise to be used for LID calulcation given the data.
    The data can have dimensions which have varying variances. E.g. a pancake has (height, 
    width, depth), where the depth << height & width. So, it is likely that for LID calculation, 
    we want to ignore the depth dimension.
    """
    
    pass
    return 

def compute_log_densities(data, delta_values, query_point):
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
        noisy_data = add_gaussian_noise(data, noise_std=delta)
        rho = estimate_density(query_point, noisy_data, sigma=delta)
        
        # Clip log densities before fitting:
        log_density = np.log(max(rho, 1e-10))
        log_densities.append(log_density)
    
    return log_densities


def compute_lid(data, log_deltas, ambient_dim):
    """
    Estimate LID by fitting a line to (log δ, log ρ) values.

    Parameters:
        log_deltas (np.ndarray): Logarithm of noise values (log δ).
        log_densities (list or np.ndarray): Corresponding log-density values (log ρ_δ(x)).

    Returns:
        lid (float): Estimated Local Intrinsic Dimensionality.
        coeffs (np.ndarray): Coefficients [slope, intercept] of the fitted line.
    """

    
    # first, we need to estimate the density at different noise levels at the given query point
    log_densities = compute_log_densities(data, log_deltas, query_point)

    # Fit a line: log_density ≈ slope * log_delta + intercept
    coeffs = np.polyfit(log_deltas, log_densities, deg=1)

    # Extract the slope (first coefficient). highest power first.
    slope = coeffs[0]

    # LID = slope + D (ambient dimension)
    lid = slope + ambient_dim

    return lid

    
data_type = "spiral"

# decide on the data type that you want to see the density drop 
if data_type == "line":
    clean_data = generate_line(num_points_per_axis=500, grid_range=5)
elif data_type == "spiral":
    clean_data = generate_spiral(num_points=700, rotations=3)
elif data_type == "grid":
    clean_data = generate_grid(num_points_per_axis=15, grid_range=5)

# choose the ambient dim
ambient_dim = 2

# this is the point where the LID will be calculated
query_point = (0,0)

# determine the noise which will be used for LID estimation
log_deltas = determine_noise(clean_data)


lid = compute_lid(clean_data, log_deltas, query_point, ambient_dim)
