"""
Given the data type (spiral, line, grid), this file tried to find the 
drop/change in density as noise level increases at a given query point.
"""


from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from generate_data import *

def estimate_density(x, data, sigma):
    """
    Estimate the density at point x using a Gaussian kernel with std=sigma.
    For more difficult datasets, we would've used normalizing flows or other
    models that learn density. 
    """
    
    # this is only a gaussian function/object (centered around "x") that can evaluate the value 
    # given a data point.  
    kernel = multivariate_normal(mean=x, cov=sigma**2 * np.eye(data.shape[1]))

    # This applies the Gaussian bump to every point in your dataset
    # A score for how close each data point is to the query point.
    densities = kernel.pdf(data)
    
    # Averages all the scores = KDE estimate of density at the query.
    return np.mean(densities)


def measure_density_drop(clean_data, sigmas, query_point):
    """
    Given the clean data, the query point, and the noise levels,
    this function measures the density drop. If the drop is fast 
    then we likely have a low dimensional data. However, if the drop
    is slow, we likely have high dimensional data. 
    """
    
    densities = []

    for sigma in sigmas:
        # creae noisy data 
        noisy_data = add_gaussian_noise(clean_data, sigma)

        # Estimate density at query point
        density = estimate_density(query_point, noisy_data, sigma)
        densities.append(density)

    # Plot density vs. noise
    plt.figure(figsize=(7, 4))
    plt.plot(sigmas, densities, 'o-', color='blue')
    plt.xlabel("Noise Std Dev (σ)")
    plt.ylabel("Estimated Density at Query Point")
    #plt.title("Density Drop vs. Noise Level")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig("/home/hevra/Desktop/thesis_project/LID_toy/LID_Toy/data_second/spiral/density_drop/noise_0.1_to_2_total_20/700_3.png", dpi=300)  

    plt.show()

    return densities


data_type = "spiral"

# decide on the data type that you want to see the density drop 
if data_type == "line":
    clean_data = generate_line(num_points_per_axis=500, grid_range=5)
elif data_type == "spiral":
    clean_data = generate_spiral(num_points=700, rotations=3)
elif data_type == "grid":
    clean_data = generate_grid(num_points_per_axis=15, grid_range=5)


# define the noise levels 
# Define noise levels
# the first parameter is start, the second is end, and the third is the number of values .
sigmas = np.linspace(0.1, 2.0, 20) 

# define the query point
query_point = np.array([0.0, 0.0])


measure_density_drop(clean_data, sigmas, query_point)