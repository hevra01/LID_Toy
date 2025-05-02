
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