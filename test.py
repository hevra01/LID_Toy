import matplotlib.pyplot as plt
import numpy as np

# Create a 250x250 grid
grid_size = 32
grid = np.zeros((grid_size, grid_size))

# Plot the grid
plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap='gray', extent=[0, grid_size, 0, grid_size])
plt.grid(True, color='white', linewidth=1.8)
plt.xticks(np.arange(0, grid_size+1, 1))
plt.yticks(np.arange(0, grid_size+1, 1))
plt.title("32x32 Grid")
plt.gca().invert_yaxis()
plt.show()
