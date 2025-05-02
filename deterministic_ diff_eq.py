import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define three different deterministic functions
def f_time_only(t):      # dx/dt = 2t
    return 2 * t

def f_exp_growth(x):     # dx/dt = x
    return x

def f_quadratic(x):      # dx/dt = x^2
    return x ** 2

# Adjusted parameters to prevent blow-up for dx/dt = x^2
T = 0.7                  # Reduced total time
dt = 0.01                # Smaller time step
N = int(T / dt)          # Number of steps
t = np.linspace(0, T, N) # Time array
x0 = 1.0                 # Initial value

# Reinitialize trajectories
x_time = np.zeros(N)
x_exp = np.zeros(N)
x_quad = np.zeros(N)

x_time[0] = x0
x_exp[0] = x0
x_quad[0] = x0

# Compute trajectories with safe bounds
for i in range(1, N):
    x_time[i] = x_time[i - 1] + f_time_only(t[i - 1]) * dt
    x_exp[i] = x_exp[i - 1] + f_exp_growth(x_exp[i - 1]) * dt
    x_quad[i] = x_quad[i - 1] + f_quadratic(x_quad[i - 1]) * dt
    if not np.isfinite(x_quad[i]):  # stop if x^2 explodes
        x_quad[i:] = np.nan
        break

# ---------- Create Animation ----------
fig, ax = plt.subplots()
line1, = ax.plot([], [], lw=2, label=r"$\frac{dx}{dt} = 2t$")
line2, = ax.plot([], [], lw=2, label=r"$\frac{dx}{dt} = x$")
line3, = ax.plot([], [], lw=2, label=r"$\frac{dx}{dt} = x^2$")
ax.set_xlim(0, T)
ax.set_ylim(0, np.nanmax([x_time[-1], x_exp[-1], x_quad[np.isfinite(x_quad)].max()]) * 1.1)
ax.set_title("Comparison of Deterministic ODEs")
ax.set_xlabel("Time $t$")
ax.set_ylabel("$x_t$")
ax.grid(True)
ax.legend()

def update(frame):
    line1.set_data(t[:frame], x_time[:frame])
    line2.set_data(t[:frame], x_exp[:frame])
    line3.set_data(t[:frame], x_quad[:frame])
    return line1, line2, line3

ani = animation.FuncAnimation(fig, update, frames=N, interval=350, blit=True)

# Save as gif
gif_path = "./deterministic_comparison.gif"
ani.save(gif_path, writer="pillow")
gif_path
