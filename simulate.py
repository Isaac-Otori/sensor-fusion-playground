import numpy as np

def simulate(steps=300, dt=0.1, gps_noise=2.0, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(steps) * dt

    # True path: robot moving in a circle (radius 10 m)
    true_x = 10 * np.cos(0.3 * t)
    true_y = 10 * np.sin(0.3 * t)
    truth = np.column_stack([true_x, true_y])

    # GPS = truth + random noise
    gps = truth + rng.normal(0, gps_noise, truth.shape)
    return t, truth, gps

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t, truth, gps = simulate()
    plt.plot(truth[:, 0], truth[:, 1], label="Truth")
    plt.scatter(gps[:, 0], gps[:, 1], s=6, c="red", label="Noisy GPS")
    plt.axis("equal")
    plt.legend()
    plt.show()