import numpy as np
import matplotlib.pyplot as plt
from simulate import simulate
from kalman import KalmanFilter


def rmse(estimate, truth):
    """Average distance error in metres."""
    return np.sqrt(np.mean(np.sum((estimate - truth) ** 2, axis=1)))


t, truth, gps = simulate()
dt = t[1] - t[0]

kf = KalmanFilter(dt, process_noise=0.1, meas_noise=2.0)
kf.initialize(gps[0])

estimates = []
for z in gps:
    kf.predict()
    kf.update(z)
    estimates.append(kf.x[:2])  # keep only x, y
estimates = np.array(estimates)

raw_error = rmse(gps, truth)
kf_error = rmse(estimates, truth)
print(f"Raw GPS RMSE:    {raw_error:.2f} m")
print(f"Kalman RMSE:     {kf_error:.2f} m")
print(f"Improvement:     {(1 - kf_error / raw_error) * 100:.1f}%")

plt.plot(truth[:, 0], truth[:, 1], label="Truth", linewidth=2)
plt.scatter(gps[:, 0], gps[:, 1], s=6, c="red", alpha=0.4, label="Noisy GPS")
plt.plot(estimates[:, 0], estimates[:, 1], "g--", label="Kalman estimate")
plt.axis("equal")
plt.legend()
plt.show()