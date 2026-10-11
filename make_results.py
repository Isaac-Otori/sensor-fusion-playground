import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from simulate import simulate
from kalman import KalmanFilter
from ekf import ExtendedKalmanFilter

os.makedirs("results", exist_ok=True)


def rmse(estimate, truth):
    return np.sqrt(np.mean(np.sum((estimate - truth) ** 2, axis=1)))


# ---------- Run everything ----------
t, truth, gps = simulate()
dt = t[1] - t[0]

rng = np.random.default_rng(7)
gyro = 0.3 + rng.normal(0, 0.05, len(t))

# Plain Kalman filter (tuned in Step 4)
kf = KalmanFilter(dt, process_noise=0.0681)
kf.initialize(gps[0])
kf_est = []
for g in gps:
    kf.predict()
    kf.update(g)
    kf_est.append(kf.x[:2].copy())
kf_est = np.array(kf_est)

# EKF (tuned in Step 5)
p0, p1 = gps[:10].mean(axis=0), gps[10:20].mean(axis=0)
d = p1 - p0
ekf = ExtendedKalmanFilter(dt, process_noise=0.005)
ekf.initialize(gps[0], np.linalg.norm(d) / (10 * dt), np.arctan2(d[1], d[0]))
ekf_est = []
for g, w in zip(gps, gyro):
    ekf.predict()
    ekf.update(np.array([g[0], g[1], w]))
    ekf_est.append(ekf.x[:2].copy())
ekf_est = np.array(ekf_est)

raw_e, kf_e, ekf_e = rmse(gps, truth), rmse(kf_est, truth), rmse(ekf_est, truth)
print(f"Raw GPS RMSE: {raw_e:.2f} m")
print(f"Kalman RMSE:  {kf_e:.2f} m")
print(f"EKF RMSE:     {ekf_e:.2f} m")
print(f"EKF improvement over raw GPS: {(1 - ekf_e / raw_e) * 100:.1f}%")

# ---------- Plot 1: final comparison ----------
plt.figure(figsize=(7, 7))
plt.plot(truth[:, 0], truth[:, 1], "b-", linewidth=2, label="Ground truth")
plt.scatter(gps[:, 0], gps[:, 1], s=8, c="red", alpha=0.3, label=f"Raw GPS ({raw_e:.2f} m)")
plt.plot(kf_est[:, 0], kf_est[:, 1], color="orange", linestyle="--", label=f"Kalman ({kf_e:.2f} m)")
plt.plot(ekf_est[:, 0], ekf_est[:, 1], "g-", linewidth=2, label=f"EKF + gyro ({ekf_e:.2f} m)")
plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("GPS vs Kalman vs EKF")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/final_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- Plot 2: error over time ----------
plt.figure(figsize=(9, 4))
plt.plot(t, np.linalg.norm(gps - truth, axis=1), "r", alpha=0.4, label="Raw GPS")
plt.plot(t, np.linalg.norm(kf_est - truth, axis=1), color="orange", label="Kalman")
plt.plot(t, np.linalg.norm(ekf_est - truth, axis=1), "g", linewidth=2, label="EKF + gyro")
plt.xlabel("Time (s)")
plt.ylabel("Position error (m)")
plt.title("Position Error Over Time")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/final_error.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- Demo GIF ----------
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(truth[:, 0], truth[:, 1], "b-", alpha=0.3)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect("equal")
ax.set_title("Sensor fusion: EKF tracking a robot")
ax.grid(alpha=0.3)
gps_dots = ax.scatter([], [], s=8, c="red", alpha=0.5, label="Noisy GPS")
(ekf_line,) = ax.plot([], [], "g-", linewidth=2, label="EKF estimate")
(robot,) = ax.plot([], [], "bo", markersize=8, label="True robot")
ax.legend(loc="upper right", fontsize=8)


def animate(i):
    gps_dots.set_offsets(gps[: i + 1])
    ekf_line.set_data(ekf_est[: i + 1, 0], ekf_est[: i + 1, 1])
    robot.set_data([truth[i, 0]], [truth[i, 1]])
    return gps_dots, ekf_line, robot


anim = FuncAnimation(fig, animate, frames=range(0, len(t), 3), blit=False)
anim.save("results/demo.gif", writer=PillowWriter(fps=20))
plt.close()
print("Saved final_comparison.png, final_error.png and demo.gif to results/")