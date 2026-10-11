import os
import numpy as np
import matplotlib.pyplot as plt
from simulate import simulate
from kalman import KalmanFilter

os.makedirs("results", exist_ok=True)


def rmse(estimate, truth):
    return np.sqrt(np.mean(np.sum((estimate - truth) ** 2, axis=1)))


def run_kf(gps, dt, process_noise, meas_noise=2.0):
    kf = KalmanFilter(dt, process_noise=process_noise, meas_noise=meas_noise)
    kf.initialize(gps[0])
    out = []
    for z in gps:
        kf.predict()
        kf.update(z)
        out.append(kf.x[:2].copy())
    return np.array(out)


t, truth, gps = simulate()
dt = t[1] - t[0]

# --- Tuning sweep: try many process_noise values, keep the best ---
q_values = np.logspace(-3, 1, 25)  # 0.001 to 10
errors = [rmse(run_kf(gps, dt, q), truth) for q in q_values]
best_q = q_values[int(np.argmin(errors))]
print(f"Best process_noise: {best_q:.4f}")

estimates = run_kf(gps, dt, best_q)
raw_err = rmse(gps, truth)
kf_err = rmse(estimates, truth)
print(f"Raw GPS RMSE: {raw_err:.2f} m")
print(f"Kalman RMSE:  {kf_err:.2f} m")
print(f"Improvement:  {(1 - kf_err / raw_err) * 100:.1f}%")

# --- Plot 1: trajectory ---
plt.figure(figsize=(7, 7))
plt.plot(truth[:, 0], truth[:, 1], "b-", linewidth=2, label="Ground truth")
plt.scatter(gps[:, 0], gps[:, 1], s=8, c="red", alpha=0.4, label="Noisy GPS")
plt.plot(estimates[:, 0], estimates[:, 1], "g--", linewidth=2, label="Kalman estimate")
plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Sensor Fusion: Kalman Filter vs Raw GPS")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/trajectory.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 2: error over time ---
gps_dist = np.linalg.norm(gps - truth, axis=1)
kf_dist = np.linalg.norm(estimates - truth, axis=1)
plt.figure(figsize=(9, 4))
plt.plot(t, gps_dist, "r", alpha=0.5, label=f"Raw GPS (RMSE {raw_err:.2f} m)")
plt.plot(t, kf_dist, "g", linewidth=2, label=f"Kalman (RMSE {kf_err:.2f} m)")
plt.xlabel("Time (s)")
plt.ylabel("Position error (m)")
plt.title("Position Error Over Time")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/error_over_time.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 3: tuning curve ---
plt.figure(figsize=(7, 4))
plt.semilogx(q_values, errors, "o-")
plt.axvline(best_q, color="g", linestyle="--", label=f"Best = {best_q:.3f}")
plt.xlabel("process_noise (log scale)")
plt.ylabel("RMSE (m)")
plt.title("Tuning the Kalman Filter")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/tuning.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved 3 images to the results/ folder")