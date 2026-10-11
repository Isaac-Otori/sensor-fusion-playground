import numpy as np
from simulate import simulate
from kalman import KalmanFilter
from ekf import ExtendedKalmanFilter


def rmse(estimate, truth):
    return np.sqrt(np.mean(np.sum((estimate - truth) ** 2, axis=1)))


t, truth, gps = simulate()
dt = t[1] - t[0]

# Simulated gyroscope: true turn rate (0.3 rad/s) plus small noise
rng = np.random.default_rng(7)
gyro = 0.3 + rng.normal(0, 0.05, len(t))

# Rough starting guess for speed and heading from the first 20 GPS points
p0 = gps[:10].mean(axis=0)
p1 = gps[10:20].mean(axis=0)
d = p1 - p0
speed0 = np.linalg.norm(d) / (10 * dt)
heading0 = np.arctan2(d[1], d[0])

# --- EKF ---
ekf = ExtendedKalmanFilter(dt, process_noise=0.005)
ekf.initialize(gps[0], speed0, heading0)

ekf_est = []
for g, w in zip(gps, gyro):
    ekf.predict()
    ekf.update(np.array([g[0], g[1], w]))
    ekf_est.append(ekf.x[:2].copy())
ekf_est = np.array(ekf_est)

# --- Plain Kalman filter, for comparison ---
kf = KalmanFilter(dt, process_noise=0.0681)
kf.initialize(gps[0])
kf_est = []
for g in gps:
    kf.predict()
    kf.update(g)
    kf_est.append(kf.x[:2].copy())
kf_est = np.array(kf_est)

raw = rmse(gps, truth)
k = rmse(kf_est, truth)
e = rmse(ekf_est, truth)
print(f"Raw GPS RMSE:  {raw:.2f} m")
print(f"Kalman RMSE:   {k:.2f} m")
print(f"EKF RMSE:      {e:.2f} m")
print(f"EKF improvement over raw GPS: {(1 - e / raw) * 100:.1f}%")