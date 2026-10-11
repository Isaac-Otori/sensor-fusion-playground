import numpy as np


class ExtendedKalmanFilter:
    """Turning-motion EKF.
    State = [x, y, v, theta, omega]  (position, speed, heading, turn rate)
    Measurements = [gps_x, gps_y, gyro_omega]
    """

    def __init__(self, dt, gps_noise=2.0, gyro_noise=0.05, process_noise=0.05):
        self.dt = dt
        self.x = np.zeros(5)
        self.P = np.diag([10.0, 10.0, 4.0, 1.0, 1.0])

        # Process noise: how much we distrust the motion model
        self.Q = np.diag([0.01, 0.01, 0.1, 0.01, 0.01]) * process_noise / 0.05

        # Measurement model is linear: GPS gives x, y and gyro gives omega
        self.H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ])
        self.R = np.diag([gps_noise ** 2, gps_noise ** 2, gyro_noise ** 2])

    def initialize(self, gps_start, speed, heading, omega=0.0):
        self.x = np.array([gps_start[0], gps_start[1], speed, heading, omega])

    def predict(self):
        px, py, v, th, w = self.x
        dt = self.dt

        # Nonlinear motion model: move along the heading, then turn
        self.x = np.array([
            px + v * np.cos(th) * dt,
            py + v * np.sin(th) * dt,
            v,
            th + w * dt,
            w,
        ])

        # Jacobian: the linear approximation of the motion model right here
        F = np.eye(5)
        F[0, 2] = np.cos(th) * dt
        F[0, 3] = -v * np.sin(th) * dt
        F[1, 2] = np.sin(th) * dt
        F[1, 3] = v * np.cos(th) * dt
        F[3, 4] = dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ self.H) @ self.P