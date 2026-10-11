import numpy as np


class KalmanFilter:
    """Constant-velocity Kalman filter for 2D position tracking.
    State = [x, y, vx, vy]. Measurement = [x, y] from GPS.
    """

    def __init__(self, dt, process_noise=0.1, meas_noise=2.0):
        # State estimate (our best guess) and its uncertainty
        self.x = np.zeros(4)
        self.P = np.eye(4) * 10.0

        # Motion model: new position = old position + velocity * dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # Measurement model: GPS only sees x and y
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Q = how much we distrust our motion model
        self.Q = np.eye(4) * process_noise
        # R = how noisy the GPS is (variance = noise squared)
        self.R = np.eye(2) * meas_noise ** 2

    def initialize(self, z):
        """Start the filter at the first GPS reading."""
        self.x = np.array([z[0], z[1], 0.0, 0.0])

    def predict(self):
        # Step forward using the motion model
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # How far off was our prediction from the GPS reading?
        y = z - self.H @ self.x
        # Total uncertainty (prediction + sensor)
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain: how much to trust the GPS vs our prediction
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Correct the estimate and shrink the uncertainty
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P