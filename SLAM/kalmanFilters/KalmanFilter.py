import numpy as np


class KalmanFilter:
    def __init__(self, timesteps, signal):
        self.x = np.zeros((timesteps,))
        self.P = np.zeros((timesteps,))
        self.xpred = np.zeros((timesteps,))
        self.ppred = np.zeros((timesteps,))
        self.signal = signal
        self.timesteps = timesteps

        # State transition model
        self.F = 1
        # control input
        self.B = 0
        # control vector
        self.u = 0
        # Noise covariance
        self.Q = 1
        # Measurement covariance
        self.R = 1
        # Sensor model
        self.H = 1

    def predict(self, x, P, t):
        # State prediction
        state_prediction = self.F*x[t-1] + self.B * self.u
        # Covariance prediciton
        covariance_prediction = self.F*P[t-1] * self.F + self.Q

        return state_prediction, covariance_prediction

    def update(self, x_pred, P_pred, z, t):
        # Innovation
        y = z - self.H*x_pred
        # Innovation Covariance
        S = self.H*P_pred*self.H + self.R
        # Kalman Gain
        K = (P_pred*self.H)/S
        # State update
        state_update = x_pred + K*y
        # Covariance Update
        covariance_update = P_pred - K*(self.H*P_pred)

        return state_update, covariance_update

    def runFilter(self):
        for i in range(1, self.timesteps):
            # predict
            self.xpred[i], self.ppred[i] = self.predict(self.x, self.P, i)
            # update
            self.x[i], self.P[i] = self.update(self.xpred[i], self.ppred[i], self.signal[i], i)

        return self.x, self.xpred, self.P