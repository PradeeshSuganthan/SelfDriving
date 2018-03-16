import numpy as np
import matplotlib.pyplot as plt

from KalmanFilter import KalmanFilter

timesteps = 100

#create synthetic data
pure = np.linspace(0,10,timesteps)
noise = np.random.normal(0,1,timesteps)

signal = pure + noise

plt.scatter(range(100), signal, label = 'sensor data')
plt.title("Synthetic data")


#create filter instance and run
filter = KalmanFilter(timesteps, signal)
filter.F = 1.015
filter.Q = 1
filter.R = 50

x, P = filter.runFilter()

# visualize
plt.plot(x, color='red', label = 'prediction')
plt.plot(pure, color = 'black', label = 'actual')
plt.legend()
plt.show()