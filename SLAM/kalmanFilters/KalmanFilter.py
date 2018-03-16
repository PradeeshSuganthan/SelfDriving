import numpy as np
import matplotlib.pyplot as plt



timesteps = 100
arrayshape = (timesteps, )

#create synthetic data
pure = np.linspace(0,10,timesteps)
noise = np.random.normal(0,1,timesteps)

signal = pure + noise

plt.scatter(range(100), signal, label = 'sensor data')
plt.title("Synthetic data")



#create filter
#state vector (estimate)
x = np.zeros(arrayshape)
xpred = np.zeros(arrayshape)

#covariance matrix (unceratainty)
P = np.zeros(arrayshape)

#State transition model
F = 1
#control input
B = 0
#control vector
u = 0
#Noise covariance
Q = 1
#Measurement covariance
R = 1
#Sensor model
H = 1
  

#prediction 
def predict(x,P,t):
    #State prediction
    x_pred = F*x[t-1] + B*u
    #Covariance prediciton
    P_pred = F*P[t-1]*F + Q

    return x_pred, P_pred

#measurement update
def update(x_pred,P_pred,z,t):
    #Innovation
    y = z-H*x_pred
    #Innovation Covariance
    S = H*P_pred*H + R
    #Kalman Gain
    K = (P_pred*H)/S
    #State update
    x = x_pred + K*y
    #Covariance Update
    P = P_pred - K*(H*P_pred)


    return x, P



for i in range(1, timesteps):
    #predict
    x_pred, P_pred = predict(x, P,i)
    xpred[i] = x_pred
    #update
    x[i], P[i] = update(x_pred,P_pred,signal[i],i)



#visualize
plt.plot(xpred, color='red', label = 'prediction')
plt.plot(x, color = 'black', label = 'actual')
plt.legend()
plt.show()


