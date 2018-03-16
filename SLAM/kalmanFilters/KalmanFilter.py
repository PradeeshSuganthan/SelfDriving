import numpy as np
import matplotlib.pyplot as plt



timesteps = 100
arrayshape = (timesteps, )

#create synthetic data
pure = np.linspace(0,10,timesteps)

noise = np.random.normal(0,1,timesteps)

signal = pure + noise

plt.scatter(range(100), signal)
plt.title("Synthetic data")
plt.show()



#create filter
#measurement update
def update(x,mean2,P,var2):
#    x = 
#    P = 

    return x, P

#prediction update
def predict(x,mean2,P,var2):
#    x[t] = 
#    P[t] = 

    return x, P



motion = np.diff(signal)

mean = 0
variance = 10000
measurement_sig = 4
motion_sig = 2

for i in range(timesteps-1):
    mean, variance = update(mean, signal[i],variance, measurement_sig)
    print('Update: %d, %d' % (mean, variance))

    mean, variance = predict(mean, motion[i], variance, motion_sig )
    print('Predict: %d, %d'% (mean, variance))




#visualize