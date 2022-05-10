import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("position-error-versus-angle-error.txt")

plt.plot(data[:,0], data[:,4])
plt.plot(data[:,0], data[:,5])
plt.plot(data[:,0], np.degrees(data[:,6]))
plt.show()

