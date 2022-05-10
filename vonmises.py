from scipy.stats import vonmises
import numpy as np
import matplotlib.pyplot as plt

n = 1000
std = np.radians(1)

x = np.random.randn(n) * std
#plt.hist(np.degrees(x))
#plt.show()

kappa, loc, scale = vonmises.fit(x, fscale=1)

#x = np.radians(np.linspace(-10.0, 10.0, 200))

x = np.linspace(np.radians(-10.0), np.radians(10.0), 200)
y = vonmises.pdf(x, kappa, loc=np.radians(2.0))
plt.plot(np.degrees(x), y)
plt.show()

#for i in range(-100, 101):
#    degrees = i / 10.0
#    x = np.radians(degrees)
#    print(f"{degrees} {y}")
