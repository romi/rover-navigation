import numpy as np
import matplotlib.pyplot as plt

from acre.setup import Setup
from acre.debug import *

def main():
    # Setup is an object factory that instatiates the rover components
    # based on the command line arguments
    setup = Setup()
    set_store_images_flag(False)

    rover = setup.rover
    track_follower = setup.track_follower

    N = 200
    dx = (Setup.FIELD_WIDTH - 10.0 - rover.length) / N
    rover.set_y(2.0)
    measurements = np.zeros((N, 2))
    
    for i in range(N):
        debug_set_iteration(i)
        rover.set_x(5.0 + i * dx)
        measurements[i] = track_follower.estimate_track_error()
        if i % 100 == 0: print(i)


    mean = np.mean(measurements, axis=0)
    std = np.std(measurements, axis=0)
    print(f"['cross-track-error', 'orientation']: mean={mean}, std={std}")

    #plt.hist(distances, bins=20)
    #plt.hist(np.degrees(angles), bins=20)
    #plt.show()
    
if __name__ == "__main__":
    main()
