import numpy as np
import matplotlib.pyplot as plt

from acre.setup import *
from acre.rover import *

def main():
    setup = Setup()
    N = 2000
    positions = np.zeros((N, 3))

    print(f"std rover: {setup.args.std_rover}")
    
    for i in range(N):
        # Drive forward starting at position (0,0), for 1 second at 1
        # m/s. The exact position after moving is (x,y)=(1,0).
        positions[i, :] = move_with_differential_speed(setup.width,
                                                       [0.0, 0.0, 0.0],
                                                       [1.0, 1.0], 1.0,
                                                       setup.args.std_rover)
        positions[i, 0] -= 1.0

    mean = np.mean(positions, axis=0)
    std = np.std(positions, axis=0)
    print(f"mean {mean}, std {std}")

    # plt.hist(positions[:, 0])
    # plt.hist(positions[:, 1])
    plt.hist(np.degrees(positions[:, 2]))
    plt.show()

    
if __name__ == "__main__":
    main()
