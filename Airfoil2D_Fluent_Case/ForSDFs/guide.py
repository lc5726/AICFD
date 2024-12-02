import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gordon_surface(u, v, guide_curve):
    n = len(guide_curve[0])

    surface = np.zeros((n, n, 3))

    for i in range(n):
        for j in range(n):
            surface[i, j] = guide_curve[:, i % n] * (1 - u) + guide_curve[:, (i + 1) % n] * u + \
                             guide_curve[:, j % n] * v + guide_curve[:, (j + 1) % n] * u * v

    return surface

def plot_surface(guide_curve):
    u = np.linspace(0, 1, 50)
    v = np.linspace(0, 1, 50)

    U, V = np.meshgrid(u, v)

    surface = gordon_surface(U, V, guide_curve)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(surface[:, :, 0], surface[:, :, 1], surface[:, :, 2], cmap='viridis', edgecolor='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

# Example: Creating a guide curve
guide_curve = np.array([[0, 1, 2], [0, 2, 0], [0, 1, 0]])

plot_surface(guide_curve)
