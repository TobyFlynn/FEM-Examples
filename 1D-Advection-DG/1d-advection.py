import numpy as np
import matplotlib.pyplot as plt

from element import Element
from grid import StructuredGrid

# Interval to be solving in
interval = (0.0, 1.0)

# Number of elements
nx = 10

# Number of points in a element
k = 4

# Wave speed
a = 1

# Initial conditions
ic = lambda x: np.exp(-40 * (x - 0.5)**2)

# Flux function for linear advection
fluxFunc = lambda x, a=a: a * x

# CFL number (0.9 * CFL limit for rk4, only exact for k = 4, others are a guess)
CFL = 0.9
if k == 3:
    CFL *= 0.145
elif k == 4:
    CFL *= 0.145
elif k == 5:
    CFL *= 0.1
elif k == 6:
    CFL *= 0.05

# Create 1D structured grid and set initial condition
grid = StructuredGrid(interval, nx, k, a, fluxFunc, ic)
dt = (CFL * grid.getdx()) / abs(a)

grid.plot(0)
plt.show()
