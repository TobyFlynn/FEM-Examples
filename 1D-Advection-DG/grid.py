from element import Element

import numpy as np
import matplotlib.pyplot as plt

class Grid:

    # Periodic boundary conditions
    def roeFlux(self):
        leftElement = self.leftElement
        for i in range(self.nx):
            rightElement = leftElement.getRightElement()
            ul = leftElement.getRightU()
            fl = leftElement.getRightFlux()
            ur = rightElement.getLeftU()
            fr = rightElement.getLeftFlux()
            au = self.a
            if ur != ul:
                au = (fr - fl) / (ur - ul)
            fUpwind = 0.5 * (fl + fr) - 0.5 * (abs(au)) * (ur - ul)
            leftElement.setRightUpwindFlux(fUpwind)
            rightElement.setLeftUpwindFlux(fUpwind)
            leftElement = rightElement

    def storeK0(self):
        element = self.leftElement
        for i in range(self.nx):
            element.storeK0()
            element = element.getRightElement()

    def storeK1AndUpdate(self, dt):
        element = self.leftElement
        for i in range(self.nx):
            element.storeK1AndUpdate(dt)
            element = element.getRightElement()

    def storeK2AndUpdate(self, dt):
        element = self.leftElement
        for i in range(self.nx):
            element.storeK2AndUpdate(dt)
            element = element.getRightElement()

    def storeK3AndUpdate(self, dt):
        element = self.leftElement
        for i in range(self.nx):
            element.storeK3AndUpdate(dt)
            element = element.getRightElement()

    def storeK4AndUpdate(self, dt):
        element = self.leftElement
        for i in range(self.nx):
            element.storeK4AndUpdate(dt)
            element = element.getRightElement()

    def rk4Step(self, dt):
        # Store initial solution in k0
        self.storeK0()

        # Calculate k1, store k1 and update solution
        self.roeFlux()
        self.storeK1AndUpdate(dt)

        # Calculate k2, store k2 and update solution
        self.roeFlux()
        self.storeK2AndUpdate(dt)

        # Calculate k3, store k3 and update solution
        self.roeFlux()
        self.storeK3AndUpdate(dt)

        # Calculate k4, store k4 and update solution
        self.roeFlux()
        self.storeK4AndUpdate(dt)

    def getdx(self):
        return self.dx

    def plot(self, t):
        plt.figure("Solution")
        currentElement = self.leftElement
        yVal = []
        xVal = []
        # Plot solution
        for i in range(self.nx):
            solution = currentElement.getSolution()
            globalSolutionPoints = currentElement.getGlobalSolutionPoints()
            # Elements share boundaries so don't get right boundary
            for n in range(self.k - 1):
                yVal.append(solution[n])
                xVal.append(globalSolutionPoints[n])
            currentElement = currentElement.getRightElement()
        yValSol = self.ic((np.array(xVal) - self.a * t) % 1.0)
        plt.plot(xVal, yVal)
        plt.plot(xVal, yValSol)

class StructuredGrid(Grid):
    def __init__(self, interval, nx, k, a, flux, ic):
        self.nx = nx
        intervalLen = interval[1] - interval[0]
        self.dx = intervalLen / nx
        self.k = k
        self.a = a
        self.fluxFunc = flux
        self.ic = ic

        # Generate the required elements
        x = interval[0]
        self.leftElement = Element(self.k, self.dx, x, self.fluxFunc)
        self.leftElement.setLeftElement(None)
        # Set initial conditions
        solutionPts = self.leftElement.getGlobalSolutionPoints()
        self.leftElement.setSolution(ic(solutionPts))

        prevElement = self.leftElement

        # Construct 1D regular mesh of elements
        for i in range(1, self.nx):
            x = interval[0]+ i * self.dx
            newElement = Element(self.k, self.dx, x, self.fluxFunc)
            newElement.setLeftElement(prevElement)
            prevElement.setRightElement(newElement)
            # Set initial conditions
            solutionPts = newElement.getGlobalSolutionPoints()
            newElement.setSolution(ic(solutionPts))

            prevElement = newElement

        self.rightElement = prevElement
        # Periodic boundary conditions
        self.leftElement.setLeftElement(self.rightElement)
        self.rightElement.setRightElement(self.leftElement)
