import numpy as np
from numpy.polynomial.polynomial import polyval, polyroots, polyder, polymul
from scipy.sparse.linalg import LinearOperator, cg

# Uses the Legendre-Gauss-Lobatto points
class Element:

    def __init__(self, k, dx, x, fluxFunc):
        self.left = None
        self.right = None
        self.k = k
        # Centre of element
        self.xLocation = x + dx / 2
        self.dx = dx
        self.j = dx / 2.0
        self.jInv = 2.0 / dx
        self.fluxFunc = fluxFunc
        self.eps = 10e-12
        self.calcVandermondeMatrix()
        self.setDifferentiationMatrix()

    def setLeftElement(self, l):
        self.left = l

    def setRightElement(self, r):
        self.right = r

    def getSolution(self):
        return self.u

    def getGlobalSolutionPoints(self):
        return self.xLocation + ((self.x) * self.dx) / 2

    def getLeftElement(self):
        return self.left

    def getRightElement(self):
        return self.right

    def setSolution(self, u):
        self.u = u

    def setLeftUpwindFlux(self, ul):
        self.ul = ul

    def setRightUpwindFlux(self, ur):
        self.ur = ur

    def getLeftU(self):
        # Using LGL points so left solution point is on boundary
        return self.u[0]

    def getRightU(self):
        # Using LGL points so right solution point is on boundary
        return self.u[self.k - 1]

    def getLeftFlux(self):
        return self.fluxFunc(self.u[0])

    def getRightFlux(self):
        return self.fluxFunc(self.u[self.k - 1])

    # Get the Legendre-Gauss-Lobatto points
    def calcLGL(self):
        orthoLeg = self.calcOrthoLegendrePoly(self.k - 1)
        derOrthoLeg = polyder(orthoLeg)
        overallPoly = polymul(np.array([1, 0, -1]), derOrthoLeg)
        return polyroots(overallPoly)

    def calcOrthoLegendrePoly(self, n):
        legendre = [0] * (n)
        legendre.append(1)
        legendre = np.array(legendre)
        legendre = np.polynomial.legendre.leg2poly(legendre)
        return legendre / np.sqrt(2 / (2 * n + 1))

    # Uses the orthonormal basis described in Hesthaven and Warburton 2008
    def calcVandermondeMatrix(self):
        x = self.calcLGL()
        vandermonde = np.zeros((self.k, self.k))
        vandermondeGrad = np.zeros((self.k, self.k))
        vandermonde[:, 0] = 1.0 / np.sqrt(2.0)
        vandermondeGrad[:, 0] = 0.0

        for i in range(1, self.k):
            OL = self.calcOrthoLegendrePoly(i)
            vandermonde[:, i] = polyval(x, OL)
            vandermondeGrad[:, i] = polyval(x, polyder(OL))
        self.x = x
        self.vandermonde = vandermonde
        self.vandermondeGrad = vandermondeGrad
        self.vandermondeInv = np.linalg.inv(self.vandermonde)

    def setDifferentiationMatrix(self):
        self.diff = np.matmul(self.vandermondeGrad, self.vandermondeInv)

    def calcRHS(self, a=1.0):
        fluxVec = np.zeros(self.k)
        fluxVec[0] = -1.0 * (self.getLeftU() - self.ul)
        fluxVec[self.k - 1] = self.getRightU() - self.ur
        rhs = -1.0 * a * self.jInv * np.matmul(self.diff, self.u)
        mInv = self.j * np.matmul(self.vandermonde, np.transpose(self.vandermonde))
        rhs = rhs + np.matmul(mInv, fluxVec)
        return rhs
