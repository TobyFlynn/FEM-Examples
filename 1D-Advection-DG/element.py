import numpy as np
from numpy.polynomial.polynomial import polyval, polyroots, polyder, polymul
from scipy.sparse.linalg import LinearOperator, cg

# Uses the Legendre-Gauss-Lobatto points
class Element:

    def __init__(self, k, dx):
        self.k = k
        self.dx = dx
        self.j = dx / 2.0
        self.jInv = 2.0 / dx
        self.eps = 10e-12
        self.setVandermondeMatrix()
        self.setDifferentiationMatrix()
        print(self.x)
        print(self.vandermonde)
        print(self.vandermondeGrad)
        print(self.diff)

    def setSolution(self, u):
        self.u = u

    def setLeftFlux(self, ul):
        self.ul = ul

    def setRightFlux(self, ur):
        self.ur = ur

    def getLeftU(self):
        a = np.matmul(self.vandermondeInv, self.u)
        return polyval(-1.0, a)

    def getRightU(self):
        a = np.matmul(self.vandermondeInv, self.u)
        return polyval(1.0, a)

    # Get the Legendre-Gauss-Lobatto points
    def getLGL(self):
        orthoLeg = self.getOrthoLegendrePoly(self.k - 1)
        derOrthoLeg = polyder(orthoLeg)
        overallPoly = polymul(np.array([1, 0, -1]), derOrthoLeg)
        return polyroots(overallPoly)

    def getOrthoLegendrePoly(self, n):
        legendre = [0] * (n)
        legendre.append(1)
        legendre = np.array(legendre)
        legendre = np.polynomial.legendre.leg2poly(legendre)
        return legendre / np.sqrt(2 / (2 * n + 1))

    # Uses the orthonormal basis described in Hesthaven and Warburton 2008
    def setVandermondeMatrix(self):
        x = self.getLGL()
        vandermonde = np.zeros((self.k, self.k))
        vandermondeGrad = np.zeros((self.k, self.k))
        vandermonde[:, 0] = 1.0 / np.sqrt(2.0)
        vandermondeGrad[:, 0] = 0.0

        for i in range(1, self.k):
            OL = self.getOrthoLegendrePoly(i)
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
        rhs = -1.0 * a * self.jInv * np.matmul(self.diff, self.u) # + flux stuff
