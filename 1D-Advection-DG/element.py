import numpy as np

# Uses the Legendre-Gauss-Lobatto points
class Element:

    def __init__(self, k):
        self.k = k
        self.eps = 10e-12
        self.setVandermondeMatrix()

    def setSolution(self, u):
        self.u = u

    # Uses the orthonormal basis described in Hesthaven and Warburton 2008
    def setVandermondeMatrix(self):
        # Use the Chebyshev-Gauss-Lobatto points as first guess
        x = np.flip(np.cos(np.pi * np.linspace(0.0, 1.0, self.k)))
        vandermonde = np.zeros((self.k, self.k))
        xOld = np.ones(self.k)

        vandermonde[:, 0] = 1.0 / np.sqrt(2.0)
        vandermonde[:, 1] = np.sqrt(3.0 / 2.0) * x

        while np.max(np.abs(x - xOld)) > self.eps:
            xOld = x
            for i in range(2, self.k):
                an = (i ** 2) / ((2 * i + 1) * (2 * i - 1))
                an = np.sqrt(an)
                an_1 = ((i - 1) ** 2) / ((2 * i - 1) * (2 * i - 3))
                an_1 = np.sqrt(an_1)
                vandermonde[:, i] = (x * vandermonde[:, i-1] - an_1 * vandermonde[:, i-2]) / an
            x = xOld - (x * vandermonde[:, self.k-1] - vandermonde[:, self.k-2]) / ((self.k - 1) * vandermonde[:, self.k-1])
        self.x = x
        self.vandermonde = vandermonde
