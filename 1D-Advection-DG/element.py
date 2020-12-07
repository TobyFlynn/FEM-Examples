import numpy as np

# Uses the Legendre-Gauss-Lobatto points
class Element:

    def __init__(self, k):
        self.k = k
        self.eps = 10e-12
        self.setVandermondeMatrix()

    def setSolution(self, u):
        self.u = u

    # Need to adapt to modified Legendre polynomials
    def setVandermondeMatrix(self):
        # Use the Chebyshev-Gauss-Lobatto points as first guess
        x = np.flip(np.cos(np.pi * np.linspace(0.0, 1.0, self.k)))
        print(x)
        vandermonde = np.zeros((self.k, self.k))
        xOld = np.ones(self.k)

        while np.max(np.abs(x - xOld)) > self.eps:
            xOld = x
            vandermonde[:, 0] = 1.0
            vandermonde[:, 1] = x
            for i in range(2, self.k):
                vandermonde[:, i] = ((2*i-1) * x * vandermonde[:, i-1] - (i-1)* vandermonde[:, i-2]) / i
            print(vandermonde)
            x = xOld - (x * vandermonde[:, self.k-1] - vandermonde[:, self.k-2]) / ((self.k - 1) * vandermonde[:, self.k-1])
        self.x = x
        self.vandermonde = vandermonde
        print(self.x)
        print(self.vandermonde)
