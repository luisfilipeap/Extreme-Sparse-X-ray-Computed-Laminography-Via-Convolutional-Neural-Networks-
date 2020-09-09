import pylops
import numpy as np
from scipy.sparse.linalg import LinearOperator


class GradientOperator(LinearOperator):
    def __init__(self, n, dtype=None):
        self.n = n
        self.shape = (n-1, n)
        if dtype is None:
            self.dtype = np.float64
        else:
            self.dtype = dtype

    def _matvec(self, x):
        return np.diff(x.ravel())

    def _rmatvec(self, x):
        x = x.ravel()
        y = np.zeros(self.n)
        y[1:-1] = -np.diff(x)
        y[0] = -x[0]
        y[-1] = x[-1]
        return y


class SquareGradientOperator(LinearOperator):
    def __init__(self, n, dtype=None):
        self.n = n
        self.shape = (n, n)
        if dtype is None:
            self.dtype = np.float64
        else:
            self.dtype = dtype

    def _matvec(self, x):
        x = x.ravel()
        y = np.zeros(self.n)
        y[1:] = np.diff(x)
        y[0] = x[0]
        return y

    def _rmatvec(self, x):
        x = x.ravel()
        y = np.zeros(self.n)
        y[:-1] = -np.diff(x)
        y[-1] = x[-1]
        return y


class SquareIntegralOperator(LinearOperator):
    def __init__(self, n, dtype=None):
        self.n = n
        self.shape = (n, n)
        if dtype is None:
            self.dtype = np.float64
        else:
            self.dtype = dtype

    def _matvec(self, x):
        x = x.ravel()
        y = np.cumsum(x)
        return y

    def _rmatvec(self, x):
        x = x.ravel()
        y = np.flip(np.cumsum(np.flip(x, 0)), 0)
        return y

# semi orthogonal splitting
def even_gradient(n):
    even = pylops.Restriction(n - 1, np.arange(0, n-1, 2))
    return even @ GradientOperator(n)

def odd_gradient(n):
    odd = pylops.Restriction(n - 1, np.arange(1, n-1, 2))
    return odd @ GradientOperator(n)


# 2D gradient operators
def gradient_x(a,b,c):
    temp = pylops.Kronecker(pylops.Identity(a), GradientOperator(b))
    return pylops.Kronecker(temp, pylops.Identity(c))

def gradient_y(a,b,c):
    temp =  pylops.Kronecker(GradientOperator(a), pylops.Identity(b))
    return pylops.Kronecker(temp, pylops.Identity(c))

def gradient_z(a,b,c):
    temp =  pylops.Kronecker(pylops.Identity(a), pylops.Identity(b))
    return pylops.Kronecker(temp, GradientOperator(c))




# Semi-orthogonal splitting of 2D gradient operators
def even_gradient_x(n):
    even = pylops.Restriction(n - 1, np.arange(0, n-1, 2))
    return pylops.Kronecker(pylops.Identity(n), even @ GradientOperator(n))

def odd_gradient_x(n):
    odd = pylops.Restriction(n - 1, np.arange(1, n-1, 2))
    return pylops.Kronecker(pylops.Identity(n), odd @ GradientOperator(n))

def even_gradient_y(n):
    even = pylops.Restriction(n - 1, np.arange(0, n-1, 2))
    return pylops.Kronecker(even @ GradientOperator(n), pylops.Identity(n))

def odd_gradient_y(n):
    odd = pylops.Restriction(n - 1, np.arange(1, n-1, 2))
    return pylops.Kronecker(odd @ GradientOperator(n), pylops.Identity(n))