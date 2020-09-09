import numpy as np
import pylops
from scipy.sparse.linalg import lsqr

def prox_box(v, l, bounds):
    return np.clip(v, *bounds)

def prox_l1(v, l):
    # proximal operator of ||.||_1
    return np.clip(v-l, 0, None) + np.clip(v+l, None, 0)

def prox_l2(v, l):
    # proximal operator of  ||.||_2
    n = np.linalg.norm(v)
    if n >= l:
        return (1-l/n)*v
    else:
        return 0*v

def prox_l2s(v, l):
    # proximal operator of  1/2||.||_2^2
    return 1/(1+l)*v

def prox_f_orthogonal(prox_f, A, v, l):
    # proximal operator of f(A), given the proximal operator
    # of f and given the orthogonal operator A (i.e. A.H @ A = A @ A.H = I)
    return A.H @ prox_f(A @ v, l)

def prox_f_semi_orthogonal(prox_f, A, v, l):
    # proximal operator of f(A), given the proximal operator
    # of f and given the semi-orthogonal operator A (i.e. A @ A.H = I)
    Av = A @ v
    return v + A.H @ (prox_f(Av, l) - Av)

def prox_lstsq2(A, b, v, l, max_iter=30):
    # proximal operator of 1/2||A.-b||_2^2

    # we need to minimize
    # 1/2 ||Ax-b||_2^2 + 1/(2l)||x-v||_2^2
    #
    # It can be rewritten as one least squares problem
    #
    # 1 || (      A    )         (    b      ) ||2
    # - || (           )  x  -   (           ) ||
    # 2 || (1/sqrt(l) I)         (1/sqrt(l) v) ||2
    #
    # which we solve using lsqr
    M = pylops.VStack([A, 1/(l**(1/2))*pylops.Identity(A.shape[1])])
    q = np.concatenate([b, 1/(l**(1/2))*v])

    return lsqr(M, q, iter_lim=max_iter)[0]