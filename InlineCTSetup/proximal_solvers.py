import numpy as np
from tqdm import tqdm
from proximal_operators import prox_l1, prox_f_semi_orthogonal, prox_box
from gradient_operators import (even_gradient_x,
                                odd_gradient_x,
                                even_gradient_y,
                                odd_gradient_y)

def proximal_gradient(grad_f, prox_g, x0, l, max_iter=50, verbose=True):
    """
    Solve

        argmin_x f(x) + g(x)

    given the gradient of f and the proximal operator of g
    """
    x = x0
    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)
    for i in loop:
        x = prox_g(x - l*grad_f(x), l)
    return x

def linearized_ADMM(prox_f, prox_g, A,
                    x0, z0, u0,
                    l_f, l_g,
                    max_iter=50,
                    verbose=True):
    """
    Solve
    
        argmin_x f(x) + g(Ax)

    given the proximal operators of f and g, and the linear operator A
    using the Linearized Alternating Direction Method of Multipliers
    """
    x = x0
    z = z0
    u = u0
    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)
    for i in loop:
        Ax = A @ x
        x = prox_f(x - (l_f/l_g)*A.H @ (Ax - z + u),l_f)
        z = prox_g(Ax + u,l_g)
        u = u + Ax - z
    return x

def PDHG(prox_f, prox_g, A,
         x0, y0,
         sigma, tau, theta,
         max_iter=50,
         verbose=True):
    """
    Solve
    
        argmin_x f(x) + g(Ax)

    given the proximal operators of f and g, and the linear operator A
    using Primal Dual Hybrid Gradient
    """
    x = x0
    y = y0
    z = x0
    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)
    for i in loop:
        v = y + sigma * A @ z 
        y = v - prox_g(v, sigma)
        x_prev = x
        x = prox_f(x - tau * A.H * y, tau)
        z = x + theta*(x - x_prev)
    return x

def GFB(grad_f, prox_g, z, l, m, max_iter=50, verbose=True, f=None, g=None):
    """
    Solve
    
        argmin_x f(x) + sum_i g_i(x)

    given the gradient of f and the proximal operators of g_i
    using Generalised Forward Backward splitting
    """
    x = np.average(z, axis=0)
    n_regs = z.shape[0]
    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    res_f = []
    res_g = []
    xs = []
    for i in loop:
        for j in range(n_regs):
            z[j, :] = z[j, :] + l*(prox_g[j](2*x - z[j, :] - m*grad_f(x), m*n_regs) - x)
        x = np.average(z, axis=0)
        xs.append(x)
        if not f is None:
            res_f.append(f(x))
        if not g is None:
            res_g.append([g[i](z[i]) for i in range(len(g))])
    return x, res_f, np.array(res_g), xs


def projected_GFB(grad_f, prox_g, z, l, m, bounds, max_iter=50, verbose=True, f=None, g=None):
    """
    Solve
    
        argmin_x f(x) + sum_i g_i(x)
        such that x in bounds

    given the gradient of f and the proximal operators of g_i
    using Generalised Forward Backward splitting and projecting
    in the bounds after each update
    """
    x = np.average(z, axis=0)
    n_regs = z.shape[0]
    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    res_f = []
    res_g = []
    xs = []
    for i in loop:
        for j in range(n_regs):
            z[j, :] = z[j, :] + l*(prox_g[j](2*x - z[j, :] - m*grad_f(x), m*n_regs) - x)
        z = np.clip(z, *bounds)
        x = np.average(z, axis=0)
        xs.append(x)
        if not f is None:
            res_f.append(f(x))
        if not g is None:
            res_g.append([g[i](z[i]) for i in range(len(g))])
    return x, res_f, np.array(res_g), xs


def TV_min_2D_GFB(grad_f, a, x0, l, m, bounds=None, max_iter=50, verbose=True):
    """
    Solve 

        argmin_x f(x) + a*||grad x||_1

    given the gradient of f and constant a. It uses GFB by splitting
    the grad operator into semi orthogonal operators, which allows
    for a closed form of their proximal operators.

    Optionally, box constraints can be added by specifying the bounds parameter.
    In this case, the minimization problem is augmented to
    
        argmin_x f(x) + a*||grad x||_1 + indicator_of_bounds(x)

    which is solved using the same method.
    """

    # semi orthogonal splitting of gradient operator
    G1 = even_gradient_x(x0.shape[0])
    G2 = even_gradient_y(x0.shape[0])
    G3 = odd_gradient_x(x0.shape[0])
    G4 = odd_gradient_y(x0.shape[0])

    # the proximal operators of ||G_i x||_1
    prox_g1 = lambda v, l: prox_f_semi_orthogonal(prox_l1, 2**(-0.5)*G1, v, a*2**(0.5)*l)
    prox_g2 = lambda v, l: prox_f_semi_orthogonal(prox_l1, 2**(-0.5)*G2, v, a*2**(0.5)*l)
    prox_g3 = lambda v, l: prox_f_semi_orthogonal(prox_l1, 2**(-0.5)*G3, v, a*2**(0.5)*l)
    prox_g4 = lambda v, l: prox_f_semi_orthogonal(prox_l1, 2**(-0.5)*G4, v, a*2**(0.5)*l)
    prox_g = [prox_g1, prox_g2, prox_g3, prox_g4]
    if bounds is not None:
        prox_g5 = lambda v, l: prox_box(v, l, bounds)
        prox_g.append(prox_g5)

    z = np.repeat(x0.ravel()[np.newaxis], len(prox_g), axis=0)
    rec = GFB(grad_f, prox_g, z, l, m, max_iter=max_iter, verbose=verbose)[0]
    return rec.reshape(x0.shape)


def TV_min_2D_projected_GFB(grad_f, a, x0, l, m, bounds=None, max_iter=50, verbose=True):
    """
    Solve 

        argmin_x f(x) + a*||grad x||_1

    given the gradient of f and constant a. It uses GFB by splitting
    the grad operator into semi orthogonal operators, which allows
    for a closed form of their proximal operators.

    Optionally, box constraints can be added by specifying the bounds parameter.
    In this case, the minimization problem is augmented to
    
        argmin_x f(x) + a*||grad x||_1
        such that x in bounds

    which is solved using projected_GFB
    """

    # semi orthogonal splitting of gradient operator
    G1 = even_gradient_x(x0.shape[0])
    G2 = even_gradient_y(x0.shape[0])
    G3 = odd_gradient_x(x0.shape[0])
    G4 = odd_gradient_y(x0.shape[0])

    # the proximal operators of ||G_i x||_1
    prox_g1 = lambda v, l: prox_f_semi_orthogonal(prox_l1, 2**(-0.5)*G1, v, a*2**(0.5)*l)
    prox_g2 = lambda v, l: prox_f_semi_orthogonal(prox_l1, 2**(-0.5)*G2, v, a*2**(0.5)*l)
    prox_g3 = lambda v, l: prox_f_semi_orthogonal(prox_l1, 2**(-0.5)*G3, v, a*2**(0.5)*l)
    prox_g4 = lambda v, l: prox_f_semi_orthogonal(prox_l1, 2**(-0.5)*G4, v, a*2**(0.5)*l)
    prox_g = [prox_g1, prox_g2, prox_g3, prox_g4]
    
    z = np.repeat(x0.ravel()[np.newaxis], len(prox_g), axis=0)
    if bounds is None:
        rec = GFB(grad_f, prox_g, z, l, m, max_iter=max_iter, verbose=verbose)[0]
    else:
        rec = projected_GFB(grad_f, prox_g, z, l, m, bounds=bounds, max_iter=max_iter, verbose=verbose)[0]
    return rec.reshape(x0.shape)


def operator_2norm(A, max_iter):
    """
    Calculate the 2-norm of a linear operator
    """
    if hasattr(A, "adjoint"):
        B = A.H @ A
    else:
        B = A.T @ A
    b = np.random.normal(0,1,B.shape[1])
    b = b/np.linalg.norm(b)
    for i in range(max_iter):
        b = B @ b
        b = b/np.linalg.norm(b)
    return np.sqrt(np.linalg.norm(B@b)/np.linalg.norm(b))