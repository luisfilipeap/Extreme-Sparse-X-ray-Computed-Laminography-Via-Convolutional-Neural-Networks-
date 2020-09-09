import pylops
import astra
import numpy as np
from matplotlib import pyplot as plt
from tomopy import shepp2d
from InlineCTSetup.gradient_operators import gradient_x, gradient_y
from InlineCTSetup.proximal_operators import prox_l1, prox_l2s
from InlineCTSetup.proximal_solvers import PDHG, operator_2norm

# optomo fix
def _matvec(self,v):
    return self.FP(v.ravel(), out=None).ravel()
def _rmatvec(self,s):
    return self.BP(s.ravel(), out=None).ravel()
astra.OpTomo._matvec = _matvec
astra.OpTomo._rmatvec = _rmatvec

#x = np.arange(9).reshape((3,3))
x = shepp2d(256)[0]/255
vol_geom = astra.create_vol_geom(*x.shape)
proj_geom = astra.create_proj_geom('parallel', 1, x.shape[0], np.linspace(0,np.pi,55))
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

W = astra.optomo.OpTomo(proj_id)
W = W*(1/operator_2norm(W, max_iter=20))

# create sino
p = W @ x.ravel()

# reconstruct gradient by minimizing
#   ||Wx - p||_2^2 + a||grad x||_1

#  || (      W    )         ( p ) ||
#  || (           )  x  -   (   ) ||
#  || (   a grad  )         ( 0 ) ||_1/2||.||22, a||.||1
#
# so g = ||. - [p,0]||_1/2||.||22, a||.||1
a = 0.0025

# gradient operator
D = pylops.VStack([gradient_x(x.shape[0]),
                   gradient_y(x.shape[0])])
#D = D*(1/operator_2norm(W, max_iter=20))
# the stacked operator we will use in lin ADMM
A = pylops.VStack([W, D])

#proximal of zero function is identity
prox_f = lambda v, l: v

# proximal of g
prox_l2l1 = lambda v, l: np.concatenate([prox_l2s(v[:p.size], l), prox_l1(v[p.size:], a*l)])
p0 = np.concatenate([p, np.zeros(D.shape[0])])
prox_g = lambda v,l: p0 + prox_l2l1(v - p0, l)

op_norm = 1.1 * operator_2norm(A, 20)
print(op_norm)
sigma = tau = 0.9**0.5/op_norm

rec = PDHG(prox_f, prox_g, A,
           np.zeros(x.size),
           np.zeros(A.shape[0]),
           sigma=sigma, tau=tau, theta=1,
           max_iter=200)

plt.figure()
plt.imshow(rec.reshape(x.shape),cmap = "gray")
plt.colorbar()
# plt.figure()
# plt.plot(res[1:])
plt.show()