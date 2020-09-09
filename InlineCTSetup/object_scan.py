from InlineCTSetup.inline_setup_3D import *
import astra
import time
from imageio import imread, imwrite
from matplotlib import pyplot as plt
from scipy import misc

from tomopy import shepp2d
import pylops
from InlineCTSetup.gradient_operators import gradient_x, gradient_y, gradient_z
#from gradient_operators import even_gradient, odd_gradient
from InlineCTSetup.proximal_operators import prox_l1, prox_l2s
from InlineCTSetup.proximal_solvers import PDHG, operator_2norm


# optomo fix
def _matvec(self,v):
    return self.FP(v.ravel(), out=None).ravel()
def _rmatvec(self,s):
    return self.BP(s.ravel(), out=None).ravel()
astra.OpTomo._matvec = _matvec
astra.OpTomo._rmatvec = _rmatvec

class ScanningObject:
    """
    This class defines an inline scanning geometry and executes image reconstructions.
    Attributes
    ----------
    proj_geom   : dict
        It holds the projection geometry to be used;
    setup       : ndarray
        It holds the scanning geometry to be used;
    vol_geom    : dict
        It holds the characteristics of the reconstruction volume;
    Methods
    -------
    run(phantom_param, rec_algorithm_param='SIRT_CUDA', n_iterations_param=100)
        It executes an image reconstruction using the projections acquired in the inline setup.
    """

    def __init__(self, alpha_param, n_cells_param, n_proj_param, rec_size_param=256):
        """
        It creates a new instance of the class ScanningExecution.
        :param alpha_param: fan-beam opening angle of the X-ray source used in the inline CT setup;
        :param n_cells_param: number of detector elements used in the inline CT setup;
        :param n_proj_param: number of X-ray projections aquired during the object movement;
        :param rec_size_param: number W of pixels of the W x W reconstruction grid;

        acquisition.
        """

        self.vol_geom = astra.create_vol_geom(128, 128, 10)

        self.setup = InlineScanningSetup(alpha=alpha_param, detector_cells=n_cells_param,
                                         number_of_projections=n_proj_param, object_size=rec_size_param)

        self.proj_geom = astra.create_proj_geom('cone_vec', n_cells_param, n_cells_param, self.setup.get_geometry_matrix())

    def run(self, phantom_param, rec_algorithm_param='SIRT3D_CUDA', n_iterations_param=100):
        """
        It executes an image reconstruction using the projections acquired in the inline setup.
        :param phantom_param: 3D volume of the phantom that should be used to simulate the acquisition of projections from
        real object;
        :param rec_algorithm_param: reconstruction algorithm to be used. The option available are: SIRT_CUDA and FBP_CUDA;
        :param n_iterations_param: number of iterations to be used in case of iterative reconstructions;
        :return: a dictionary containing the reconstructed image into 'rec' index, the reconstruction time into 'time' index,
        and the acquired sinogram into the 'sino' index;
        """


        #proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
        proj_id, proj_data = astra.create_sino3d_gpu(phantom_param, self.proj_geom, self.vol_geom)

        #plt.figure("sino")
        #plt.imshow(proj_data[:,5,:])
        #plt.show()

        rec_id = astra.data3d.create('-vol', self.vol_geom)


        if rec_algorithm_param == 'SIRT3D_CUDA':
            cfg = astra.astra_dict(rec_algorithm_param)
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = proj_id
            alg_id = astra.algorithm.create(cfg)

            start_time = time.time()
            astra.algorithm.run(alg_id, n_iterations_param)
            elapsed_time = time.time() - start_time

            astra.algorithm.delete(alg_id)

            output = {'rec': astra.data3d.get(rec_id), 'time': elapsed_time, 'sino': proj_data}

        elif rec_algorithm_param == 'TV3D':
            projector_id = astra.create_projector('cuda3d', self.proj_geom, self.vol_geom)

            W = astra.optomo.OpTomo(projector_id)
            W = W * (1 / operator_2norm(W, max_iter=20))

            phantom_param = phantom_param/255
            p = W @ phantom_param.ravel()

            #a = 0.0025
            a = 0.0025

            # gradient operator
            i, j, k = gradient_x(10,128,128), gradient_y(10,128,128), gradient_z(10,128,128)
            D = k
            #D = pylops.VStack([j, k])
            # D = D*(1/operator_2norm(W, max_iter=20))
            # the stacked operator we will use in lin ADMM
            A = pylops.VStack([W, D])

            # proximal of zero function is identity
            prox_f = lambda v, l: v

            # proximal of g
            prox_l2l1 = lambda v, l: np.concatenate([prox_l2s(v[:p.size], l), prox_l1(v[p.size:], a * l)])
            p0 = np.concatenate([p, np.zeros(D.shape[0])])
            prox_g = lambda v, l: p0 + prox_l2l1(v - p0, l)

            op_norm = 1.1 * operator_2norm(A, 20)
            print(op_norm)
            sigma = tau = 0.9 ** 0.5 / op_norm

            start_time = time.time()
            rec = PDHG(prox_f, prox_g, A,
                       np.zeros(phantom_param.size),
                       np.zeros(A.shape[0]),
                       sigma=sigma, tau=tau, theta=1,
                       max_iter=200)

            rec = rec.reshape(phantom_param.shape)
            elapsed_time = time.time() - start_time
            output = {'rec': rec, 'time': elapsed_time, 'sino': proj_data}




        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)


        return output


if __name__ == '__main__':

    #test code by running scanning_object.py
    src = "D:\\Datasets\\demo_data_plates\\plate_00000\\"

    plane = np.zeros((10, 128, 128))
    for k in range(10):
        i = imread(src + 'slice_{}.png'.format(k), pilmode='F')
        plane[k,:, :] = i

    setup = ScanningObject(alpha_param=30, n_cells_param=256, n_proj_param=4, rec_size_param=128)
    out = setup.run(plane, rec_algorithm_param = "SIRT3D_CUDA")

    final = out['rec']

    plt.figure("REC")
    plt.imshow(final[5, :,:], cmap="gray")
    print(np.min(final))
    print(np.max(final))

    #for k in range(10):

        #imwrite("gt_{}.png".format(k), plane[k,:,:])

    plt.figure("PHANTOM")
    plt.imshow(plane[5,:,:], cmap="gray")

    plt.show()

   # plt.show()