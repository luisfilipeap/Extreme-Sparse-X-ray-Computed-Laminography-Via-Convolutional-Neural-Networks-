from math import floor, radians, tan, atan2, sin, cos
import numpy as np


class InlineScanningSetup:
    """
    This class represents a Compute Tomography (CT) inline scanning setup composed of a fixed X-ray source and detector
    for imaging objects passing on a conveyor belt.

    Attributes
    ---------
    geometry_matrix : ndarray
        It holds the position of each element (object, X-ray source, and detector) in a vector space at each projection
        acquisition. The data is encapsulated according to the specifications of ASTRA Toolbox.
    Methods
    -------
    get_geometry_matrix()
        Returns the geometry_matrix built by the class constructor.
    """

    def __init__(self, alpha, detector_cells, number_of_projections, object_size):
        """
        It creates a new instance of the class InlineScanningSetup.
        :param alpha: fan-beam opening angle in the X-ray source;
        :param detector_cells: number of detector elements;
        :param number_of_projections: number of X-ray projections acquired during the object movement;
        :param object_size: number W of pixels of the W x W reconstruction grid;
        acquisition.
        """

        h = (detector_cells / 2) / tan(radians(alpha / 2))

        # src: the ray source
        srcX = np.linspace(-detector_cells / 2, detector_cells / 2, num=number_of_projections)
        srcZ = np.linspace(h - object_size / 2, h - object_size / 2, num=number_of_projections)
        srcY = np.linspace(0,0, num=number_of_projections)

        # d :  the center of the detector
        dX = np.linspace(-detector_cells / 2, detector_cells / 2, num=number_of_projections)
        dZ = np.linspace(-object_size / 2, -object_size / 2, num=number_of_projections)
        dY = np.linspace(0, 0, num=number_of_projections)

        # u :  the vector between the centers of detector pixels 0 and 1
        uX = np.linspace(1, 1, num=number_of_projections)
        uY = np.linspace(0, 0, num=number_of_projections)
        uZ = np.linspace(0, 0, num=number_of_projections)
        # pdb.set_trace()

        vX = np.linspace(0, 0, num=number_of_projections)
        vY = np.linspace(1, 1, num=number_of_projections)
        vZ = np.linspace(0, 0, num=number_of_projections)


        self.geometry_matrix = np.column_stack((srcX, srcY, srcZ,  dX, dY, dZ, uX, uY, uZ, vX, vY, vZ))

    def get_geometry_matrix(self):
        """
        It provides access to the inline CT setup built in the constructor method.
        :return: the attribute geometry_matrix
        """
        return self.geometry_matrix