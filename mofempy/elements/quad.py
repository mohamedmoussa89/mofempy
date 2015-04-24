__author__ = 'Mohamed Moussa'

from itertools import product
from math import sqrt

from numpy import array, empty, zeros, hstack
from numpy.linalg import det, inv as inverse

from mofem.elements.element import Element
from mofem.dof import DOF, DOFSet

class Quad(metaclass=Element):
    """2D Quadrilateral Element"""
    dof_sig = [DOFSet(DOF.X, DOF.Y),
               DOFSet(DOF.X, DOF.Y),
               DOFSet(DOF.X, DOF.Y),
               DOFSet(DOF.X, DOF.Y)]
    param_names = ["thickness"]

    gauss_weights = list(product([1,1],repeat=2))
    gauss_loc = list(product([-1/sqrt(3), 1/sqrt(3)],repeat=2))

    def __init__(self, nodes, enm, mat, **params):
        from mofem.elements.element import check_params_valid
        assert(check_params_valid(Quad, params))

        self.nodes = [nodes[i] for i in enm]
        self.enm = enm
        self.mat = mat
        self.params = params


    @staticmethod
    def calc_N(xi, eta):
        """
        Shape Function
        N_I
        """
        return 1/4 * array([(1-xi)*(1-eta),
                            (1+xi)*(1-eta),
                            (1+xi)*(1+eta),
                            (1-xi)*(1+eta)])

    @staticmethod
    def get_local_coord_on_edge(edge, xi):
        xi = 2*xi - 1
        eta = xi
        if edge == (0,1): return ( xi,  -1)
        if edge == (1,2): return (  1, eta)
        if edge == (2,3): return (-xi,   1)
        if edge == (3,0): return ( -1,-eta)
        raise ValueError("Invalid edge provided")

    @staticmethod
    def calc_dNdXi(xi, eta):
        """
        Shape function derivatives w.r.t. element coordinates
        dN_I/dXi_i
        """
        return 1/4 * array([[-(1-eta),-(1-xi)],
                            [ (1-eta),-(1+xi)],
                            [ (1+eta), (1+xi)],
                            [-(1+eta), (1-xi)]])

    @staticmethod
    def calc_J(X, dNdXi):
        """
        Jacobian
        J[i,j] = X[I,i] * dNdXi[I,i]
        """
        J = empty((2,2))
        for (i,j) in list(product(range(0,2), repeat=2)):
            J[i,j] = 0
            for I in range(0,4):
                J[i,j] += dNdXi[I,i] * X[2*I+j]
        return J

    @staticmethod
    def calc_dNdX(J, dNdXi):
        """
        Shape function derivatives w.r.t. spatial coordinates
        dN_I/dX_i = dN_I/dXi_j dXi_j/dX_i
        Note that the inverse Jacobian is defined as
        J'_ij = dXi_j/dX_i
        """
        Jinv = inverse(J)
        dNdX = empty((4,2))
        for I,i in product(range(0,4), range(0,2)):
            dNdX[I,i] = 0
            for j in range(0,2):
                dNdX[I,i] += dNdXi[I,j] * Jinv[i,j]
        return dNdX

    @staticmethod
    def calc_B(dNdX):
        """
        Strain Displacement Matrix
        B_ijkJ = 1/2*(dN_J/dX_j del_ik +  dN_J/dX_i del_jk)
        """
        B = empty((3,8))
        for I in range(0,4):
            B[:, 2*I:2*I+2] = array([[dNdX[I,0],         0],
                                     [        0, dNdX[I,1]],
                                     [dNdX[I,1], dNdX[I,0]]])
        return B


    def calc_lumped_mass(self):

        X = hstack(self.nodes)
        M = zeros(Quad.num_dof)

        rho = self.mat.params["rho"]
        h = self.params["thickness"]

        points = [(-1,-1), (1,-1), (1,1), (-1,1)]
        for i, (eta, xi) in enumerate(points):
            dNdXi = Quad.calc_dNdXi(xi, eta)
            J     = Quad.calc_J(X, dNdXi)
            M[2*i:2*i+2] = rho*h*det(J)

        return M


    def calc_internal_force(self, nodes_curr, u):
        from numpy import hstack

        h = self.params["thickness"]
        mat = self.mat
        X = hstack(self.nodes)

        fint = zeros(8)
        for (w, xi) in Quad.gauss_points:
            w1, w2  = w
            xi, eta = xi

            dNdXi = Quad.calc_dNdXi(xi, eta)
            J     = Quad.calc_J(X, dNdXi)
            dNdX  = Quad.calc_dNdX(J, dNdXi)
            B     = Quad.calc_B(dNdX)
            detJ  = det(J)

            epsilon = B.dot(u)
            sigma = mat.calc_stress_2d(epsilon)

            fint += w1*w2*B.transpose().dot(sigma)*h*detJ

        return fint


def test():
    from mofem.materials.linearelastic import LinearElastic

    # Test create
    nodes = [array((0,0)), array((1,0)), array((1,1)), array((0,1))]
    enm = [0,1,2,3]
    mat = LinearElastic(lmbda=0,mu=1.0,rho=1)
    elem = Quad(nodes, enm, mat, thickness=1)

    # Test internal force
    u = array([0,0, 0,0, 0,-1, 0,-1])
    fint = elem.calc_internal_force(u)
    print(fint)

    # Test lumped mass matrix
    M = elem.calc_lumped_mass()
    print(M)

if __name__ == "__main__":
    test()
