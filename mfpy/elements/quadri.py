from numpy import empty, array, dot, hstack, sum as array_sum

from mfpy.dof import DOF, DOFSet
from mfpy.elements.quad import Quad

class QuadRI(Quad):
    """
    2D Quadrilateral Element
    Reduced Integration with hourglass stabalization

    Reference
    ---------
    Belytschko, T., Liu, W., Moran, B. and Elkhodary, K. (2014).
    Nonlinear finite elements for continua and structures.
    """
    dof_sig = [DOFSet(DOF.X, DOF.Y),
               DOFSet(DOF.X, DOF.Y),
               DOFSet(DOF.X, DOF.Y),
               DOFSet(DOF.X, DOF.Y)]
    param_names = ["thickness"]

    gauss_weights = [(2,2)]
    gauss_loc = [(0,0)]

    @staticmethod
    def calc_A(X):
        # Area
        # Eqn. (8.4.7)
        return ( (X[2]-X[6])*(X[1]-X[5]) + (X[4]-X[0])+(X[7]-X[3]) )


    @staticmethod
    def calc_detJ0(A):
        # Jacobian (evaluated at xi=eta=0)
        # Eqn. (8.4.7)
        return A/4

    @staticmethod
    def calc_bx(X, A):
        # dNdX, part of strain displacement matrix
        # Eqn. (8.4.9)
        bx =  1/(2*A) * array([X[3]-X[7],
                               X[5]-X[1],
                               X[7]-X[3],
                               X[1]-X[5]])
        return bx

    @staticmethod
    def calc_by(X, A):
        # dNdX, part of strain displacement matrix
        # Eqn. (8.4.9)
        by =  1/(2*A) * array([X[6]-X[2],
                               X[0]-X[4],
                               X[2]-X[6],
                               X[4]-X[0]])
        return by

    @staticmethod
    def calc_gamma(X, bx, by):
        # Eqn (8.4.11)
        h = array([1, -1, 1, -1])
        hx = dot(h,X[0::2])
        hy = dot(h,X[1::2])
        gamma = 0.25 * (h - hx*bx - hy*by)

        # Adjust gamma for our DOF signature
        gamma = array([[gamma[0],         0, gamma[1],        0, gamma[2],        0, gamma[3],       0],
                       [       0 , gamma[0],        0, gamma[1],        0, gamma[2],        0, gamma[3]]])

        return gamma.transpose()


    @staticmethod
    def calc_B0(bx,by):
        # Strain-Displacement matrix (evaluated at xi=eta=0)
        # In reference, element DOF signature is (x1,x2,x3,x4,y1,y2,y3,y4)
        # Here, we build the B matrix for the DOF signature (x1,y1,x2,y2,x3,y3,x4, y4)
        B0 = empty((3,8))
        for I in range(0,4):
            B0[:, 2*I:2*I+2] = array([[bx[I],    0],
                                     [    0, by[I]],
                                     [by[I], bx[I]]])
        return B0


    @staticmethod
    def calc_CQ(alpha, A, kappa, bx, by):
        # Stabilizing material parameter
        # Eqn. (8.7.14)
        # Note c^2 = kappa/rho
        return 0.5*alpha*kappa*A*(dot(bx,bx) + dot(by,by))

    def calc_internal_force(self, nodes_curr, u):

        X = hstack(self.nodes)
        t = self.params["thickness"]

        A = self.calc_A(X)
        detJ0 = self.calc_detJ0(A)
        bx = self.calc_bx(X,A)
        by = self.calc_by(X,A)
        gamma = self.calc_gamma(X, bx, by)
        B0 = self.calc_B0(bx, by)

        # Stabilizing material parameter
        alpha = 0.1
        kappa = self.mat.calc_bulk_modulus()
        CQ = self.calc_CQ(alpha, A, kappa, bx, by)

        # Calculate strain and stress
        epsilon = B0.dot(u)
        sigma, C = self.mat.calc_2d(epsilon)

        # Stabilization strain and stress
        # Eqn (8.7.9)
        stab_epsilon = dot(gamma.transpose(), u)
        stab_sigma = CQ * stab_epsilon

        # Reduced integrated internal force
        fint_ri = 2*2*B0.transpose().dot(sigma)*t*detJ0

        # Stabilization internal force
        # Eqn. (8.7.8)
        fint_stab = A * gamma.dot(stab_sigma)

        return fint_ri + fint_stab


    def calc_linear_stiffness(self, nodes_curr, u):

        X = hstack(self.nodes)
        t = self.params["thickness"]

        A = self.calc_A(X)
        detJ0 = self.calc_detJ0(A)
        bx = self.calc_bx(X,A)
        by = self.calc_by(X,A)
        gamma = self.calc_gamma(X, bx, by)
        B0 = self.calc_B0(bx, by)

        # Stabilizing material parameter
        alpha = 0.1
        kappa = self.mat.calc_bulk_modulus()
        CQ = self.calc_CQ(alpha, A, kappa, bx, by)

        # Calculate strain and stress
        epsilon = B0.dot(u)
        sigma, C = self.mat.calc_2d(epsilon)

        # Reduced integrated stiffness matrix
        K_ri = 2*2*B0.transpose().dot(C).dot(B0)*t*detJ0

        # Stabilization stiffness
        # Eqn. (8.7.11)
        K_stab = CQ * A * dot(gamma, gamma.transpose())

        return K_ri + K_stab


def test():
    from numpy.linalg import matrix_rank
    from mfpy.materials.linearelastic import LinearElastic

    # Test create
    nodes = [array((0,0)), array((1,0)), array((1,1)), array((0,1))]
    enm = [0,1,2,3]
    mat = LinearElastic(lmbda=0,mu=1.0,rho=1)
    elem = QuadRI(nodes, enm, mat, thickness=1)

    # Test internal force
    u = array([-1,0,
               +1,0,
               -1,0,
               +1,0])
    fint = elem.calc_internal_force([], u)
    K = elem.calc_linear_stiffness([], u)
    print("Rank =", matrix_rank(K))
    print(fint)
    print(K.dot(u))


    elem = Quad(nodes, enm, mat, thickness=1)
    fint = elem.calc_internal_force([], u)
    K = elem.calc_linear_stiffness([], u)
    print("Rank =", matrix_rank(K))
    print(fint)
    print(K.dot(u))

    # Test lumped mass matrix
    M = elem.calc_lumped_mass()
    #print(M)

if __name__ == "__main__":
    test()