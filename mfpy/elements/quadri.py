from numpy import empty, array, dot, hstack

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


    def calc_linear_stiffness(self, nodes_curr, u):

        X = hstack(self.nodes)
        t = self.params["thickness"]

        # Area
        # Eqn. (8.4.7)
        A = 0.5*( (X[1,0]-X[3,0])*(X[2,1]-X[0,1]) +
                  (X[2,0]-X[0,0])+(X[3,1]-X[1,1]) )

        # Jacobian (evaluated at xi=eta=0)
        # Eqn. (8.4.7)
        detJ0 = A/4

        # dNdX, part of strain displacement matrix
        # Eqn. (8.4.9)
        bx =  1/(2*A) * array([X[1,1]-X[3,1],
                               X[2,1]-X[0,1],
                               X[3,1]-X[1,1],
                               X[0,1]-X[2,1]])

        by =  1/(2*A) * array([X[3,0]-X[1,0],
                               X[0,0]-X[2,0],
                               X[1,0]-X[3,0],
                               X[2,0]-X[0,0]])

        # Eqn (8.4.11)
        h = array([1, -1, 1, -1])
        hx = dot(h,X[:,0])
        hy = dot(h,X[:,1])
        gamma = 0.25 * (h - hx*bx - hy*by)

        # In reference, element DOF signature is (x1,x2,x3,x4,y1,y2,y3,y4)
        # Here, we build the B matrix for the DOF signature (x1,y1,x2,y2,x3,y3,x4, y4)
        B = empty((3,8))
        for I in range(0,4):
            B[:, 2*I:2*I+2] = array([[bx[I],     0],
                                     [    0, by[I]],
                                     [by[I], bx[I]]])

        # Calculate strain and stress
        epsilon = B.dot(u)
        sigma, C = self.mat.calc_2d(epsilon)

        # Reduced integrated stiffness matrix
        K_ri = 2*B.transpose().dot(C).dot(B)*t*detJ0

        # Stabilizing material parameter
        # Eqn. (8.7.14)
        # Note c^2 = kappa/rho
        alpha = 0.1
        kappa = self.mat.calc_bulk_modulus()
        CQ = 0.5*alpha*kappa*A*(dot(bx,bx) + dot(by,by))

        # Adjust gamma for our DOF signature (see B above)
        gamma_mod = array([[gamma[0],         0, gamma[1],        0, gamma[2],        0, gamma[3],       0],
                           [       0 , gamma[0],        0, gamma[1],        0, gamma[2],        0, gamma[3]]]).transpose()

        # Stabilization stiffness
        # Eqn. (8.7.11)
        K_stab = CQ * A * dot(gamma_mod, gamma_mod.transpose())

        return K_ri + K_stab
