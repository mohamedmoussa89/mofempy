__author__ = 'Mohamed Moussa'

from numpy import array, empty, zeros, hstack
from numpy.linalg import norm, det, inv as inverse

import mofem.coordsys as cs
from mofem.elements.element import Element
from mofem.dof import DOF, DOFSet

class Truss(metaclass=Element):
    dof_sig = [DOFSet(DOF.X, DOF.Y),
               DOFSet(DOF.X, DOF.Y)]
    param_names = ["area"]


    def __init__(self, nodes, enm, mat, **params):
        from mofem.elements.element import check_params_valid
        assert(check_params_valid(Truss, params))

        self.nodes = [nodes[i] for i in enm]
        self.enm = enm
        self.mat = mat
        self.params = params


    @staticmethod
    def calc_N(xi):
        """
        Shape Function
        N_I
        """
        return array([1-xi, xi])


    @staticmethod
    def get_local_coord_on_edge(edge, xi):
        if edge == (0,1): return xi
        if edge == (1,0): return 1-xi
        raise ValueError("Invalid edge provided")


    def calc_lumped_mass(self):
        rho = self.mat.params["rho"]
        A = self.params["area"]

        X1 = self.nodes[0][0:2]
        X2 = self.nodes[1][0:2]
        L0 = norm(X2 - X1)

        m = rho*A*L0
        return m * array([0.5, 0.5, 0.5, 0.5])


    def calc_internal_force(self, nodes_curr, u):
        # Reference nodes
        X1 = self.nodes[0][0:2]
        X2 = self.nodes[1][0:2]

        # Current nodes
        x1 = nodes_curr[0]
        x2 = nodes_curr[1]

        # Displacements
        u1 = u[0:2]
        u2 = u[2:4]

        # Original length
        L0 = norm(X2 - X1)

        # Current length
        L = norm(x2-x1)

        # Calculate strain
        strain = (L-L0)/L0

        # Calculate stress
        A = self.params["area"]
        stress = self.mat.calc_stress_1d(strain)

        # Internal force
        fint_loc = stress * A * array([-1, 1])

        # Convert to global internal force
        local_cs = cs.create(x2-x1)
        T = cs.calc_transform(local_cs, cs.global_cs_2d)
        fint = zeros(4)
        fint[0:2] = T.dot(fint_loc[0])
        fint[2:4] = T.dot(fint_loc[1])

        return fint



def test():
    from mofem.materials.linearelastic import LinearElastic

    # Test create
    nodes = [array((0,0)), array((1,0))]
    enm = [0,1]
    mat = LinearElastic(lmbda=0,mu=0.5,rho=1)
    elem = Truss(nodes, enm, mat, area=1)

    # Test internal force
    u = [0,0,1,0]
    fint = elem.calc_internal_force(u)
    print(fint)

    # Test lumped mass matrix
    M = elem.calc_lumped_mass()
    print(M)

if __name__ == "__main__":
    test()