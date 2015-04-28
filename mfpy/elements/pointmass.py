"""Simple point mass"""

from numpy import array, zeros, ones

from mfpy.dof import DOF, DOFSet
from .element import Element

class PointMass(metaclass=Element):
    dof_sig = [DOFSet(DOF.X, DOF.Y)]
    param_names = ["mass"]

    def __init__(self, nodes, enm, mat, **params):
        from mfpy.elements.element import check_params_valid
        assert(check_params_valid(PointMass, params))

        self.nodes = nodes[0]
        self.enm = enm
        self.mat = None
        self.params = params

    def calc_lumped_mass(self):
        m = self.params["mass"]
        return array([m,m])

    def calc_internal_force(self, nodes_curr, u):
        return zeros(2)






