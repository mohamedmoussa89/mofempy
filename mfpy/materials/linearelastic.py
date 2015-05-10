__author__ = 'Mohamed Moussa'

from numpy import array

from mfpy.materials.material import Material
from mfpy.tensor import trace, I, IoI, II

class LinearElastic(metaclass = Material):
    param_names =["lmbda" , "mu", "rho"]

    def __init__(self, **params):
        from mfpy.materials.material import check_params_valid
        assert(check_params_valid(LinearElastic, params))
        self.params = params


    def calc_1d(self, strain):
        lmbda = self.params["lmbda"]
        mu = self.params["mu"]
        E = mu*(3*lmbda + 2*mu) / (lmbda + mu)
        return E*strain, E


    def calc_2d(self, strain):
        lmbda = self.params["lmbda"]
        mu = self.params["mu"]

        stress = lmbda*trace(strain)*I + 2*mu*strain
        tangent = lmbda*IoI + 2*mu*II

        return stress, tangent