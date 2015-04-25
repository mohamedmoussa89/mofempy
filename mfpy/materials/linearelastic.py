__author__ = 'Mohamed Moussa'

from mfpy.materials.material import Material
from mfpy.tensorutils import trace, I, IoI, II

class LinearElastic(metaclass = Material):
    param_names =["lmbda" , "mu", "rho"]

    def __init__(self, **params):
        from mfpy.materials.material import check_params_valid
        assert(check_params_valid(LinearElastic, params))
        self.params = params


    def calc_stress_1d(self, strain):
        lmbda = self.params["lmbda"]
        mu = self.params["mu"]
        E = mu*(3*lmbda + 2*mu) / (lmbda + mu)
        return E*strain


    def calc_stress_2d(self, strain):
        lmbda = self.params["lmbda"]
        mu = self.params["mu"]
        return lmbda*trace(strain)*I + 2*mu*strain


    def tangent_matrix(self, params):
        lmbda = self.params["lmbda"]
        mu = self.params["mu"]
        return lmbda*IoI + 2*mu*II