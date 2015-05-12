from mfpy.dof import DOF, DOFSet
from mfpy.elements.quad import Quad

class QuadRI(Quad):
    """
    2D Quadrilateral Element
    Reduced Integration
    """
    dof_sig = [DOFSet(DOF.X, DOF.Y),
               DOFSet(DOF.X, DOF.Y),
               DOFSet(DOF.X, DOF.Y),
               DOFSet(DOF.X, DOF.Y)]
    param_names = ["thickness"]

    gauss_weights = [(2,2)]
    gauss_loc = [(0,0)]