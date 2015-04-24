__author__ = 'Mohamed Moussa'

from numpy import array

class Element(type):
    # Element type information that needs to be provided
    dof_sig = []
    param_names = []
    gauss_weights = []
    gauss_loc = []

    # Element type information that is automatically determined
    dim = 0
    num_dof = 0
    num_nodes = 0
    gauss_points = []

    def __new__(meta, name, bases, dct):
        from functools import reduce
        from mofem.dof import DOF

        # Calculate number of degrees of freedom
        dct["num_dof"] = sum([len(ds) for ds in dct["dof_sig"]])

        # Calculate number of nodes
        dct["num_nodes"] = len(dct["dof_sig"])

        # Create gauss point/weight pairs
        try: dct["gauss_points"] = list(zip(dct["gauss_weights"], dct["gauss_loc"]))
        except KeyError: dct["gauss_points"] = []

        # Determine if 3D element (if element has Z, RX or RY)
        dct["dim"] = 2
        dofs = reduce(set.union, dct["dof_sig"])
        if (DOF.Z in dofs or DOF.RX in dofs or DOF.RY in dofs):
            dim = 3

        return super(Element, meta).__new__(meta, name, bases, dct)


def check_params_valid(typ, params):
    return set(typ.param_names) == set(params.keys())