__author__ = 'Mohamed Moussa'

class BC(object):
    def __init__(self, node_ids, dofs, val):
        from collections import Iterable

        if not isinstance(node_ids, Iterable): node_ids = [node_ids]
        if not isinstance(dofs, Iterable): dofs = [dofs]

        self.node_ids = node_ids
        self.dofs = dofs
        self.val = val

    def get_dof_ids(self, nds, ndm):
        from itertools import product

        dof_ids = []
        for (node_id, dof) in product(self.node_ids, self.dofs):
            node_dof_id = nds[node_id].indexOf(dof)
            if node_dof_id is not None:
                dof_ids.append(ndm[node_id][node_dof_id])

        return dof_ids

def generate_bc_pairs(nds, ndm, bc_list):
    return [(dof_id, bc.val) for bc in bc_list for dof_id in bc.get_dof_ids(nds, ndm)]

def apply_bc_pairs_on_vec(bc_pairs, vec):
    for (dof_id, val) in bc_pairs:
        vec[dof_id] = val

def test():
    from mofem.elements.truss import Truss
    from mofem.materials.linearelastic import LinearElastic
    from mofem.assembly import calculate_nds, calculate_ndm, calculate_edm
    from mofem.dof import DOF
    from numpy import array

    elements = [ Truss([], [0,1], None, area=1),
                 Truss([], [1,2], None, area=1),
                 Truss([], [2,3], None, area=1)]

    bc_list = [BC([0,1,2,3], [DOF.X,DOF.RZ], 0)]

    enm = [e.enm for e in elements]
    elem_types = [type(e) for e in elements]
    nds = calculate_nds(enm, elem_types, 4)
    ndm = calculate_ndm(nds)

    bc_pairs = generate_bc_pairs(nds, ndm, bc_list)
    print(bc_pairs)

if __name__ == "__main__":
    test()