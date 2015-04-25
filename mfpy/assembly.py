__author__ = 'Mohamed Moussa'

from numpy import array, empty, zeros, hstack
from numpy.linalg import det, inv as inverse

from mfpy.dof import DOFSet

def calculate_nds(elem_node_maps, elem_types, num_nodes=None):
    """Calculate DOF Set for every node in the element node map (ENM)"""
    if not num_nodes:
        num_nodes = max([max(nodes) for nodes in elem_node_maps]) + 1
    nds = [DOFSet() for i in range(num_nodes)]
    for (typ, nodes) in zip(elem_types, elem_node_maps):
        for (l_nid, g_nid) in enumerate(nodes):  # local/global node IDs
            nds[g_nid] |= typ.dof_sig[l_nid]
    return nds


def calculate_ndm(node_dof_sets):
    """Calculate Node DOF Map
    The NDM is the mapping of node IDs to DOF IDs"""
    dof_id = 0
    ndm = []
    for dof_set in node_dof_sets:
        ndm.append( tuple(range(dof_id, dof_id + len(dof_set))) )
        dof_id += len(dof_set)

    return ndm


def get_number_dofs(node_dof_maps):
    return max([max(dm) for dm in node_dof_maps]) + 1


def calculate_edm(elem_node_maps, node_dof_maps):
    """Calculate Element DOF Map
    The EDM is the mapping of element IDs to DOF IDs"""
    def nodes_to_dofs(nodes):
        """Converts list of node IDs to list of dof IDs using node_dof_map"""
        return tuple(dof_id for node_id in nodes for dof_id in node_dof_maps[node_id])

    return [nodes_to_dofs(nodes) for nodes in elem_node_maps]


def calculate_ntdm(node_dof_sets, elem_node_maps, node_dof_maps):
    """Calculate Node Translational DOF Map
    The NTDM is the mapping of the node ID to the translational DOF only"""
    from mfpy.dof import DOF

    local_td = [ds.indexOf([DOF.X, DOF.Y]) for ds in node_dof_sets]

    return [[ndm[td] for td in ltd] for (ndm,ltd) in zip(node_dof_maps, local_td)]


def calculate_etdm(elem_node_maps, node_trans_dof_maps, num_elements):
    """Calculate Element Translational DOF Map
    The ETDM is the mapping of the element ID to the translational DOF only"""

    elem_trans_dof_maps = []
    for eid in range(num_elements):
        elem_ntdm = [node_trans_dof_maps[nid] for nid in elem_node_maps[eid]]
        elem_etdm = [x for entdm in elem_ntdm for x in entdm]
        elem_trans_dof_maps.append(elem_etdm)

    return elem_trans_dof_maps


def gather_element_vector(edm, global_vec, local_vec):
    for local_dof in range(len(local_vec)):
        global_dof = edm[local_dof]
        local_vec[local_dof] = global_vec[global_dof]


def scatter_element_vector(edm, local_vec, global_vec):
    for local_dof in range(len(local_vec)):
        global_dof = edm[local_dof]
        global_vec[global_dof] += local_vec[local_dof]


def updated_node_positions(ntdm, nodes_prev, u):
    return [n + u[ntdm[nid]] for (nid,n) in enumerate(nodes_prev)]


def assemble_lumped_mass(edm, elements, global_M):
    global_M *= 0

    for elem_id, elem in enumerate(elements):
        M = elem.calc_lumped_mass()
        scatter_element_vector(edm[elem_id], M, global_M)

    return global_M


def assemble_internal_force(enm, edm, nodes_curr, elements, global_u, global_fint):
    global_fint *= 0

    for elem_id, elem in enumerate(elements):
        u = empty(elem.num_dof)
        elem_nodes_curr = [nodes_curr[nid] for nid in enm[elem_id]]
        gather_element_vector(edm[elem_id], global_u, u)
        fint = elem.calc_internal_force(elem_nodes_curr, u)
        scatter_element_vector(edm[elem_id], fint, global_fint)


def test():
    pass
if __name__ == "__main__":
    test()

