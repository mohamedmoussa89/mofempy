"""Penalty method for contact in explicit dynamics"""

from numpy import zeros

from .nodesegmentpair import get_penetrating_active_pairs

def contact_penalty_method(active_list, num_dof, enm, ntdm, elements, penalty_stiff, t):

    active_list = get_penetrating_active_pairs(active_list)

    fcont = zeros(num_dof)

    if not active_list: return fcont

    #print("TIME =",t)
    for pair in active_list:
        #print("\tNODE =", pair.node_id)

        # Calculate penalty force, in direction of normal
        contact_force = penalty_stiff * (-pair.d_min) * pair.normal

        # Unilateral contact - apply force only on hitting node
        fcont[ntdm[pair.node_id]] = contact_force
        #print("\t\tFCONT =", fcont)

        # Bilateral contact - apply force on segment
        element = elements[pair.elem_id]
        local_edge_coord = element.get_local_coord_on_edge(pair.local_seg, pair.xi)
        weights = element.calc_N(*local_edge_coord)

        #print("\t\tSCATTER")
        for i, node_id in enumerate(enm[pair.elem_id]):
            fcont[ntdm[node_id]] +=  - weights[i] * contact_force
            #print("\t\tFCONT =", fcont)
    #print("")

    return fcont
