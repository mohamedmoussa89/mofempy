"""Penalty method for contact in explicit dynamics"""

from numpy import array, zeros

from .nodenodepair import NodeNodePair
from .nodesegmentpair import NodeSegmentPair

def contact_penalty_method(num_dof, contact_pairs, enm, ntdm, sem, elements, penalty_stiff, t):

    fcont = zeros(num_dof)

    for pair in contact_pairs:

        # Calculate penalty force, in direction of normal
        contact_force = penalty_stiff * abs(pair.d_min) * pair.normal

        # Apply force on slave node
        fcont[ntdm[pair.slave_id]] += contact_force

        # NODE -> SEGMENT CONTACT
        # Distribute the contact force to the target nodes on the segment
        if isinstance(pair, NodeSegmentPair):
            # Get element
            element_id, element_segment_id = sem[pair.master_id]
            element = elements[element_id]

            # Get segment/target nodes
            local_segment_nodes = element.get_segment_local_nodes(element_segment_id)
            target_nodes = [enm[element_id][i] for i in local_segment_nodes]

            # Shape function weights for the segment
            weights = element.calc_N(*pair.xi)
            phi = [weights[i] for i in local_segment_nodes]

            for i, node_id in enumerate(target_nodes):
                fcont[ntdm[node_id]] +=  - phi[i] * contact_force

        # NODE -> NODE CONTACT
        # Apply force on the master node (negative direction)
        elif isinstance(pair, NodeNodePair):
            fcont[ntdm[pair.master_id]] -= contact_force

    return fcont
