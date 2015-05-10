"""Defence Node Algorithm for contact in explicit dynamics

Reference
---------
Zhong, ZH. (1993) Finite element procedures for contact-impact problems,
New York, Oxford University Press
"""

from numpy import array, zeros, dot, vstack, newaxis, sum as arraysum

from mfpy.contact.nodenodepair import NodeNodePair
from mfpy.contact.nodesegmentpair import NodeSegmentPair


def contact_defence_node(contact_pairs, enm, ntdm, sem, elements, M, R, v, dt, t):
    """Calculate the global contact force vector using the defence node algorithm

    Normal contact only (no frictional effects).

    Parameters
    ----------
    contact_pairs : list of NodeNodePair/NodeSegmentPair
        List of all NodeSegmentPairs that are currently considered active
    enm : array
        Element Node Map
    ntdm : array
        Node Translational DOF map
    sem :
        Segment Element Map
    elements : list of Element
        List of elements to consider
    M : array
        Global mass vector
    R : array
        Global residual vector (sum of internal and external forces)
    v : array
        Global velocity vector
    dt : float
        Time-step

    Returns
    -------
    fcont : array
        Global contact force vector

    Notes
    -----
    See sections 9.5.1 to 9.5.3.
    """

    fcont = zeros(len(R))

    for pair in contact_pairs:

        # Hitting (slave) node mass, residual and velocity in normal direction
        hitting_node_mass = M[ntdm[pair.slave_id][0]]
        hitting_node_R = dot(R[ntdm[pair.slave_id]], pair.normal)
        hitting_node_v = dot(v[ntdm[pair.slave_id]], pair.normal)

        # NODE -> SEGMENT CONTACT
        # Need to calculate target node properties
        if isinstance(pair, NodeSegmentPair):
            # Get element
            element_id, element_segment_id = sem[pair.master_id]
            element = elements[element_id]
            local_segment_nodes = element.get_segment_local_nodes(element_segment_id)

            # Get the target nodes for this segment
            target_nodes = [enm[element_id][i] for i in local_segment_nodes]

            # Calculate shape function weights
            weights = element.calc_N(*pair.xi)

            # Shape function weights for the segment
            phi = array([weights[i] for i in local_segment_nodes])

            # Calculate phi_bar
            # See section 9.5.2, equation (9.5.14). Here k = 2, as in equation (9.5.17)
            phi_bar = sum([v**2 for v in phi])

            # Gather target node masses
            target_node_masses = array( [M[ntdm[node_id][0]] for node_id in target_nodes] )

            # Calculate defence node mass
            # See section 9.5.2, equation (9.5.19)
            defence_node_mass = sum(phi*target_node_masses / phi_bar)

            # Get target nodes residuals and velocities
            target_nodes_R =  vstack([ R[ntdm[node_id]] for node_id in target_nodes ])
            target_nodes_v =  vstack([ v[ntdm[node_id]] for node_id in target_nodes ])

            # Defence node residual in normal direction
            # See section 9.5.2, equation (9.5.6)
            defence_node_R = defence_node_mass * arraysum((target_nodes_R / target_node_masses) * phi[:, newaxis], axis = 0)
            defence_node_R = dot(defence_node_R, pair.normal)

            # Defence node velocity in normal direction
            # See section 9.5.2, equation (9.5.1)
            defence_node_v = arraysum(target_nodes_v * phi[:, newaxis], axis=0)
            defence_node_v = dot(defence_node_v, pair.normal)

        # NODE -> NODE CONTACT
        elif isinstance(pair, NodeNodePair):
            defence_node_mass = M[ntdm[pair.master_id][0]]
            defence_node_v = dot(v[ntdm[pair.master_id]], pair.normal)
            defence_node_R = dot(R[ntdm[pair.master_id]], pair.normal)


        # Match variables names in reference
        M2 = hitting_node_mass
        v2 = hitting_node_v
        F2 = hitting_node_R

        M1 = defence_node_mass
        v1 = defence_node_v
        F1 = defence_node_R

        p = -pair.d_min

        # Force increment at hitting node
        # Equation (9.5.35)
        delta_f = M1 * M2 * (  F2/M2 - F1/M1 + v2/dt - v1/dt - p/(dt*dt) ) / (M1 + M2)

        # Add contact increment to this pair
        try: pair.contact_force += -delta_f
        except AttributeError: pair.contact_force = -delta_f

        if (pair.contact_force < 0):
            pair.contact_force = 0

        # Distribute to contact force vector
        # Equations (9.5.8) and (9.5.17)
        fcont[ ntdm[pair.slave_id] ] += pair.contact_force*pair.normal

        # NODE -> SEGMENT CONTACT
        if isinstance(pair, NodeSegmentPair):
            for i, node_id in enumerate(target_nodes):
                fcont[ntdm[node_id]] +=  -phi[i]*target_node_masses[i]/phi_bar/defence_node_mass * pair.contact_force * pair.normal

        # NODE -> NODE CONTACT
        elif isinstance(pair, NodeNodePair):
            fcont[ ntdm[pair.master_id] ] += pair.contact_force*pair.normal


    return fcont