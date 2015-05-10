"""Explicit-Implicit for contact in explicit dynamics

Reference
---------
Zhong, ZH. (1993) Finite element procedures for contact-impact problems
"""

from numpy import dot, zeros, empty
from numpy.linalg import solve

from mfpy.contact.nodenodepair import NodeNodePair
from mfpy.contact.nodesegmentpair import NodeSegmentPair

def calculate_cdm(contact_pairs, ntdm, snpm):
    """Contact Contact DOF map

    Maps global DOFs to local 'contact DOFs'. These DOFs are coupled due to contact and
    are solved for implicitly

    Parameters
    ----------
    pen_active_list : list of NodeSegmentPair
        List of all NodeSegmentPairs that are currently *penetrating*
    ntdm : array
        Node translational DOF map
    snpm : array
        Segment to Node Pair map

    Returns
    -------
    cdm : dict int -> int
        Contact DOF map
    """

    contact_dofs = set()
    for pair in contact_pairs:
        # Slave node DOFs
        contact_dofs.update(ntdm[pair.slave_id])

        # Master node DOFs
        if isinstance(pair, NodeNodePair):
            contact_dofs.update(ntdm[pair.master_id])

        # Master segment DOFs
        elif isinstance(pair, NodeSegmentPair):
            for node in snpm[pair.master_id]:
                contact_dofs.update(ntdm[node])

    # Assign contact DOFs
    cdm = {j:i for i,j in enumerate(sorted(contact_dofs))}

    return cdm

def contact_explicit_implicit(contact_pairs, ntdm, snpm, sem, elements, dt, M, R, v, a):
    """Modify the acceleration vector to account for contact using the explicit-implicit algorithm.

    Parameters
    ----------
    contact_pairs : list of NodeNodePair/NodeSegmentPair
    ntdm : list of array
        Node translational DOF map
    snpm : array
        Segment Node Pair map
    sem : array
        Segment Element Map
    elements : list of Element
        List of elements to consider
    dt : float
        Time-step
    M : array
        Global mass vector
    R : array
        Global residual vector
    v : array
        Global velocity vector
    a : array
        Global acceleration vector

    Returns
    -------
    a : array
        Modified acceleration vectors

    Notes
    -----
    See sections 9.1 to 9.3
    Here we follow the explicit-implicit split described in section 9.3
    """

    contact_pairs = [pair for pair in contact_pairs if pair.d_min < 0]

    # Filter the active list and build the contact DOF map
    cdm = calculate_cdm(contact_pairs, ntdm, snpm)

    # Number of lagrange multipliers (forces)
    num_lm = len(contact_pairs)
    if (num_lm) == 0: return a

    # Number of contact DOFs
    num_cd = len(cdm)
    if num_cd == 0: return a

    # Weight matrix
    B = zeros((num_cd, num_lm))

    # Penetration matrix
    P = zeros(num_lm)

    # Coupled residual vector
    R_cpl = empty(num_cd)

    # Couple mass vector
    M_cpl = empty(num_cd)

    for pair_id, pair in enumerate(contact_pairs):

        # Hitting (slave) node weight in B matrix
        # Equation (9.2.8a)
        hit_dofs = ntdm[pair.slave_id]
        contact_hit_dofs = [cdm[dof] for dof in hit_dofs]
        B[contact_hit_dofs, pair_id] = -pair.normal

        # Coupled residual and mass vectors
        R_cpl[contact_hit_dofs] = R[hit_dofs]
        M_cpl[contact_hit_dofs] = M[hit_dofs]


        # NODE -> SEGMENT Contact
        if isinstance(pair, NodeSegmentPair):
            element_id, element_segment_id = sem[pair.master_id]

            # Get element
            element = elements[element_id]
            local_segment_nodes = element.get_segment_local_nodes(element_segment_id)
            segment_nodes = snpm[pair.master_id]

            # Calculate shape function weights
            weights = element.calc_N(*pair.xi)

            # Weights for the segment only
            phi = [weights[i] for i in local_segment_nodes]

            # Calculate velocity of target point
            v_target = zeros(2)
            for target_local_node_id, target_node_id in enumerate(segment_nodes):
                target_dofs = ntdm[target_node_id]
                contact_target_dofs = [cdm[dof] for dof in target_dofs]

                # Equations (9.2.14)
                # Also note equation (9.2.8b) to explain lack of minus sign
                B[contact_target_dofs, pair_id] = pair.normal * phi[target_local_node_id]

                # Coupled residual and mass vectors
                R_cpl[contact_target_dofs] = R[target_dofs]
                M_cpl[contact_target_dofs] = M[target_dofs]

                # Velocity of target point
                v_target += phi[target_local_node_id] * v[target_dofs]


        # NODE -> NODE Contact
        else:
            # Target (master) node weight in B matrix
            target_dofs = ntdm[pair.master_id]
            contact_target_dofs = [cdm[dof] for dof in target_dofs]
            B[contact_target_dofs, pair_id] = pair.normal

            # Coupled residual and mass vectors
            R_cpl[contact_target_dofs] = R[target_dofs]
            M_cpl[contact_target_dofs] = M[target_dofs]

            # Velocity of target (master) node
            v_target = v[target_dofs]

        # Penetration Matrix
        # Equation (9.2.27)
        v_hit = v[hit_dofs]
        p0 = -pair.d_min
        P[pair_id] = (p0 + dot(v_target - v_hit, pair.normal)*dt)/(dt*dt)


    # Solve for forces
    # Equation (9.3.3a)
    BTMinvB =  B.transpose().dot(B / M_cpl[:,None])
    BTMinvR =  B.transpose().dot(R_cpl / M_cpl)
    F = solve(BTMinvB, BTMinvR+P)

    # Zero out adhesion forces
    F[F<0] = 0

    # Solve for accelerations
    # Equation (9.3.3b)
    a_cpl = (R_cpl - B.dot(F)) / M_cpl

    # Scatter results to global acceleration vector
    for (global_id, contact_id) in cdm.items():
        a[global_id] = a_cpl[contact_id]

    return a
