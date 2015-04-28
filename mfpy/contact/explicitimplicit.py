"""Explicit-Implicit for contact in explicit dynamics

Reference
---------
Zhong, ZH. (1993) Finite element procedures for contact-impact problems,
New York, Oxford University Press
"""

from numpy import dot, zeros, empty
from numpy.linalg import solve

from .contactsearch import get_penetrating_active_nodes

def calculate_cdm(active_list, ntdm):
    """Contact Contact DOF map

    Maps global DOFs to 'contact DOFs'. These DOFs are coupled due to contact and
    are solved for implicitly

    Parameters
    ----------
    active_list : list of NodeSegmentPair
        List of all NodeSegmentPairs that are currently considered active
    ntdm : list of array
        Node translational DOF map

    Returns
    -------
    cdm : dict int -> int
        Contact DOF map
    """

    # Filter out non-penetrating nodes
    active_list = get_penetrating_active_nodes(active_list)

    contact_dofs = set()
    for pair in active_list:
        contact_dofs.update(ntdm[pair.node_id])
        for node in pair.global_seg:
            contact_dofs.update(ntdm[node])

    cdm = {j:i for i,j in enumerate(sorted(contact_dofs))}

    return cdm

def contact_explicit_implicit(active_list, ntdm, cdm, elements, dt, M, R, v, a):
    """Modify the acceleration vector to account for contact using the explicit-implicit algorithm.

    Parameters
    ----------
    active_list : list of NodeSegmentPair
        List of all NodeSegmentPairs that are currently considered active
    ntdm : list of array
        Node translational DOF map
    cdm : dict int -> int
        Contact DOF map
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

    active_list = get_penetrating_active_nodes(active_list)

    # Number of lagrange multipliers
    num_lm = len(active_list)
    if (num_lm) == 0: return a

    # Number of contact DOFs
    num_cd = len(cdm.keys())
    if num_cd == 0: return a

    # Weight matrix
    B = zeros((num_cd, num_lm))

    # Penetration matrix
    P = zeros(num_lm)

    # Coupled residual vector
    R_cpl = empty(num_cd)

    # Couple mass vector
    M_cpl = empty(num_cd)

    for pair_id, pair in enumerate(active_list):

        # Calculate weights
        element = elements[pair.elem_id]
        coords = element.get_local_coord_on_edge(pair.local_seg, pair.xi)
        weights = element.calc_N(*coords)

        # Hitting node weight in B matrix
        # Equation (9.2.8a)
        hit_dofs = ntdm[pair.node_id]
        contact_hit_dofs = [cdm[dof] for dof in hit_dofs]
        B[contact_hit_dofs, pair_id] = -pair.normal

        # Coupled residual and mass vectors
        R_cpl[contact_hit_dofs] = R[hit_dofs]
        M_cpl[contact_hit_dofs] = M[hit_dofs]

        # Target node weights
        # Also calculate velocity of target point
        v_target = zeros(2)
        for target_local_node_id, target_node_id in zip(pair.local_seg , pair.global_seg):
            target_dofs = ntdm[target_node_id]
            contact_target_dofs = [cdm[dof] for dof in target_dofs]

            # Equations (9.2.14)
            # Also note equation (9.2.8b) to explain lack of minus sign
            B[contact_target_dofs, pair_id] = pair.normal * weights[target_local_node_id]

            # Coupled residual and mass vectors
            R_cpl[contact_target_dofs] = R[target_dofs]
            M_cpl[contact_target_dofs] = M[target_dofs]

            # Velocity of target point
            v_target += weights[target_local_node_id] * v[target_dofs]

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
