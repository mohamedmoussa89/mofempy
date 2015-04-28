"""Defence Node Algorithm for contact in explicit dynamics

Reference
---------
Zhong, ZH. (1993) Finite element procedures for contact-impact problems,
New York, Oxford University Press
"""

from numpy import array, zeros, dot

def get_phi(weights, local_seg):
    """ Gets the shape function vector for the segment

    Parameters
    ----------
    weights : array
        Shape functions evaluated at the projection point
    local_seg : tuple of int
        Tuple that describes the segment based on the element local node IDs

    Returns
    -------
    - : array
        Weight vector for the segment

    Notes
    -----
    See section 9.5.2, e.g. equations (9.5.1) and (9.5.2).
    phi is simply the shape function vector for the nodes of the segments
    """
    return array([weights[local_node_id] for local_node_id in local_seg])


def calculate_phi_bar(phi):
    """ Calculates phi_bar from phi

    Parameters
    ----------
    phi : array
        Shape function vector for a segment. See: get_phi

    Returns
    -------
    - : float
        phi_bar

    Notes
    -----
    See section 9.5.2, equation (9.5.14). Here k = 2, as in equation (9.5.17)
    """
    return sum([v**2 for v in phi])


def get_target_node_masses(ntdm, M, global_seg):
    """Gets the mass vector for the given segment described using global node IDs

    The mass is determined from the global mass vector and the mapping from nodes to
    translational DOFs (i.e. x/y)

    TODO Does it have to be translational DOFS?? What happens in the case of a beam element?

    Parameters
    ----------
    ntdm : list of array
        Node Translational DOF Map
    M : array
        Global mass vector
    global_seg : tuple of int
        Tuple that describes the segment based on the global node IDs

    Returns
    - : array
        Mass vector for the segment
    """
    return array([M[ntdm[node_id][0]] for node_id in global_seg])


def calculate_defence_node_mass(phi, phi_bar, target_node_masses):
    """Calculate mass of defence node

    Parameters
    ----------
    phi : array
    phi_bar : float
    target_node_masses : array

    Returns
    -------
    - : float
        Mass of defence node

    Notes
    -----
    See section 9.5.2, equation (9.5.19)
    """
    return sum(phi*target_node_masses/phi_bar)


def calc_defence_node_velocity(phi, target_node_velocities):
    """Calculate velocity of defence node

    Parameters
    ----------
    phi : array
    target_node_velocities : array

    Returns
    -------
    - : array
        Velocity of defence node

    Notes
    -----
    See section 9.5.2, equation (9.5.1)
    """
    return sum([ p*tnv for p,tnv in zip(phi, target_node_velocities) ])


def calculate_defence_node_residual(phi, target_node_masses, defence_node_mass, target_node_R):
    """
        Calculate the residual (sum of internal and external forces) at the defence node

        Parameters
        ----------
        phi : array
        target_node_velocities : array
        target_node_masses : array
        defence_node_mass : float
        target_node_R : list of array

        Returns
        -------
        - : array
            Defence node residual

        Notes
        -----
        See section 9.5.2, equation (9.5.6)
    """
    return sum( [defence_node_mass * p * tnr / tnm for (p,tnr,tnm) in zip(phi, target_node_R, target_node_masses)] )


def contact_defence_node(active_list, ntdm, elements, M, R, v, dt, t):
    """Calculate the global contact force vector using the defence node algorithm

    Normal contact only (no frictional effects).

    Parameters
    ----------
    active_list : list of NodeSegmentPair
        List of all NodeSegmentPairs that are currently considered active
    ntdm : list of array
        Node translational DOF map
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

    for pair in active_list:

        # If no penetration, zero out contact force and move to next pair
        if (pair.d_min > 0):
            pair.contact_force = 0
            continue

        # Get shape function weights for this element
        element = elements[pair.elem_id]
        local_edge_coord = element.get_local_coord_on_edge(pair.local_seg, pair.xi)
        weights = element.calc_N(*local_edge_coord)

        # Surface normal
        n = pair.normal

        # Calculate defence node mass
        phi = get_phi(weights, pair.local_seg)
        phi_bar = calculate_phi_bar(phi)
        target_node_masses = get_target_node_masses(ntdm, M, pair.global_seg)
        defence_node_mass = calculate_defence_node_mass(phi, phi_bar, target_node_masses)

        # Get hitting node mass
        hitting_node_mass = M[ntdm[pair.node_id][0]]

        # Get target node residuals and velocities
        target_nodes_R =  [ R[ntdm[nid]] for nid in pair.global_seg ]
        target_nodes_v =  [ v[ntdm[nid]] for nid in pair.global_seg ]

        # Hitting and defence residuals in normal direction
        hitting_node_R = dot(R[ntdm[pair.node_id]], pair.normal)
        defence_node_R = dot(calculate_defence_node_residual(phi, target_node_masses, defence_node_mass, target_nodes_R),pair.normal)

        # Hitting and defence node velocities in normal direction
        hitting_node_v = dot(v[ntdm[pair.node_id]], pair.normal)
        defence_node_v = dot(calc_defence_node_velocity(phi, target_nodes_v), pair.normal)

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

        try: pair.contact_force += -delta_f
        except AttributeError: pair.contact_force = -delta_f

        if (pair.contact_force < 0):
            pair.contact_force = 0

        print(pair.contact_force)

        # Distribute to contact force vector
        # Equations (9.5.8) and (9.5.17)
        fcont[ ntdm[pair.node_id] ] = pair.contact_force*pair.normal
        for i, node_id in enumerate(pair.global_seg):
            fcont[ntdm[node_id]] +=  - target_node_masses[i] * phi[i]/phi_bar/defence_node_mass * pair.contact_force * pair.normal


    return fcont