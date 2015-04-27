__author__ = 'Mohamed Moussa'

from numpy import array, dot, zeros

def calculate_cdm(active_list, ntdm):
    """Contact DOF Map"""

    # Get contact DOFs
    contact_dofs = set()
    for ap in active_list:
        if (ap.d_min > 0): continue
        contact_dofs.update(ntdm[ap.node_id])
        for node in ap.segment:
            contact_dofs.update(ntdm[node])

    cdm = {j:i for i,j in enumerate(sorted(contact_dofs))}

    return cdm

def calculate_contact_accel_ei(active_list, enm, ntdm, cdm, elements, threshold, dt, M, R, v, a):
    from numpy import dot, zeros, empty, diag, hstack, vstack, concatenate
    from numpy.linalg import solve, eigvals

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

    for ap_id, ap in enumerate(active_list):
         # No contact
        if (ap.d_min > 0): continue

        # Calculate weights
        element = elements[ap.element_id]
        coords = element.get_local_coord_on_edge(ap.element_segment, ap.xi)
        weights = element.calc_N(*coords)

        # Slave node weight
        slave_dofs = ntdm[ap.node_id]
        contact_slave_dofs = [cdm[dof] for dof in slave_dofs]
        B[contact_slave_dofs, ap_id] = -ap.normal

        # Coupled residual and mass vectors
        R_cpl[contact_slave_dofs] = R[slave_dofs]
        M_cpl[contact_slave_dofs] = M[slave_dofs]

        # Master node weights
        # Also calculate velocity of target node
        v_target = zeros(2)
        for master_elem_node, master_node in zip(ap.element_segment , ap.segment):
            master_dofs = ntdm[master_node]
            contact_master_dofs = [cdm[dof] for dof in master_dofs]

            B[contact_master_dofs, ap_id] = ap.normal * weights[master_elem_node]

            # Coupled residual and mass vectors
            R_cpl[contact_master_dofs] = R[master_dofs]
            M_cpl[contact_master_dofs] = M[master_dofs]

            v_target += weights[master_elem_node] * v[master_dofs]

        # Penetration Matrix
        v_slave = v[slave_dofs]
        p0 = -ap.d_min
        P[ap_id] = (p0 + dot(v_target - v_slave, ap.normal)*dt)/(dt*dt)


    # Solve for forces
    BTMinvB =  B.transpose().dot(B / M_cpl[:,None])
    BTMinvR =  B.transpose().dot(R_cpl / M_cpl)
    F = solve(BTMinvB, BTMinvR+P)

    # Zero out adhesion forces
    F[F<0] = 0

    # Solve for accelerations
    a_cpl = (R_cpl - B.dot(F)) / M_cpl


    for (global_id, contact_id) in cdm.items():
        a[global_id] = a_cpl[contact_id]

    return a


def get_phi(weights, elem_segment):
    return array([weights[local_node_id] for local_node_id in elem_segment])


def calc_phi_bar(phi):
    return sum([p**2 for p in phi])


def get_target_node_masses(ntdm, M, segment):
    return array([M[ntdm[node_id][0]] for node_id in segment])

def calc_defence_node_mass(phi, phi_bar, target_node_masses):
    return sum(phi*target_node_masses/phi_bar)


def calc_defence_node_residual(phi, target_node_masses, defence_node_mass, target_node_R):
    return sum( [defence_node_mass * p * tnR / tnM for (p,tnR,tnM) in zip(phi, target_node_R, target_node_masses)] )


def calc_defence_node_velocity(phi, target_node_velocities):
    return sum([ p*tnv for p,tnv in zip(phi, target_node_velocities) ])


def calculate_contact_force_dna(active_list, ntdm, elements, M, R, v, dt, t):

    f_cont = zeros(len(R))

    for ap in active_list:
        if (ap.d_min > 0):
            ap.contact_force = 0
            continue

        # Get shape function weights for this element
        element = elements[ap.element_id]
        coords = element.get_local_coord_on_edge(ap.element_segment, ap.xi)
        weights = element.calc_N(*coords)

        # Surface normal
        n = ap.normal

        # Calculate defence node mass
        phi = get_phi(weights, ap.element_segment)
        phi_bar = calc_phi_bar(phi)
        target_node_masses = get_target_node_masses(ntdm, M, ap.segment)
        defence_node_mass = calc_defence_node_mass(phi, phi_bar, target_node_masses)

        # Get hitting node mass
        hitting_node_mass = M[ntdm[ap.node_id][0]]

        # Get target node residuals and velocities
        target_nodes_R = [ R[ntdm[nid]] for nid in ap.segment]
        target_nodes_v =  [ v[ntdm[nid]] for nid in ap.segment]

        # hitting and defence residuals in normal direction
        hitting_node_R = dot(R[ntdm[ap.node_id]], ap.normal)
        defence_node_R = dot(calc_defence_node_residual(phi, target_node_masses, defence_node_mass, target_nodes_R),
                             ap.normal)

        # Hitting and defence node velocities in normal direction
        hitting_node_v = dot(v[ntdm[ap.node_id]], ap.normal)
        defence_node_v = dot(calc_defence_node_velocity(phi, target_nodes_v), ap.normal)

        M2 = hitting_node_mass
        v2 = hitting_node_v
        F2 = hitting_node_R

        M1 = defence_node_mass
        v1 = defence_node_v
        F1 = defence_node_R

        p = -ap.d_min

        # Force increment at hitting node
        delta_f = M1 * M2 * (  F2/M2 - F1/M1 + v2/dt - v1/dt - p/(dt*dt) ) / (M1 + M2)

        ap.contact_force += -delta_f

        if (ap.contact_force < 0):
            ap.contact_force = 0

        # Distribute
        f_cont[ ntdm[ap.node_id] ] = ap.contact_force*ap.normal
        for i, node_id in enumerate(ap.segment):
            f_cont[ntdm[node_id]] =  - target_node_masses[i] * phi[i]/phi_bar/defence_node_mass * ap.contact_force * ap.normal


    return f_cont





















def test():
    pass

if __name__ == "__main__":
    test()