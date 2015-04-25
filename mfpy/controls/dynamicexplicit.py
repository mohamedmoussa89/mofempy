__author__ = 'Mohamed Moussa'

from numpy import array, empty, float64

from mfpy.controls.control import Control

class DynamicExplicit(metaclass=Control):
    param_names = ["dt", "t_end"]

    @staticmethod
    def run(nodes, elements, materials, vel_ic, vel_bc, fext_bc, kin_output, **params):
        from mfpy.assembly import calculate_nds, calculate_ndm, calculate_edm
        from mfpy.assembly import calculate_ntdm
        from mfpy.assembly import assemble_lumped_mass, assemble_internal_force
        from mfpy.assembly import get_number_dofs, updated_node_positions
        from mfpy.boundcond import generate_bc_pairs, apply_bc_pairs_on_vec

        from mfpy.topology import find_boundary_segment_map, find_boundary_nodes
        from mfpy.contact import find_new_active_nodes, update_active_node_distances, remove_inactive_nodes
        from mfpy.contact import calculate_contact_accel_ei, calculate_cdm
        from mfpy.contact import calculate_contact_force_dna

        from numpy import zeros

        dt = params["dt"]
        t_end = params["t_end"]

        # Calculate DOF maps
        enm = [e.enm for e in elements]
        elem_types = [type(e) for e in elements]
        nds = calculate_nds(enm, elem_types, len(nodes))
        ndm = calculate_ndm(nds)
        edm = calculate_edm(enm, ndm)

        # Translation DOF maps
        ntdm = calculate_ntdm(nds, enm, ndm)

        # Prepare boundary conditions
        vel_bc_pairs = generate_bc_pairs(nds, ndm, vel_bc)
        fext_bc_pairs = generate_bc_pairs(nds, ndm, fext_bc)

        # Kinematic vectors
        num_dof = get_number_dofs(ndm)
        u = zeros(num_dof)
        v = zeros(num_dof)
        a = zeros(num_dof)

        # Global lumped mass matrix
        M = empty(num_dof)
        assemble_lumped_mass(edm, elements, M)

        # Global internal force vector
        fint = empty(num_dof)

        # External force vector
        fext = zeros(num_dof)
        apply_bc_pairs_on_vec(fext_bc_pairs, fext)

        # Reference node positions
        nodes_ref = nodes
        nodes_prev = nodes
        nodes_curr = nodes

        # Initial acceleration
        assemble_internal_force(enm, edm, nodes_curr, elements, u, fint)
        R = (fext - fint)
        a = 1/M * R

        # Initial velocity
        apply_bc_pairs_on_vec(generate_bc_pairs(nds, ndm, vel_ic), v)

        # Contact
        boundary_segment_map = find_boundary_segment_map(enm)
        boundary_nodes = find_boundary_nodes(boundary_segment_map)
        active_list = []

        t = 0
        kin_output.add(t, u, v, a)

        while t + dt <= t_end:
            t += dt

            v += dt/2 * a
            apply_bc_pairs_on_vec(vel_bc_pairs, v)
            u += dt * v

            # Current node positions
            nodes_prev = [n for n in nodes_curr]
            nodes_curr = updated_node_positions(ntdm, nodes_ref, u)

            assemble_internal_force(enm, edm, nodes_curr, elements, u, fint)


            d_threshold = 2*max(abs(v))*dt
            active_list = update_active_node_distances(active_list, nodes_curr, elements)
            active_list = find_new_active_nodes(active_list, nodes_curr, elements, boundary_nodes, boundary_segment_map, d_threshold)
            active_list = remove_inactive_nodes(active_list, d_threshold)

            # CONTACT - Defence node algorithm
            fcont = 0*calculate_contact_force_dna(active_list, ntdm, elements, M, R, v, dt, t)

            R = (fext - fint + fcont)
            a = 1/M * R

            # CONTACT - Calculate contact DOF map
            cdm = calculate_cdm(active_list, ntdm)

            # CONTACT - Modify acceleration using Explicit-Implicit algorithm
            a = calculate_contact_accel_ei(active_list, enm, ntdm, cdm, elements, d_threshold, dt, M, R, v, a)

            # ------------------------------------------------------------------------------------------

            v += dt/2*a
            apply_bc_pairs_on_vec(vel_bc_pairs, v)

            kin_output.add(t, u, v, a)

        kin_output.finalize()

def VerticalTrussProblem():
    from mfpy.elements.quad import Quad
    from mfpy.elements.truss import Truss
    from mfpy.materials.linearelastic import LinearElastic
    from mfpy.dof import DOF
    from mfpy.boundcond import BC
    from mfpy.postproc.output import KinematicOutput

    dt = 0.01

    nodes = [(0.,0.), (4.,0.), (4.,1.), (0.,1.),
             (2.,3.0), (2.,2.0)]
    nodes = [array(n, dtype=float64) for n in nodes]

    materials = [ LinearElastic(lmbda=0.0, mu=1.0, rho=1.0) ]

    elements = [ Quad(nodes, [0,1,2,3], materials[0], thickness=10),
                 Truss(nodes, [4,5], materials[0], area=10) ]

    vel_ic = [BC((4,5), DOF.Y, -1.0)]

    vel_bc = [BC((0,1), (DOF.X,DOF.Y), 0),
              BC((2,3,4,5), DOF.X, 0)]

    fext_bc = []

    kin_output = KinematicOutput(dt)

    DynamicExplicit.run(nodes, elements, materials, vel_ic, vel_bc, fext_bc, kin_output, dt=dt, t_end=5.0)

    from pylab import plot, show
    plot(kin_output.t, kin_output.u[:,11]+1)
    plot(kin_output.t, kin_output.u[:,9]+2.)
    plot(kin_output.t, kin_output.u[:,7]*0.5 + kin_output.u[:,5]*0.5 )

    show()

def HorizontalTrussProblem():
    from mfpy.elements.quad import Quad
    from mfpy.elements.truss import Truss
    from mfpy.materials.linearelastic import LinearElastic
    from mfpy.dof import DOF
    from mfpy.boundcond import BC
    from mfpy.postproc.output import KinematicOutput

    dt = 0.01

    nodes = [(0.,0.), (4.,0.), (4.,1.5), (0.,1.),
             (1.,2.05), (3.,2.05)]
    nodes = [array(n, dtype=float64) for n in nodes]

    materials = [ LinearElastic(lmbda=0.0, mu=1.0, rho=1.0) ]

    elements = [ Quad(nodes, [0,1,2,3], materials[0], thickness=10),
                 Truss(nodes, [4,5], materials[0], area=10) ]

    vel_ic = [BC((4,5), DOF.Y, -1)]

    vel_bc = [BC((0,1), (DOF.X,DOF.Y), 0),
              BC((2,3,4,5), DOF.X, 0)]

    fext_bc = []

    kin_output = KinematicOutput(dt)

    DynamicExplicit.run(nodes, elements, materials, vel_ic, vel_bc, fext_bc, kin_output, dt=dt, t_end=10.)

    from pylab import plot, show
    plot(kin_output.t, kin_output.u[:,11]+1.05)
    plot(kin_output.t, kin_output.u[:, 9]+1.05)
    plot(kin_output.t, kin_output.u[:,7]*0.75 + (kin_output.u[:,5]+0.5)*0.25 )
    plot(kin_output.t, kin_output.u[:,7]*0.25 + (kin_output.u[:,5]+0.5)*0.75 )

    show()

if __name__ == "__main__":
    #VerticalTrussProblem()
    HorizontalTrussProblem()
