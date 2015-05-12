"""Dynamic Explicit Solver with support for contact"""

from numpy import array, empty, float64, zeros, dot, vstack

from mfpy.controls.control import Control
from mfpy.postproc import TemporalOutput
from mfpy.boundcond import generate_bc_pairs, apply_bc_pairs_on_vec
from mfpy.segment import calculate_nsm, calculate_npsm, calculate_snpm, calculate_sem, calculate_ssm

# Assembly
from mfpy.assembly import calculate_enm, calculate_nds, calculate_ndm, calculate_edm, calculate_ntdm
from mfpy.assembly import get_number_dofs, updated_node_positions
from mfpy.assembly import assemble_lumped_mass, assemble_internal_force, assemble_stiffness_sum

# Critical timesteps
from mfpy.controls.criticaltimestep import calculate_nodal_critical_timestep

# Contact
from mfpy.contact import create_surface_pairs
from mfpy.contact import bucketsearch
from mfpy.contact import contact_penalty_method, contact_defence_node, contact_explicit_implicit

def calc_kinetic_energy(m,v):
    return 0.5*sum(m*v*v)

def calc_internal_energy(edm, elements, u):
    from mfpy.assembly import gather_element_vector

    int_energy = 0

    for elem_id, elem in enumerate(elements):
        u_loc = empty(elem.num_dof)
        gather_element_vector(edm[elem_id], u, u_loc)
        K = elem.calc_linear_stiffness([], u_loc)
        int_energy += dot(u_loc, K.dot(u_loc))

    return 0.5*int_energy

class DynamicExplicit(metaclass=Control):
    param_names = ["dt", "t_end"]

    @staticmethod
    def run(nodes, elements, materials, surfaces, pairs, vel_ic, vel_bc, fext_bc, dt_scale, t_end, pp_dt):

        # Calculate DOF maps
        enm = calculate_enm(elements)
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
        nodes_ref = vstack(nodes)
        nodes_curr = array(nodes_ref)

        # Initial acceleration
        assemble_internal_force(enm, edm, nodes_curr, elements, u, fint)
        R = (fext - fint)
        a = 1/M * R

        # Initial velocity
        apply_bc_pairs_on_vec(generate_bc_pairs(nds, ndm, vel_ic), v)

        # Data required for contact algorithms
        node_pairs = [np for surf in surfaces for np in surf]
        npsm = calculate_npsm(node_pairs)               # Node Pair -> Segment ID
        snpm = calculate_snpm(npsm)                     # Segment ID -> Node Pair
        nsm = calculate_nsm(snpm, len(nodes))           # Node -> Parent Segment
        sem = calculate_sem(npsm, elements, len(snpm))  # Segment -> [Element ID, Element Segment ID]
        ssm = calculate_ssm(nsm, snpm)                  # Segment -> Segment Neighbours map
        surface_pairs = create_surface_pairs(npsm, surfaces, pairs)
        contact_pairs = []
        contact_energy = 0

        t = 0

        # Output
        out_vec = TemporalOutput(pp_dt, ["u","v","a"])
        out_vec.add(t, u=u, v=v, a=a)

        out_energy = TemporalOutput(pp_dt, ["kin_e","int_e","cnt_e"])
        out_energy.add(t, kin_e = calc_kinetic_energy(M, v),
                       int_e = calc_internal_energy(edm, elements, u),
                       cnt_e = 0)

        # Estimate critical time-step
        K_sum = assemble_stiffness_sum(num_dof, edm, elements, u)
        dt = dt_scale*calculate_nodal_critical_timestep(M, K_sum)

        while t + dt <= t_end:
            t += dt

            v += dt/2 * a
            apply_bc_pairs_on_vec(vel_bc_pairs, v)
            u += dt * v

            nodes_curr = updated_node_positions(ntdm, nodes_ref, u)

            assemble_internal_force(enm, edm, nodes_curr, elements, u, fint)

            # CONTACT - Search for node-segment pairs that have penetrated
            fcont = 0
            contact_pairs = bucketsearch.one_pass_search(nsm, snpm, sem, ssm, contact_pairs, nodes_curr, elements, surface_pairs, t)

            # DEFENCE NODE
            #fcont = contact_defence_node(contact_pairs, enm, ntdm, sem, elements, M, R, v, dt, t)

            # PENALTY METHOD
            # Update critical timestep if required
            fcont, K_sum_penalty = contact_penalty_method(num_dof, contact_pairs, enm, ntdm, sem, elements, 2000, K_sum)
            dt = dt_scale*calculate_nodal_critical_timestep(M, K_sum_penalty)

            R = (fext - fint + fcont)
            a = 1/M * R

            # CONTACT - Explicit-Implicit Algorithm
            #a = contact_explicit_implicit(contact_pairs, ntdm, snpm, sem, elements, dt, M, R, v, a)

            v += dt/2*a

            apply_bc_pairs_on_vec(vel_bc_pairs, v)

            # Contact energy
            delta_cnt_e = dot(fcont, v*dt)
            contact_energy += delta_cnt_e
            #if abs(delta_cnt_e) > 1e-8:
            #    print("CONTACT at t =", t)
            #    print("TOTAL E =", cnt_e, "DELTA E =", delta_cnt_e)
            #    print(fcont*v*dt)
            #    print(fcont)
            #    print()

            # Output
            out_vec.add(t, u=u, v=v, a=a)
            out_energy.add(t, kin_e = calc_kinetic_energy(M, v),
                           int_e = calc_internal_energy(edm, elements, u),
                           cnt_e = contact_energy)

        out_vec.finalize()
        out_energy.finalize()

        return out_energy, out_vec

def VerticalTrussProblem():
    from mfpy.elements.quad import Quad
    from mfpy.elements.truss import Truss
    from mfpy.materials.linearelastic import LinearElastic
    from mfpy.dof import DOF
    from mfpy.boundcond import BC

    dt = 0.01
    t_end = 5.0
    pp_dt = dt

    nodes = [(0.,0.), (4.,0.), (4.,1.), (0.,1.),
             (2.,3.0), (2.,2.0)]
    nodes = [array(n, dtype=float64) for n in nodes]

    materials = [ LinearElastic(lmbda=0.0, mu=1.0, rho=1.0) ]

    elements = [ Quad(nodes, [0,1,2,3], materials[0], thickness=1),
                 Truss(nodes, [4,5], materials[0], area=1) ]

    vel_ic = [BC((4,5), DOF.Y, -1.0)]

    vel_bc = [BC((0,1), (DOF.X,DOF.Y), 0),
              BC((2,3,4,5), DOF.X, 0)]

    fext_bc = []

    out_scl, out_vec = DynamicExplicit.run(nodes, elements, materials, vel_ic, vel_bc, fext_bc, dt, t_end, pp_dt)

    from pylab import plot, show
    plot(out_vec.t, out_vec.u[:,11]+1)
    plot(out_vec.t, out_vec.u[:,9]+2.)
    plot(out_vec.t, out_vec.u[:,7]*0.5 + out_vec.u[:,5]*0.5 )

    show()

def HorizontalTrussProblem():
    from mfpy.elements import Quad
    from mfpy.elements import Truss
    from mfpy.materials.linearelastic import LinearElastic
    from mfpy.dof import DOF
    from mfpy.boundcond import BC

    dt = 0.01
    t_end = 5.0
    pp_dt = dt

    nodes = [(0.,0.), (4.,0.), (4.,1.), (0.,1.),
             (0.1 ,2.0), (3.9, 2.0)]
    nodes = [array(n, dtype=float64) for n in nodes]

    materials = [ LinearElastic(lmbda=0.0, mu=1.0, rho=1.0) ]

    elements = [ Quad(nodes, [0,1,2,3], materials[0], thickness=10),
                 Truss(nodes, [4,5], materials[0], area=10) ]

    vel_ic = [BC((4,5), DOF.Y, -1.)]

    vel_bc = [BC((0,1), (DOF.X,DOF.Y), 0),
              BC((2,3), DOF.X, 0)]

    fext_bc = []

    output = DynamicExplicit.run(nodes, elements, materials, vel_ic, vel_bc, fext_bc, dt, t_end, pp_dt)

    from pylab import plot, show
    plot(output.t, output.u[:,11]+1.0)
    plot(output.t, output.u[:, 9]+1.0)
    plot(output.t, output.u[:,7]*1.00 + output.u[:,5]*0.00)
    plot(output.t, output.u[:,7]*0.5 + output.u[:,5]*0.50)

    from mfpy.postproc import vtk_write_output

    vtk_write_output("C:/Users/Mohamed/Dropbox/COMMAS/thesis/contact/results","horiz-truss", nodes, elements, output, True)

    show()

def generate_mesh(width, height, num_x, num_y, start_x, start_y, node_offset=0):
    from numpy import linspace, meshgrid
    from itertools import product

    x_pos = linspace(0, width, num_x) + start_x
    y_pos = linspace(0, height, num_y) + start_y

    nodes = list(product(x_pos, y_pos))

    conn = []
    for elem_x in range(0,num_y-1):
        for elem_y in range(0,num_x-1):
            bl = elem_x +  elem_y*num_y + node_offset
            conn.append([bl, bl+num_y, bl+1+num_y, bl+1])

    return nodes, conn

