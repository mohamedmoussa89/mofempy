from numpy import array, float64

from mfpy.elements import Quad
from mfpy.materials.linearelastic import LinearElastic
from mfpy.dof import DOF
from mfpy.boundcond import BC
from mfpy.controls import DynamicExplicit

def BlockProblem():


    dt = 1e-3
    t_end = 2.0
    pp_dt = dt

    # Node definitions
    nodes = [(0.,0.), (4.,0.), (4.,1.), (0.,1.),
             (1.,2.,), (3, 2.), (3.,3.), (1.,3.)]
    nodes = [array(n, dtype=float64) for n in nodes]

    # Surface definitions
    surfaces = [[(1,2),(2,3),(3,0)],
                [(7,4),(4,5),(5,6)]]

    # Surface pair definitions
    pairs = [(0,1)]

    materials = [ LinearElastic(lmbda=10.0, mu=10.0, rho=1.0) ]

    elements = [ Quad(nodes, c, materials[0], thickness=1) for c in [[0,1,2,3], [4,5,6,7]]]

    vel_ic = [BC((4,5,6,7), DOF.Y, -1.)]

    vel_bc = [BC((0,1), (DOF.X,DOF.Y), 0)]

    fext_bc = []#[BC((4,5,6,7), DOF.Y, -5.0)]

    out_scl, out_vec = DynamicExplicit.run(nodes, elements, materials,
                                           surfaces, pairs,
                                           vel_ic, vel_bc, fext_bc,
                                           dt, t_end, pp_dt)

    from pylab import plot, show, figure
    #plot(out_vec.t, out_vec.u[:,11]+1.0)
    #plot(out_vec.t, out_vec.u[:, 9]+1.0)
    #plot(out_vec.t, out_vec.u[:,7]*1.00 + out_vec.u[:,5]*0.00)
    #plot(out_vec.t, out_vec.u[:,7]*0.5 + out_vec.u[:,5]*0.50)

    figure()
    plot(out_scl.t, out_scl.kin_e)
    plot(out_scl.t, out_scl.int_e)
    plot(out_scl.t, out_scl.cnt_e)
    plot(out_scl.t, array(out_scl.kin_e) + array(out_scl.int_e) - array(out_scl.cnt_e))

    from mfpy.postproc import vtk_write_output
    vtk_write_output("C:/Users/Mohamed/Dropbox/COMMAS/thesis/contact/results","horiz-truss", nodes, elements, out_vec, True)

    show()

if __name__ == "__main__":
    #VerticalTrussProblem()
    #HorizontalTrussProblem()
    BlockProblem()
