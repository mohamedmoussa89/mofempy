"""VTK related functionality"""

import shutil, glob
from os import path, makedirs

from numpy import array, zeros, empty
from evtk.vtk import *

from mfpy.elements import Truss, Quad
from mfpy.assembly import calculate_nds, calculate_ndm, calculate_edm, calculate_ntdm

def vtk_write_output(results_root, name, nodes, elements, output, overwrite = False):
    """Write given output as a VTK group.

    A VTK group is a group of VTK files that represents transient data.

    Parameters
    ----------
    results_root : str
        Root path to result directory
    name : str
        Name of simulation
    enm : list of array
        Element Node Map
    nodes : list of array
        List of all nodes in the simulation
    elements : list of Element
        List of all elements in the simulation
    kin_out : KinematicOutput
        KinematicOutput object that stores all displacement, velocity and acceleration data.
    """

    # Create result folder
    folder_path = __vtk_create_result_folder(results_root, name, overwrite)

    # Create grouping for time dependant data
    grp_name = name + "-group"
    grp_path = path.join(folder_path, grp_name)
    grp = VtkGroup(grp_path)

    # Create mappings
    enm = [e.enm for e in elements]
    elem_types = [type(e) for e in elements]
    nds = calculate_nds(enm, elem_types, len(nodes))
    ndm = calculate_ndm(nds)
    ntdm = calculate_ntdm(nds, enm, ndm)

    # Create a file for every time-step
    for t_id, t in enumerate(output.t):
        file_name = name + "-%s" % t
        file_path = path.join(folder_path, file_name)

        # Create file, add it to the group
        f = VtkFile(file_path, VtkUnstructuredGrid)
        grp.addFile(file_path+".vtu", t)

        num_nodes = len(nodes)
        num_elements = len(elements)

        f.openElement("UnstructuredGrid")
        f.openPiece(npoints=num_nodes, ncells=num_elements)

        node_data = __vtk_add_nodes_to_file(f, nodes)

        f.openElement("PointData")
        u_data = __vtk_add_vector_data_to_file(f, ntdm, output.u[t_id,:], "displacement")
        f.closeElement("PointData")

        # Add elements
        conn_data, offset_data, type_data = __vtk_add_elements_to_file(f, enm, elements)

        f.closePiece()
        f.closeElement("UnstructuredGrid")

        f.appendData(node_data)
        f.appendData(u_data)
        f.appendData(conn_data)
        f.appendData(offset_data)
        f.appendData(type_data)

        f.save()

    grp.save()


def __vtk_create_result_folder(results_root, name, overwrite):
    """Make result folder for this simulation

    Parameters
    ----------
    result_path : str
        Path to where the new folder will be created for this simulation
    name : str
        Name of new result folder
    overwrite : bool
        Whether or not to overwrite the result folder, if it already exists for this simulation

    Returns
    -------
    folder_path : str
        Path to result folder where all time-step output files will be stored
    """

    folder_name = name + "-vtk"
    folder_path = path.join(results_root, folder_name)

    if path.exists(folder_path):
        # Folder already exists

        if overwrite:
            # Remove all sub files
            files = glob.glob(path.join(folder_path, "*.*"))
            for f in files:
                os.remove(f)

        if not overwrite:
            # Find the next available folder name by adding integer post-fix
            n = 1
            while path.exists(folder_path):
                folder_path = path.join(results_root, folder_name + "-%s" % n)
                n += 1

    else:
        # Path doesnt exists
        makedirs(folder_path)

    return folder_path


def __vtk_add_nodes_to_file(f, nodes):
    """Add node data header to file. Actual node data is returned to be appended at the end

    Parameters
    ----------
    f : VtkFile
        File to edit
    nodes : list of array
        List of nodes to output to file

    Returns
    ------
    node_data : tuple of array
        Tuple of x,y and z coordinate arrays of nodes
    """

    x = array([n[0] for n in nodes], dtype="Float64")
    y = array([n[1] for n in nodes], dtype="Float64")
    z = zeros(len(nodes), dtype="Float64")
    node_data = (x,y,z)

    f.openElement("Points")
    f.addData("Points", node_data)
    f.closeElement("Points")

    return node_data


def __vtk_get_cell_type(element_type):
    """Map mfpy element type to VTK cell type"""

    if (element_type == Truss): return VtkLine.tid
    if (element_type == Quad): return VtkQuad.tid

    # Empty cell type
    return 0


def __vtk_add_elements_to_file(f, enm, elements):
    """Add cell data header to file. Actual cell data is returned to be appended at the end

    Parameters
    ----------
    f : VtkFile
        File to edit
    enm : list of array
        Element node map
    elements : list of Element
        List of elements to add to file

    Returns
    -------
    conn_data : array
        Element-node connectivity data
    offset_data : array
        Offset data for each element, equal to the cumulative sum of the number of nodes
    type_data : Array
        VTK cell type for each element
    """

    # Connectivity
    conn_data = array([nid for elem_nm in enm for nid in elem_nm], dtype="UInt32")

    # Offsets
    num_elem = len(elements)
    offset_data = empty(num_elem, dtype="UInt32")
    offset = 0
    for elem_id, nm in enumerate(enm):
        offset += len(nm)
        offset_data[elem_id] = offset

    # Types
    type_data = array([__vtk_get_cell_type(type(elem)) for elem in elements], dtype = "UInt32")

    f.openElement("Cells")
    f.addData("connectivity", conn_data)
    f.addData("offsets", offset_data)
    f.addData("types", type_data)
    f.closeElement("Cells")

    return conn_data, offset_data, type_data


def __vtk_add_vector_data_to_file(f, dof_map, vec, name):
    """Adds vector data to file, mapping from a given vector vector

    Parameters
    ----------
    f : VtkFile
        File to edit
    dof_map : list of array
        Some mapping from node ID to global DOFs
    vec : array
        Global array
    name : str
        Name to use for data in file

    Returns
    -------
    node_data : tuple of array
        The data that needs to be appended at the end of the file
    """

    num_nodes = len(dof_map)

    x_map = [dof_map[node_id][0] for node_id in range(num_nodes)]
    y_map = [dof_map[node_id][1] for node_id in range(num_nodes)]

    x = vec[x_map]
    y = vec[y_map]
    z = y * 0

    data = (x,y,z)
    f.addData(name, data)

    return data