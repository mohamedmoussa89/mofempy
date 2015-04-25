__author__ = 'Mohamed Moussa'

from evtk.vtk import VtkFile, VtkUnstructuredGrid, VtkGroup
from os import path

def vtk_output_simulation(result_folder, name, enm, nodes, elements, kin_out):

    # Path to result folder
    result_path = path.join(result_folder, name)




