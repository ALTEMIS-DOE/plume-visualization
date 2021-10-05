import sys

sys.dont_write_bytecode = True

from pyprojroot import here

root = here(project_files=[".here"])
sys.path.append(str(root))

import os
import h5py
import time
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

from typing import Tuple, List

import src.utils as utils


def get_data(
    input_dir: str, verbose: bool = False
) -> Tuple[h5py._hl.group.Group, h5py._hl.files.File]:
    """Reads plot_mesh.h5 and plot_data.h5 from the specified input_dir.
    
    Args:
        input_dir (str): The directory containing the input files.
        verbose (bool): Boolean flag to print summary of data within plot_mesh.h5 and plot_data.h5.
    
    Returns:
        Tuple[h5py._hl.group.Group, h5py._hl.files.File]: Returns a tuple of the data within plot_mesh.h5 and plot_data.h5 files.
    """

    ###############
    ## plot_mesh ##
    ###############

    # Creating the file path
    plot_mesh_path = os.path.join(input_dir, "plot_mesh.h5")

    # Reading the file
    f_plot_mesh = h5py.File(plot_mesh_path, "r")

    # Extracting all the mappings and coordinate values
    mesh_k1 = list(f_plot_mesh)[0]
    plot_mesh = f_plot_mesh[mesh_k1]["Mesh"]

    # Print summary of the mesh file
    if verbose:
        print("\n>>> Plot Mesh <<<")
        for k in plot_mesh:
            print(f"{k: >20} => {plot_mesh[k].shape}")

    ###############
    ## plot_data ##
    ###############

    # Creating the file path
    plot_data_path = os.path.join(input_dir, "plot_data.h5")

    # Reading the file
    f_plot_data = h5py.File(plot_data_path, "r")

    # Print the list of scientific variables within the data file
    if verbose:
        variable_list = list(f_plot_data.keys())
        print("\n>>> Plot Data Variables <<<")
        utils.lprint(variable_list, col_width=25)

    return plot_mesh, f_plot_data


def get_cycles(plot_data: h5py._hl.files.File) -> List[str]:
    """Get the list of simulation cycles.
    
    Args:
        plot_data (h5py._hl.files.File): Contents of the plot_data.h5 file.
    
    Returns:
        List[str]: Returns the list of cycle values that were obtained from the simulation run.
    """

    # get the first key from this dataset
    k1 = list(plot_data.keys())[0]

    # assuming that the cycle count is same for all variables
    # read all the cycle values from this first variable
    return list(plot_data[k1].keys())


def compute_node_mappings(plot_mesh: h5py._hl.group.Group) -> List[np.array]:
    """Gets the list of x,y,z coordinates corresponding to the shape (prism in this case) corners. The shape corners are mentioned in order in the MixedElements section within the plot_mesh.h5 file.
    
    Args:
        plot_mesh (h5py._hl.group.Group): Data within the plot_mesh.h5 file. Contains the coordinates and the mapping between the coordinates and the shape corners under the MixedElements key.
    
    Returns:
        Returns the list of coordinates corresponding to the shape corners.
    """

    # fetch mixedElements index values
    mixedElements = np.squeeze(plot_mesh[f"MixedElements"])

    # fetch node coordinates from the mesh
    nodes = np.array(plot_mesh[f"Nodes"])

    # map node xyz coordinates to the mixedElements index values
    me_nodes = list(map(nodes.__getitem__, mixedElements))

    return me_nodes


def compute_prism_centers(coords_list: List[np.array]) -> List[np.array]:
    """Computes the center coordinate of the prism using its corner coordinates. For a prism, there are 6 corner coordinates. This function takes the mean of these coordinates to compute the spatial center of the 3D prism. 
    It is assumed here that within the list of xyz coords, each 7 consecutive values correspond to a single prism. The first value for each prism in the MixedElements is some kind of identification value followed by the corner coordinate index values coorsponding to the x,y,z coordinates in the Nodes array.
    
    Args:
        coords_list (List[np.array]): List of coordinates corresponding to the shape corners.
    
    Returns:
        List[np.array]: Returns the list of prism center coordinates.
    """

    num_corners = 6  # depends on the model shape (prism:6; cube:8)

    prism_coords = list()

    # cannot parallelize this because the order of values needs to be maintained.
    for i in range(0, len(coords_list), num_corners + 1):
        # i <- some ID value that is not currently being used.
        start_idx = i + 1
        end_idx = i + num_corners + 1
        shape_corners = np.array(coords_list[start_idx:end_idx])

        # Computing mean to represent the center coordinates of the prism
        prism_coords.append(shape_corners.mean(axis=0))

    return prism_coords


def get_prism_coords(plot_mesh: h5py._hl.group.Group) -> List[np.array]:
    """Computes coordinates representing the prism using the contents of plot_mesh.h5 file.
    In this case the prism is being represented by the mean of the corner coordinates.
    
    Args:
        plot_mesh (h5py._hl.group.Group): Data within the plot_mesh.h5 file. Contains the coordinates and the mapping between the coordinates and the shape corners under the MixedElements key.
        
    Returns:
        List[np.array]: Returns the list of prism center coordinates.
    """

    me_nodes = compute_node_mappings(plot_mesh)
    prism_coords = compute_prism_centers(me_nodes)

    return prism_coords
