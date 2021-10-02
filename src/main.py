import sys
sys.dont_write_bytecode=True

import os
import h5py
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

import utils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plume layer visualization.",
        fromfile_prefix_chars="@",  # helps read the arguments from a file.
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to the input directory that contains all the plot data and plot mesh files.",
    )

    parser.add_argument(
        "--layer_number",
        type=int,
        default=None,
        choices=range(1, 18),
        help="The layer number to visualize.",
    )
    
    parser.add_argument(
        "--variables_of_interest",
        type=str,
        default=None,
        nargs="*",
        help="List of variables that are to be visualized."
    )

    args, unknown = parser.parse_known_args()

    # print("--- args ---")
    # print(args)

    return args


def get_data(input_dir, verbose=False):
    ###############
    ## plot_mesh ## 
    ###############
    
    # Creating the file path
    plot_mesh_path = os.path.join(input_dir, "plot_mesh.h5")
    
    # Reading the file
    f_plot_mesh = h5py.File(plot_mesh_path, 'r')
    
    # Extracting all the mappings and coordinate values
    mesh_k1 = list(f_plot_mesh)[0]
    plot_mesh = f_plot_mesh[mesh_k1]['Mesh']
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
    f_plot_data = h5py.File(plot_data_path, 'r')
    
    # Extracting all the cycle and layer information for all scientific variables
    variable_list = list(f_plot_data.keys())
    if verbose:
        print("\n>>> Plot Data Variables <<<")
        utils.lprint(variable_list, col_width=25)
    
    return plot_mesh, f_plot_data


def get_cycles(plot_data):
    # get the first key from this dataset
    k1 = list(plot_data.keys())[0]
    
    # assuming that the cycle count is same for all variables
    # read all the cycle values from this first variable
    return list(plot_data[k1].keys())


###########################
### >>>>>>>>>>>>
### >>>> TODO: IMPLEMENT THE MAIN CODE LOGIC IN THIS FUNCTION
### >>>>>>>>>>>>
###########################
def get_layer_info(cycle_i, layer_i):
#     get mixedElements for cycle_i
#     assign xyz for mixedElements
#     create prism center coordinates from mixedElements
#     assign variable info to each prism center coordinate
#     extract layer_i from this data
#     either return this data or return the plot made from this data.
    pass


def main():
    # Get the command-line arguments
    args = parse_arguments()    
    
    # Get the data and cycles to be used for the time-series plot
    plot_mesh, plot_data = get_data(args.input_dir, verbose=False)
    cycles = get_cycles(plot_data)
    
    # Create a time-series GIF for each variable of interest
    for variable in args.variables_of_interest:
        
        # all the gif frames are stored in this list
        gif_list = list()
        
        # extracting frame from each cycle
        for cycle_i in cycles:
            
            # extracting the specified frame for the current cycle
            layer_info = get_layer_info(cycle_i, args.layer_number)
            gif_list.append(layer_info)

        # create GIF using frames for each cycle
        create_gif(gif_list)

        return    # only for debugging. Making sure this works with one variable. 

if __name__ == "__main__":
    main()
