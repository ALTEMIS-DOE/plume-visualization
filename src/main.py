import sys

sys.dont_write_bytecode = True

from pyprojroot import here

root = here(project_files=[".here"])
sys.path.append(str(root))

import os
import h5py
import time
import imageio
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pygifsicle import optimize
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
from joblib import Parallel, delayed

import src.utils as utils
import src.data as data

parallel_function = Parallel(n_jobs=-1, verbose=5)


def parse_arguments() -> argparse.Namespace:
    """Reads commandline arguments and returns the parsed object.
    
    Returns:
        argparse.Namespace: parsed object containing all the input arguments.
    """
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
        help="List of variables that are to be visualized.",
    )
    
    parser.add_argument(
        "--gif_fps",
        type=int,
        default=5,
        help="Frames per second for the output GIF.",
    )
    
    parser.add_argument(
        "--temp_out_dir",
        type=str,
        default=".tmp",
        help="Relative or absolute path to the temporary directory that stores the intermediate output files such as CSVs and PNGs.",
    )

    args, unknown = parser.parse_known_args()

    # print("--- args ---")
    # print(args)

    return args


def get_year_for_cycle(plot_data_path: str, cycle_i: int) -> str:
    """Get the year for the specified cycle.
    
    Args:
        plot_data_path (str): Path to the plot_data.h5 file.
        cycle_i (int): Cycle number.
        
    Returns:
        str: Returns year in string format. Deafults to "????".
    """
    
    year = "????"
    
    if plot_data_path is not None:
        xmf_path = f"{plot_data_path}.{cycle_i}.xmf"
        if not os.path.exists(xmf_path):
            xmf_path = xmf_path.replace(".xmf", "ic.xmf")
            
        if os.path.exists(xmf_path):
            tree = et.parse(xmf_path)
            year = round(float(tree.getroot()[0][0][2].attrib["Value"]))
            
    return year


def get_layer_info(
    plot_data_path: str,
    temp_out_dir: str,
    shape_coords: pd.core.frame.DataFrame,
    variable_i: str,
    cycle_i: str,
    layer_i: int,
    groupby_obj: pd.core.groupby.generic.DataFrameGroupBy = None,
) -> None:
    """Gets information from a single cycle for a specified layer.

    Args:
        plot_data_path (str): Path to the plot_data.h5 file.
        temp_out_dir (str): The intermediate csv files are stored here.
        shape_coords (pd.core.frame.DataFrame): Pandas DataFrame containing the coordinate values 
            w.r.t. the scientific variables.
        variable_i (str): The variable of interest.
        cycle_i (str): The cycle of interest.
        layer_i (int): The layer of interest.
        groupby_obj (pd.core.groupby.generic.DataFrameGroupBy): shape_coords grouped by x,y coordinates to efficiently extract the layer.

    Returns:
        type: Returns the layer information for the specified variable, 
            cycle, and layer.
    """

    # Creating the file path
    plot_data = h5py.File(plot_data_path, "r")

    # extracting the specified frame for the current cycle
    shape_coords[variable_i] = np.squeeze(plot_data[variable_i][cycle_i])

    # for computational efficiency it is better to pass this as an argument
    if groupby_obj is None:
        groupby_obj = shape_coords.groupby(["x", "y"])

    # extract variable values only for layer_i
    layer_i_vals = groupby_obj.apply(lambda x: x.iloc[layer_i])
    
    # extracting the year for this cycle
    year_i = get_year_for_cycle(plot_data_path, cycle_i)
    
    # Write the extracted information to csv with the filename format being: YEARy_00000CYCLE.csv
    os.makedirs(temp_out_dir, exist_ok=True)
    layer_i_vals.to_csv(f"{temp_out_dir}/{year_i}y_{cycle_i.rstrip('ic').rjust(10,'0')}.csv")


def create_gif(
    io_dir: str, 
    layer_number: int, 
    fps: int=5, 
) -> None:
    """Creates GIF using the frames for each cycle.

    Args:
        io_dir (str): The directory containing the CSV files for each frame. 
            The CSV files contain the x,y,z coordinates along with the variable information.
        layer_number (int): The layer of interest. Only used for the plot title.
        fps (int): Frames per second for the resulting GIF. Defaults to 5.
    """
    csv_input_dir = os.path.join(io_dir, "csvs")
    png_output_dir = os.path.join(io_dir, "pngs")
    
    # Getting the CSV paths for each frame. Sorting them to have the correct order of cycles.
    csvs = [
        os.path.join(csv_input_dir, f)
        for f in os.listdir(csv_input_dir)
        if f.split(".")[-1] == "csv"
    ]

    #################
    # Creating PNGs #
    #################
    
    # Creating a subfunction to parallelize the plotting process.
    def plot_n_save(csv_path):
        csv_filename = os.path.basename(csv_path)
        year_n_cycle = csv_filename.split("y_")
        year = year_n_cycle[0]
        cycle = year_n_cycle[1].split(".")[0].lstrip("0")
        
        # Creating the plots
        cycle_df = pd.read_csv(csv_path)
        plt.figure()
        plt.title(
            f"Year: {year: >5}             Cycle: {cycle: >6}"
        )
        plt.scatter(
            cycle_df.x,
            cycle_df.y,
            c=cycle_df["total_component_concentration.cell.Tritium conc"],
            cmap="Blues",
            vmin=0,
            vmax=2e-9,
        )
        plt.colorbar()
        
        # Saving the plot
        os.makedirs(png_output_dir, exist_ok=True)
        png_path = os.path.join(png_output_dir, csv_filename.replace(".csv", ".png"))
        plt.savefig(png_path)
    # ^^ End of sub-function

    # Save individual frames for each cycle
    parallel_function(delayed(plot_n_save)(csv_path=csv) for csv in csvs)

    ################
    # Creating GIF #
    ################
    
    # Use all the frames for each cycle to create the gif
    gif_output_path = f"layer_{layer_number}_cycles.gif"
    
    pngs = [
        os.path.join(png_output_dir, f)
        for f in os.listdir(png_output_dir)
        if f.split(".")[-1] == "png"
    ]
    
    # Sort the pngs files using only the cycle count and not the year included in the filename.
    pngs.sort(key=lambda x: x.split("y_")[-1])
    
    with imageio.get_writer(gif_output_path, mode="I", fps=fps) as writer:
        with tqdm(pngs) as tqdm_pngs:
            tqdm_pngs.set_description("Generating GIF from PNGs")

            for png in tqdm_pngs:
                cycle_frame = imageio.imread(png)
                writer.append_data(cycle_frame)
    
    # Using this function to regulate the size of the output GIF file.
    try:
        optimize(gif_output_path)
    except ValueError as ve:
        # raises this exception if the input file path is incorrect.
        raise ve
    except FileNotFoundError as fnfe:
        # mostly raises this exception if the gifsicle package is not installed.
        print("Ignoring GIF file optimization because required package is not found...")


def main():
    # Get the command-line arguments
    args = parse_arguments()

#     # Used to store intermediate output files. Aids parallel computation. # TODO: Add as an argparser.
#     temp_out_dir = ".tmp"
    csvs_dir = os.path.join(args.temp_out_dir, "csvs")

    # Only to be used while debugging to save time on generating CSVs.
    gen_csv = True
    if gen_csv:

        # Get the data from the input directory
        plot_mesh, plot_data = data.get_data(args.input_dir, verbose=False)
        prism_coords = data.get_prism_coords(plot_mesh)

        # Storing prism coordinates in a dataframe to efficiently extract layer
        coords_map_df = pd.DataFrame(prism_coords)
        coords_map_df.columns = "x", "y", "z"
        coords_map_groupby_layer = coords_map_df.groupby(["x", "y"])

        # working with only one variable of interest right now.
        variable_i = args.variables_of_interest[0]

        # preparing inputs for the get_layer_info function.
        cycles = data.get_cycles(plot_data)
        plot_data_path = os.path.join(args.input_dir, "plot_data.h5")

        # Parallely extracting information for each cycle for the specified layer.
        parallel_function(
            delayed(get_layer_info)(
                plot_data_path=plot_data_path,
                temp_out_dir=csvs_dir,
                shape_coords=coords_map_df,
                variable_i=variable_i,
                cycle_i=cycle_i,
                layer_i=args.layer_number,
                groupby_obj=coords_map_groupby_layer,
            )
            for cycle_i in cycles
        )

    # create GIF using frames for each cycle
    create_gif(
        io_dir=args.temp_out_dir,
        layer_number=args.layer_number,
        fps=args.gif_fps,
    )

    return


if __name__ == "__main__":
    main()
