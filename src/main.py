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
import matplotlib.pyplot as plt
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

    args, unknown = parser.parse_known_args()

    # print("--- args ---")
    # print(args)

    return args


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

    # Write the extracted information to csv
    os.makedirs(temp_out_dir, exist_ok=True)
    layer_i_vals.to_csv(f"{temp_out_dir}/{cycle_i.strip('ic').rjust(10,'0')}.csv")


def create_gif(input_dir: str, layer_number: int) -> None:
    """Creates GIF using the frames for each cycle.

    Args:
        input_dir (str): The directory containing the CSV files for each frame. 
            The CSV files contain the x,y,z coordinates along with the variable information.
        layer_number (int): The layer of interest. Only used for the plot title.
    """

    # Getting the CSV paths for each frame. Sorting them to have the correct order of cycles.
    csvs = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.split(".")[-1] == "csv"
    ]
    csvs.sort()

    # Creating a subfunction to parallelize the plotting process.
    def plot_n_save(csv_path):
        # TODO: Fix the varaible concentration limits to avoid random color shifts.
        # TODO: Display year instead of cycles.
        cycle_df = pd.read_csv(csv_path)
        plt.figure()
        plt.title(f"cycle: {os.path.basename(csv_path).split('.')[0].strip('0')}")
        plt.scatter(
            cycle_df.x,
            cycle_df.y,
            c=cycle_df["total_component_concentration.cell.Tritium conc"],
            cmap="Blues",
        )
        plt.colorbar()
        plt.savefig(csv_path.replace(".csv", ".png"))

    # Save individual frames for each cycle
    parallel_function(delayed(plot_n_save)(csv_path=csv) for csv in csvs)

    # Use all the frames for each cycle to create the gif
    # TODO: Have an option to control the frame interval in the gif
    with imageio.get_writer(f"layer_{layer_number}_cycles.gif", mode="I") as writer:
        with tqdm(csvs) as tqdm_csvs:
            tqdm_csvs.set_description("Generating GIF")

            for csv in tqdm_csvs:
                cycle_frame = imageio.imread(csv.replace(".csv", ".png"))
                writer.append_data(cycle_frame)


def main():
    # Get the command-line arguments
    args = parse_arguments()

    # Used to store intermediate output files. Aids parallel computation.
    temp_out_dir = ".tmp"

    # Only to be used while debugging to save time one generating CSVs.
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
                temp_out_dir=temp_out_dir,
                shape_coords=coords_map_df,
                variable_i=variable_i,
                cycle_i=cycle_i,
                layer_i=args.layer_number,
                groupby_obj=coords_map_groupby_layer,
            )
            for cycle_i in cycles
        )

    # create GIF using frames for each cycle
    create_gif(input_dir=temp_out_dir, layer_number=args.layer_number)

    return


if __name__ == "__main__":
    main()
