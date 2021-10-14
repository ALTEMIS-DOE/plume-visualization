# lbnl-plume-time-series-viz

This tool is used to create a time-series 2D visualization of the a specified element.
The 2D plot is of a single specified layer from the 3D mesh.

## Setting up the environment (recommended)
Although not mandatory, it is recommended to create a virtualenv to predict the outputs of the code.
Use the provided `requirements.yml` file to create the virtual env.

#### To create a new environment, use the following command.
```
conda env create -f requirements.yml
```

#### To update an existing environment, use the following command.
```
conda env update -f requirements.yml
```


## How to run the code?
This code needs as input the following params:
- `--input_dir`: the directory containing the plot_mesh and plot_data files.
- `--layer_number`: the layer of interest.
- `--variables_of_interest`: the variable/element of interest. Currently this only supports a single variable, but in the later versions it will support multiple variables of interest.
- `--gif_fps`: frames per second for creating the output GIF.
- `--temp_out_dir`: temporary output directory where all the intermediate output files are stored.

These commands can also be specified in a configuration file. Check [`src/main.cfg`](src/main.cfg).

Use the following command to run the code:
#### OPTION 1: Using configuration file
```
python main.py @main.cfg
```


#### OPTION 2: Using commandline arguments
```
python main --input_dir path/to/input_dir --layer_number 7 --variables_of_interest variable_name
```

## Sample GIF using a sample simulation output dataset
This sample plots tritium concentrations at layer 7 from the surface.
![Layer 7 GIF](layer_7_cycles.gif)


