# lbnl-plume-time-series-viz

This tool is used to create a time-series 2D visualization of the a specified element.
The 2D plot is of a single specified layer from the 3D mesh.

## How to run the code?
This code needs as input the following params:
- the directory containing the plot_mesh and plot_data files.
- the layer of interest
- the variable/element of interest

These commands can also be specified in a configuration file. Check [`src/main.cfg`](src/main.cfg).

use the following command to run the code:
> Using configuration file
```
python main.py @main.cfg
```

> Using commandline arguments
```
python main --input_dir path/to/input_dir --layer_number 7 --variables_of_interest variable_name
```
