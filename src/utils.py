import sys

sys.dont_write_bytecode = True

from pyprojroot import here

root = here(project_files=[".here"])
sys.path.append(str(root))

from typing import List


# Function to print list in a better compact way
def lprint(list_data: List, ncols: int = 3, col_width: int = 40):
    """A simple utility function to print long lists in a more compact way using multiple columns.
    
    Args:
        list_data (List): list of items.
        ncols (int): number of columns to print in.
        col_width (int): column width for printing the list.
    """

    # Prepare all the separate columns
    cols = list()
    for i in range(ncols):
        col_data = list_data[i::ncols]
        if i > 0:
            col_data.extend([""] * (len(cols[0]) - len(col_data)))
        cols.append(col_data)

    # Combine the columns to create rows and print using the column widths
    for row in zip(*cols):
        for row_item in row:
            print(f"{row_item:<{col_width}}", end="\t")
        print()
