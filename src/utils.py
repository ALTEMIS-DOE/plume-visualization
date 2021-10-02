# Function to print list in a better compact way
def lprint(list_data, ncols=3, col_width=40):
    cols = list()
    for i in range(ncols):
        cols.append(list_data[i::ncols])

    for row in zip(*cols):
        for row_item in row:
            print(f"{row_item:<{col_width}}", end="\t")
        print()
