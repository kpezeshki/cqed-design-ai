import gdspy
import numpy as np
import os

def slice_and_boolean(inpath, savepath, focus_box, booleans, layers_to_save = "all"):
    """
    function is designed to make converting from full cad for a chip to a focused region of the chip
    typically with only essential layers for simulation.  This is especially useful when itterating over sonnet
    simulations where repertative boolean operations in klayout become tedious.
    __________________________________________________________________
    inputs:
    inpath - the file which will be modified
    savepth - the path to save the modified gds file
    focus_box - the bounding box of the output design.  Must be 2x2 numpy array of format
        np.array([[xmin, ymin], [xmax, ymax]])
    booleans - dictionary of all boolean operations to be preformed.  Should be of format
        booleans = {"layer 1": [],
                "layer 2": [],
                "layer out": [],
                "operation":[]}
        layer entries can be either integers (layer only) or tuples (layer, datatype)
        availible operations defined by the gdspy documentation: 'or', 'and', 'xor', 'not'
    layers_to_save - optional. A subset of the design layers which should be saved in the new file.
                     Can be "all", a list of integers (layers only), or a list of tuples (layer, datatype)
    """
    library = gdspy.GdsLibrary(infile = inpath)

    ## create a new gds library to write to
    manipulatedLibrary = gdspy.GdsLibrary()
    main_cell = manipulatedLibrary.new_cell("NEW")


    top_cells = library.top_level()
    for cell in top_cells:   # for all cells in the GDS file, get all polygons and the associated layer data
        poly_dict = cell.get_polygons(by_spec = True)
        sliced_polygons = {}
        for metadata, polygons in poly_dict.items(): # slice each layer to fit the focus box, add to dictionary
            layer, datatype = metadata
            print(f"slicing polygons from layer {layer}, datatype {datatype}")
            to_add = gdspy.slice(polygons, list(focus_box[:, 1]), axis = 1, layer = layer, datatype = datatype)[1]
            to_add = gdspy.slice(to_add, list(focus_box[:, 0]), axis = 0, layer = layer, datatype = datatype)[1]
            sliced_polygons[metadata] = to_add

    for bool_num in range(len(booleans["layer 1"])): # preform all boolean operations according to layers
        print(f"preforming boolean {bool_num+1}")
        layer_1 = booleans["layer 1"][bool_num]
        layer_2 = booleans["layer 2"][bool_num]
        out_layer = booleans["layer out"][bool_num]
        operation = booleans["operation"][bool_num]

        # Handle both integer (layer only) and tuple (layer, datatype) formats
        if isinstance(layer_1, int):
            layer_1 = (layer_1, 0)  # default datatype 0
        if isinstance(layer_2, int):
            layer_2 = (layer_2, 0)  # default datatype 0
        if isinstance(out_layer, int):
            out_layer = (out_layer, 0)  # default datatype 0

        result = gdspy.boolean(sliced_polygons[layer_1],
                              sliced_polygons[layer_2],
                              operation,
                              layer = out_layer[0],
                              datatype = out_layer[1])
        sliced_polygons[out_layer] = result

    if layers_to_save == "all": # if saving all layers, set the variable appropriatly
        layers_to_save = sliced_polygons.keys()
    else:
        # Convert integer layer specifications to (layer, datatype) tuples
        converted_layers = []
        for layer_spec in layers_to_save:
            if isinstance(layer_spec, int):
                # Find all (layer, datatype) combinations for this layer
                for metadata in sliced_polygons.keys():
                    if metadata[0] == layer_spec:
                        converted_layers.append(metadata)
            else:
                converted_layers.append(layer_spec)
        layers_to_save = converted_layers

    for metadata, polygon in sliced_polygons.items():  # add the sliced and booleans polygons to the main cell
        if metadata in layers_to_save:
            layer, datatype = metadata
            print(f"adding polygons on layer {layer}, datatype {datatype}...")
            main_cell.add(polygon)

    # save the new gds library
    manipulatedLibrary.write_gds(savepath)
    return



def main():
    # define the read and write paths for the gds files
    inpath = r"InBB04_forsim.gds"
    savepath = r"InBB04_R1_fromscript.gds"

    # define the box we'd like to cut out
    x_min = 600
    x_max = 2030
    y_min = -2300
    y_max = -850
    focus_box = np.array([(x_min, y_min),(x_max, y_max)])

    # define boolean operations, if no booleans are required, leave lists blank
    booleans = {"layer 1": [],
                "layer 2": [],
                "layer out": [],
                "operation":[]}

    # optional argument, list of layers that should be saved to the new gds
    # Can be integers (layer only) or tuples (layer, datatype)
    layers_to_save = [1, 3]  # or [(1, 0), (3, 0)] for specific datatypes

    slice_and_boolean(inpath, savepath, focus_box, booleans, layers_to_save)

if __name__ == "__main__":
    main()
