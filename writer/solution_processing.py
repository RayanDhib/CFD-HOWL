import numpy as np
from utils import compute_solShape_function
from cgns_utils import get_outputPnts_mapped_coords, get_CG_elemtype

def interpolate_to_nodes(nodes, connectivity, solution_data, ElemType, geoOrder,solOrder, var_names, nbEqs):
    """
    Interpolate solution data from solution points within each element to the element nodes.

    Args:
        nodes (np.array): Array of node coordinates.
        connectivity (np.array): Connectivity array defining elements.
        solution_data (list): Solution data at solution points within each element.
        ElemType (str): element type (e.g., 'Quad', 'Triag').
        geoOrder (int): geometric order of the elements.
        solOrder (int): Solution order of the elements.
        var_names (list): List of variable names like ['Pressure', 'Velocity'].
        nbEqs (int): Number of equations, should match the number of variable names.

    Returns:
        dict: Dictionary of interpolated node-based solution data, keyed by variable names.
    """
    # Initialize the dictionary to store interpolated node data
    node_based_solution_data = {var: np.zeros(len(nodes), dtype=np.float32) for var in var_names}

    # Determine shape functions (assuming all elements are of same type)
    outputPntsMappedCoords = get_outputPnts_mapped_coords(get_CG_elemtype(ElemType, geoOrder))
    solShapeFuncs = [compute_solShape_function(outputPntsMappedCoord, ElemType, solOrder) for outputPntsMappedCoord in (outputPntsMappedCoords)]

    # Loop over each element
    counter = np.zeros(len(nodes))
    for iElem, elem_nodes in enumerate(connectivity):
        # Retrieve solution data for current element
        element_sol_data = solution_data[iElem]  # Assuming each element's data is an array [num_sol_points, nbEqs]

        # Initialize array to store interpolated solution data at the element's nodes
        nodal_values = np.zeros((len(elem_nodes), nbEqs))

        # Evaluate the states at the output nodes
        for iPnt in range(len(outputPntsMappedCoords)):
            for iState in range(len(element_sol_data)):
                weight = solShapeFuncs[iPnt][iState]
                nodal_values[iPnt] += weight * element_sol_data[iState]

        # Aggregate interpolated values to global node-based data
        for iNode, node_id in enumerate(elem_nodes):
            counter[node_id - 1]+=1
            for iEq, var_name in enumerate(var_names):
                node_based_solution_data[var_name][node_id - 1] += nodal_values[iNode, iEq]  # Adjust index for 0-based Python indexing

    # Optional normalization or averaging can be performed here if necessary
    for i in range(len(nodes)):
        for var_name in (var_names):
            node_based_solution_data[var_name][i]=node_based_solution_data[var_name][i]/counter[i]
    return node_based_solution_data


