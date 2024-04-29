import numpy as np
from .utils import compute_solShape_function
from .cgns_utils import get_outputPnts_mapped_coords, get_CG_elemtype

def interpolate_to_nodes(mesh, solution_data, var_names, nbEqs):
    """
    Interpolate solution data from solution points within each element to the element nodes.
    
    Parameters:
        mesh (MeshData): MeshData object containing mesh information.
        solution_data (list of np.array): Solution data at solution points for each element.
        var_names (list of str): Names of the variables (e.g., ['Pressure', 'Velocity']).
        nbEqs (int): Number of equations, should correspond to the length of var_names.

    Returns:
        dict: Dictionary where keys are variable names and values are interpolated node-based data.
    """
    node_based_solution_data = {var: np.zeros(mesh.num_nodes, dtype=np.float32) for var in var_names}

    # Determine shape functions (assuming all elements are of same type)
    outputPntsMappedCoords = get_outputPnts_mapped_coords(get_CG_elemtype(mesh.elem_type, mesh.geo_order))
    solShapeFuncs = [compute_solShape_function(outputPntsMappedCoord, mesh.elem_type, mesh.sol_order) for outputPntsMappedCoord in (outputPntsMappedCoords)]

    # Loop over each element
    counter = np.zeros(mesh.num_nodes)
    for iElem, elem_nodes in enumerate(mesh.connectivity):
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
            counter[node_id - 1] += 1
            for iEq, var_name in enumerate(var_names):
                node_based_solution_data[var_name][node_id - 1] += nodal_values[iNode, iEq]  # Adjust index for 0-based Python indexing

    # Normalize the interpolated data by the number of contributions to each node
    for i in range(mesh.num_nodes):
        for var_name in var_names:
            if counter[i] > 0:
                node_based_solution_data[var_name][i] /= counter[i]
    
    return node_based_solution_data
