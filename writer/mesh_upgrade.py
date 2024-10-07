import numpy as np
from scipy.spatial import cKDTree
from .utils import compute_geoShape_function
from .cgns_utils import get_cgns_cell_node_conn, get_CG_elemtype, get_outputPnts_mapped_coords

def upgrade_geoOrder(nodes, connectivity, old_geoOrder, new_geoOrder, ElemType):
    """
    Upgrade the geometric order of the mesh and connectivity.

    Parameters:
        nodes (np.array): The original node positions.
        connectivity (list[list[int]]): Original connectivity data.
        old_geoOrder (int): Original geometric order.
        new_geoOrder (int): Target geometric order to upgrade to.
        ElemType (str): Type of the elements in the mesh.

    Returns:
        tuple: A tuple containing upgraded nodes and connectivity arrays.
    """
    # Initialize upgraded nodes with existing node positions
    upgraded_nodes = np.copy(nodes)

    # Initialize the connectivity table with the old data (assuming right CGNS ordering)
    nbrNodesQ1 = len(get_outputPnts_mapped_coords(get_CG_elemtype(ElemType, 1)))
    upgraded_connectivity = [list(row[:nbrNodesQ1]) for row in connectivity]

    # Build a KD-Tree for quick node lookup
    node_tree = build_node_tree(upgraded_nodes)
    CG_elemtype = get_CG_elemtype(ElemType, new_geoOrder)
    outputPntsMappedCoords = get_outputPnts_mapped_coords(CG_elemtype)
    geoShapeFuncs = [compute_geoShape_function(coord, ElemType, old_geoOrder) for coord in outputPntsMappedCoords]

    for iElem, elem_nodes in enumerate(connectivity):
        element_nodes = [nodes[i-1] for i in elem_nodes]
        new_nodes_position = compute_new_nodes_position(outputPntsMappedCoords, element_nodes, geoShapeFuncs, nbrNodesQ1)

        for newPos in new_nodes_position:
            node_exists, node_index = check_node_existence(node_tree, newPos)
            if not node_exists:
                upgraded_nodes = np.vstack([upgraded_nodes, [newPos]])
                node_tree = build_node_tree(upgraded_nodes)  # Rebuild the tree with new node
                node_index = upgraded_nodes.shape[0] - 1

            upgraded_connectivity[iElem].append(node_index + 1)  # Append 1-based index

    return upgraded_nodes, np.array(upgraded_connectivity)

def build_node_tree(nodes):
    """ Build a KD-Tree for quick node lookup. """
    return cKDTree(nodes)

def check_node_existence(node_tree, new_pos, tolerance=1e-8):
    """
    Check if a node exists in the tree and return its index or indicate non-existence.

    Parameters:
        node_tree (cKDTree): KD-Tree containing node positions.
        new_pos (np.array): The position of the node to check.
        tolerance (float): Distance tolerance for considering two nodes as the same.

    Returns:
        tuple: Boolean indicating if the node exists and the index if it does.
    """
    closest_dist, closest_idx = node_tree.query(new_pos)
    if closest_dist < tolerance:
        return True, closest_idx
    return False, None

def preprocess_connectivity(connectivity, CGelement_type):
    """
    Adjust the connectivity array to meet CGNS specifications with 1-based indexing.

    Parameters:
        connectivity (list[list[int]]): Original connectivity data.
        CGelement_type (int): CGNS element type specification.

    Returns:
        list[list[int]]: Adjusted connectivity with 1-based indexing.
    """
    cgns_order = get_cgns_cell_node_conn(CGelement_type)
    adjusted_connectivity = [[conn[i] + 1 for i in cgns_order] for conn in connectivity]
    return adjusted_connectivity

def compute_new_nodes_position(outputPntsMappedCoords, element_nodes, geoShapeFuncs, nbrNodesQ1):
    """
    Compute new node positions based on geometric shape functions.

    Parameters:
        outputPntsMappedCoords (np.array): Mapped coordinates for output points.
        element_nodes (np.array): Node coordinates for the current element.
        geoShapeFuncs (list): Geometric shape functions for each output point.
        nbrNodesQ1 (int): Number of corner nodes (base geometric order nodes).

    Returns:
        np.array: New node positions excluding corner nodes.
    """
    nbrOutPnts = len(outputPntsMappedCoords)
    dim = len(outputPntsMappedCoords[0])
    outputPntCoords = np.zeros((nbrOutPnts, dim))
    for iPnt in range(nbrOutPnts):
        for iNode in range(len(element_nodes)):
            for iDim in range(dim):
                outputPntCoords[iPnt][iDim] += geoShapeFuncs[iPnt][iNode] * element_nodes[iNode][iDim]
    return outputPntCoords[nbrNodesQ1:]

def clean_data(nodes, connectivity):
    """
    Clean up the node data and connectivity by removing unused nodes and updating indices.

    Parameters:
        nodes (np.array): The original node data.
        connectivity (np.array): The original connectivity data.

    Returns:
        tuple: Cleaned nodes and connectivity arrays.
    """
    zero_based_connectivity = connectivity - 1
    max_index = nodes.shape[0]
    is_node_used = np.zeros(max_index, dtype=bool)
    np.put(is_node_used, zero_based_connectivity.ravel(), True)
    new_indices = np.cumsum(is_node_used) - 1
    cleaned_connectivity = new_indices[zero_based_connectivity]
    cleaned_nodes = nodes[is_node_used]
    cleaned_connectivity += 1  # Convert back to 1-based indexing
    return cleaned_nodes, cleaned_connectivity