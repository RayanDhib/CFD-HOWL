import numpy as np
from utils import compute_geoShape_function
from cgns_utils import get_cgns_cell_node_conn, get_CG_elemtype, get_outputPnts_mapped_coords
from scipy.spatial import cKDTree

def upgrade_mesh(nodes, connectivity, solOrder, geoOrder, ElemType):
    upgraded_connectivity = preprocess_connectivity(connectivity,  get_CG_elemtype(ElemType, geoOrder)) #Needed if connectivity has 0-based indexing instead of 1-based indexing and if numbering convention is different from CGNS conventions
    upgraded_nodes = nodes
    upgraded_geoOrder = geoOrder
    if (solOrder > geoOrder):
        upgraded_geoOrder = min(solOrder, 4)  # Maximum geometric order = 4
        upgraded_nodes, upgraded_connectivity = upgrade_geoOrder(nodes, upgraded_connectivity, geoOrder, upgraded_geoOrder, ElemType)
    if (upgraded_geoOrder  == 3) and (geoOrder == 2): # if we upgrade from Q2 to Q3
        upgraded_nodes,upgraded_connectivity = clean_data(upgraded_nodes, upgraded_connectivity)
    return upgraded_nodes,upgraded_connectivity, upgraded_geoOrder

def upgrade_geoOrder(nodes, connectivity, old_geoOrder, new_geoOrder, ElemType):
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
    """Check if a node exists in the tree and return its index or indicate non-existence."""
    closest_dist, closest_idx = node_tree.query(new_pos)
    if closest_dist < tolerance:
        return True, closest_idx
    return False, None


def preprocess_connectivity(connectivity, CGelement_type):
    """
    Adjust the connectivity array for CGNS compatibility.

    Args:
    connectivity (list[list[int]]): Original connectivity from mesh.
    element_type (str): Type of the elements in CG.

    Returns:
    list[list[int]]: Connectivity adjusted for CGNS ordering and 1-based indexing.
    """
    cgns_order = get_cgns_cell_node_conn(CGelement_type)
    nbrElems = len(connectivity)
    nbrNodes = len(cgns_order)
    nodal_connectivity = [[0]*nbrNodes for _ in range(nbrElems)]

    for iElem in range(nbrElems):
        for j, cgns_idx in enumerate(cgns_order):
            nodal_connectivity[iElem][j] = connectivity[iElem][cgns_idx] + 1  # 1-based index adjustment

    return nodal_connectivity



def compute_new_nodes_position(outputPntsMappedCoords, element_nodes, geoShapeFuncs, nbrNodesQ1):
    """
    Compute new nodes position based on geometric shape functions.

    Args:
    outputPntsMappedCoords (np.array): Mapped coordinates for output points.
    element_nodes (np.array): Node coordinates for the current element.
    geoShapeFuncs (list): Geometric shape functions for each output point.
    nbrNodesQ1 (int): Number of corner nodes (base geometric order nodes).

    Returns:
    np.array: New node positions excluding corner nodes.
    """
    nbrOutPnts = len(outputPntsMappedCoords)
    dim = len(outputPntsMappedCoords[0])  # Assuming each coord is a row in the array
    
    # Initialize an array for output point coordinates
    outputPntCoords = np.zeros((nbrOutPnts, dim))
    
    # Evaluate node coordinates at the output points
    for iPnt in range(nbrOutPnts):
        for iNode in range(len(element_nodes)):  # Loop over each node in the element
            for iDim in range(dim):  # Loop over each dimension
                outputPntCoords[iPnt][iDim] += geoShapeFuncs[iPnt][iNode] * element_nodes[iNode][iDim]

    # Exclude corner nodes based on nbrNodesQ1
    new_node_positions = outputPntCoords[nbrNodesQ1:]
    
    return new_node_positions


def clean_data(nodes, connectivity):
    # Convert 1-based connectivity indices to 0-based
    zero_based_connectivity = connectivity - 1

    # Determine which nodes are used
    max_index = nodes.shape[0]
    is_node_used = np.zeros(max_index, dtype=bool)
    np.put(is_node_used, zero_based_connectivity.ravel(), True)

    # Create a mapping from old indices to new indices
    new_indices = np.cumsum(is_node_used) - 1

    # Update the connectivity with new indices
    cleaned_connectivity = new_indices[zero_based_connectivity]

    # Filter the nodes to include only those that are used
    cleaned_nodes = nodes[is_node_used]

    # Convert back to 1-based indexing for the cleaned connectivity
    cleaned_connectivity += 1

    return cleaned_nodes, cleaned_connectivity    