import numpy as np
from CGNS.PAT import cgnskeywords as CK

def preprocess_coordinates(upgradedNodes):
    """
    Extract X, Y, and optional Z coordinates from a list of coordinate arrays.
    
    Args:
    upgradedNodes : list of numpy arrays
        Each array contains coordinates for all nodes in one dimension.
    
    Returns:
    coordX, coordY, coordZ (optional) : numpy arrays
        Separate arrays for X, Y, and Z coordinates.
    """
    coordX = np.array([node[0] for node in upgradedNodes])
    coordY = np.array([node[1] for node in upgradedNodes])
    coordZ = np.array([node[2] for node in upgradedNodes]) if len(upgradedNodes[0]) > 2 else None

    return coordX, coordY, coordZ


def get_CG_elemtype(element_type, geoOrder):
    """
    Get the CGNS element type based on the element type and geometric order.

    Args:
        element_type (str): The type of element ('Quad', 'Triag', 'Prism', 'Tetra', 'Hexa').
        geoOrder (int): Geometric order of the element.

    Returns:
        str: CGNS element type.
    """
    # Mapping dictionary for element types to their CGNS counterparts based on geometric order
    elemtype_map = {
        'Quad': {
            1: CK.QUAD_4,
            2: CK.QUAD_9,
            3: CK.QUAD_16,
            4: CK.QUAD_25
        },
        'Triag': {
            1: CK.TRI_3,
            2: CK.TRI_6,
            3: CK.TRI_10,
            4: CK.TRI_15
        },
        'Prism': {
            1: CK.PENTA_6,
            2: CK.PENTA_18,
            3: CK.PENTA_40,
            4: CK.PENTA_75
        },
        'Tetra': {
            1: CK.TETRA_4,
            2: CK.TETRA_10,
            3: CK.TETRA_20,
            4: CK.TETRA_35
        },
        'Hexa': {
            1: CK.HEXA_8,
            2: CK.HEXA_27,
            3: CK.HEXA_64,
            4: CK.HEXA_125
        }
    }

    # Retrieve the correct CGNS element type based on the provided element type and geometric order
    if element_type in elemtype_map and geoOrder in elemtype_map[element_type]:
        return elemtype_map[element_type][geoOrder]
    else:
        raise ValueError("Unsupported element type or geometric order")


def get_outputPnts_mapped_coords(element_type):
    """
    Get the mapped coordinates of output points for a given element type following CGNS conventions.

    Args:
        str: CGNS element type.

    Returns:
        np.array: An array of mapped coordinates.
    """
    coordinates = {
        CK.TRI_3: np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        CK.TRI_6: np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]),
        # Add more definitions for other element types...
        CK.QUAD_4: np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
        CK.QUAD_9: np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]]),
        CK.QUAD_16: np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0/3.0, -1.0], [1.0/3.0, -1.0], [1.0, -1.0/3.0], [1.0, 1.0/3.0], [1.0/3.0, 1.0], [-1.0/3.0, 1.0], [-1.0, 1.0/3.0], [-1.0, -1.0/3.0], [-1.0/3.0, -1.0/3.0], [1.0/3.0, -1.0/3.0], [1.0/3.0, 1.0/3.0], [-1.0/3.0, 1.0/3.0]]),
        CK.QUAD_25: np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-0.5, -1.0], [0.0, -1.0], [0.5, -1.0], [1.0, -0.5], [1.0, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 1.0], [-0.5, 1.0], [-1.0, 0.5], [-1.0, 0.0], [-1.0, -0.5], [-0.5, -0.5], [0.0, -0.5], [0.5, -0.5], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [-0.5, 0.5], [-0.5, 0.0], [0.0, 0.0]]),
            # Similarly for TETRA, HEXA, etc.
        }

    try:
        return coordinates[element_type]
    except KeyError:
        raise ValueError(f"Element type {element_type} is not supported")


def get_cgns_cell_node_conn(element_type):
    """
    Get the CGNS node connectivity order for different element types.
    This is needed in case input ordering is different from CGNS conventions.
    This function is hard-coded for COOLFluiD CFmesh conventions
    Args:
    element_type (str): Type of the element (e.g., 'TRI_3', 'QUAD_4', 'TETRA_10', etc.)

    Returns:
    list[int]: A list representing the node order for CGNS.
    """
    mappings = {
        CK.TRI_3: [0, 1, 2],
        CK.TRI_6: [0, 1, 2, 3, 4, 5],
        CK.QUAD_4: [0, 1, 2, 3],
        CK.QUAD_9: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        CK.TETRA_4: [0, 1, 2, 3],
        CK.TETRA_10: [0, 1, 2, 3, 4, 5, 6, 9, 7, 8],
        CK.PENTA_6: [0, 1, 2, 3, 4, 5],
        CK.PENTA_18: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 10, 12, 14],
        CK.HEXA_8: [0, 1, 2, 3, 4, 5, 6, 7],
        CK.HEXA_27: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 22, 23, 24, 25, 12, 20, 16, 18, 14, 26, 21]
    }
    if element_type in mappings:
        return mappings[element_type]
    else:
        raise ValueError(f"Unsupported ElementType: {element_type} in get_cgns_cell_node_conn")
