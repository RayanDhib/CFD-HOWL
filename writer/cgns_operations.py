import numpy as np
import CGNS.PAT.cgnslib as CGL
import CGNS.MAP
from CGNS.PAT import cgnskeywords as CK

def load_cgns_file(filename):
    """
    Load a CGNS file and return its tree, links, and paths.

    Parameters:
        filename (str): The path to the CGNS file to be loaded.

    Returns:
        tuple: A tuple containing the CGNS tree, links, and paths if successful, or
               three None values if an error occurs.
    """
    try:
        tree, links, paths = CGNS.MAP.load(filename)
        print("CGNS file loaded successfully.")
        return tree, links, paths
    except Exception as e:
        print(f"Error loading CGNS file: {e}")
        return None, None, None

def create_new_cgns_tree():
    """
    Create a new CGNS tree, analogous to creating a new file.

    Returns:
        CGNSNode: A new CGNS tree node.
    """
    return CGL.newCGNSTree()

def save_cgns_file(tree, filename):
    """
    Save a CGNS tree to a specified file.

    Parameters:
        tree (CGNSNode): The CGNS tree to be saved.
        filename (str): The path where the CGNS file will be saved.
    """
    try:
        CGNS.MAP.save(filename, tree)
        print("CGNS file saved successfully.")
    except Exception as e:
        print(f"Error saving CGNS file: {e}")

def create_base(tree, name, cell_dim, phys_dim):
    """
    Create a CGNS base node within a given CGNS tree.

    Parameters:
        tree (CGNSNode): The parent CGNS tree where the base will be added.
        name (str): The name of the new base.
        cell_dim (int): The cell dimensionality.
        phys_dim (int): The physical dimensionality.

    Returns:
        CGNSNode: The newly created base node, or None if an error occurs.
    """
    try:
        base_node = CGL.newCGNSBase(tree, name, cell_dim, phys_dim)
        print("Base node created successfully.")
        return base_node
    except Exception as e:
        print(f"Error creating base node: {e}")
        return None

def create_zone(base_node, zone_name, zone_size, zone_type='Unstructured'):
    """
    Create a CGNS zone node within a base node.

    Parameters:
        base_node (CGNSNode): The parent base node where the zone will be added.
        zone_name (str): Name of the zone.
        zone_size (numpy.ndarray): Size of the zone, an array describing dimensions.
        zone_type (str): Type of the zone ('Unstructured' or 'Structured').

    Returns:
        CGNSNode: The newly created zone node, or None if an error occurs.
    """
    try:
        if zone_type not in CK.ZoneType_l:
            raise ValueError("Unsupported zone type. Use 'Structured' or 'Unstructured'.")
        if not isinstance(zone_size, np.ndarray) or zone_size.ndim != 2 or zone_size.shape[1] != 3:
            raise ValueError("Zone size must be a numpy array of shape (IndexDimension, 3)")

        zone_node = CGL.newZone(base_node, zone_name, zone_size, zone_type)
        print("Zone created successfully.")
        return zone_node
    except Exception as e:
        print(f"Error creating zone: {e}")
        return None

def write_coordinates(zone_node, coordX, coordY, coordZ=None):
    """
    Write nodal coordinate data to a CGNS zone.

    Parameters:
        zone_node (CGNSNode): The zone node to which the coordinates will be added.
        coordX (numpy.ndarray): Array of X coordinates.
        coordY (numpy.ndarray): Array of Y coordinates.
        coordZ (numpy.ndarray, optional): Array of Z coordinates, if applicable.

    """
    try:
        grid_coords = CGL.newGridCoordinates(zone_node, "GridCoordinates")
        CGL.newCoordinates(zone_node, 'CoordinateX', coordX)
        CGL.newCoordinates(zone_node, 'CoordinateY', coordY)
        if coordZ is not None:
            CGL.newCoordinates(zone_node, 'CoordinateZ', coordZ)
        print("Coordinates written successfully.")
    except Exception as e:
        print(f"Error writing coordinates: {e}")
        raise 

def write_connectivity(zone_node, element_name, element_type, connectivity_array, element_range=None):
    """
    Add connectivity information to a CGNS zone.

    Parameters:
        zone_node (CGNSNode): The zone node where elements are defined.
        element_name (str): Name for the element data.
        element_type (int): Type of elements in CGNS standards (e.g., 'TRI_3', 'QUAD_9').
        connectivity_array (numpy.ndarray): Array describing the connectivity of the elements.
        element_range (tuple, optional): Range of indices covered by these elements.

    Returns:
        CGNSNode: The elements node containing the connectivity, or None if an error occurs.
    """
    try:
        elements_node = CGL.newElements(zone_node, element_name, element_type, element_range, connectivity_array)
        print("Connectivity added successfully.")
        return elements_node
    except Exception as e:
        print(f"Error adding connectivity: {e}")
        return None

def write_solution(zone_node, solution_name, solution_data):
    """
    Write solution data to a CGNS zone.

    Parameters:
        zone_node (CGNSNode): The zone node where the solution will be written.
        solution_name (str): Name of the solution node.
        solution_data (dict): Dictionary of variable names and their corresponding data arrays.
    """
    try:
        solution_node = CGL.newFlowSolution(zone_node, solution_name, 'Vertex')
        for var_name, data_array in solution_data.items():
            CGL.newDataArray(solution_node, var_name, data_array)
        print("Solution data added successfully.")
    except Exception as e:
        print(f"Error writing solution data: {str(e)}")
