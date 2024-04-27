import numpy as np
import CGNS.PAT.cgnslib as CGL
import CGNS.MAP
from CGNS.PAT import cgnskeywords as CK

# File Operations

def load_cgns_file(filename):
    """
    Load a CGNS file and return its tree, links, and paths.
    """
    try:
        tree, links, paths = CGNS.MAP.load(filename)
        print("CGNS file loaded successfully.")
        return tree, links, paths
    except Exception as e:
        print(f"Error loading CGNS file: {str(e)}")
        return None, None, None

def create_new_cgns_tree():
    """
    Create a new CGNS tree (like creating a new file).
    """    
    return CGL.newCGNSTree()

def save_cgns_file(tree, filename):
    """
    Save a CGNS tree to a file.
    """
    try:
        CGNS.MAP.save(filename, tree)
        print("CGNS file saved successfully.")
    except Exception as e:
        print(f"Error saving CGNS file: {str(e)}")
        
# Base Operations
def create_base(tree, name, cell_dim, phys_dim):
    """
    Create a CGNS base node in a CGNS tree. 
    tree: The parent CGNS tree where the base will be added.
    name: The name of the new base.
    cell_dim: The cell dimensionality (integer).
    phys_dim: The physical dimensionality (integer).
    """
    try:
        base_node = CGL.newCGNSBase(tree, name, cell_dim, phys_dim)
        print("Base node created successfully.")
        return base_node
    except Exception as e:
        print(f"Error creating base node: {str(e)}")
        return None

# Zone Operations
def create_zone(base_node, zone_name, zone_size, zone_type='Unstructured'):
    """
    Create a CGNS zone node within a base node.
    base_node: The parent base node where the zone will be added.
    zone_name: Name of the zone.
    zone_size: Size of the zone (typically a numpy array describing dimensions).
    zone_type: Type of the zone (e.g., 'Unstructured' or 'Structured').
    """
    try:
        # Ensure the zone type is supported
        if zone_type not in CK.ZoneType_l:
            raise ValueError("Unsupported zone type. Use 'Structured' or 'Unstructured'.")

        # Check if zone_size is a numpy array and has correct dimensions
        if not isinstance(zone_size, np.ndarray) or zone_size.ndim != 2 or zone_size.shape[1] != 3:
            raise ValueError("zone_size must be a numpy array of shape (IndexDimension, 3)")

        # Create the zone node
        zone_node = CGL.newZone(base_node, zone_name, zone_size, zone_type)
        print("Zone created successfully.")
        return zone_node
    except Exception as e:
        print(f"Error creating zone: {e}")
        return None

# Coordinate Operations
def write_coordinates(zone_node, coordX, coordY, coordZ=None):
    """
    Write nodal coordinate data to a CGNS zone.
    zone_node: The zone node to which the coordinates will be added.
    coordX: Array of X coordinates.
    coordY: Array of Y coordinates.
    coordZ: Optional array of Z coordinates (for 3D).
    """
    try:
        # Create grid coordinates node
        grid_coords = CGL.newGridCoordinates(zone_node, "GridCoordinates")
        
        # Create coordinate arrays
        CGL.newCoordinates(zone_node, 'CoordinateX', coordX)
        CGL.newCoordinates(zone_node, 'CoordinateY', coordY)
        if coordZ is not None:
            CGL.newCoordinates(zone_node, 'CoordinateZ', coordZ)
        
        print("Coordinates written successfully.")
    except Exception as e:
        print(f"Error writing coordinates: {str(e)}")


# Connectivity Operations
def write_connectivity(zone_node, element_name, element_type, connectivity_array, element_range=None):
    """
    Add connectivity information to a CGNS zone.
    zone_node: The zone node where elements are defined.
    element_name: Name for the element data.
    element_type: Type of elements (e.g., 'TETRA', 'HEXA').
    connectivity_array: Array describing the connectivity of the elements.
    element_range: Optional parameter describing the range of indices covered by these elements.
    """
    try:
        elements_node = CGL.newElements(zone_node, element_name, element_type, element_range, connectivity_array)
        print("Connectivity added successfully.")
        return elements_node
    except Exception as e:
        print(f"Error adding connectivity: {str(e)}")
        return None


# Solution Operations
def write_solution(zone_node, solution_name, solution_data):
    """
    Write solution data to a CGNS zone.
    zone_node: The zone node where the solution will be written.
    solution_name: Name of the solution node.
    solution_data: Dictionary of variable names and their corresponding data arrays.
    """
    try:
        solution_node = CGL.newFlowSolution(zone_node, solution_name, 'Vertex')
        for var_name, data_array in solution_data.items():
            CGL.newDataArray(solution_node, var_name, data_array)
        
        print("Solution data added successfully.")
    except Exception as e:
        print(f"Error writing solution data: {str(e)}")

