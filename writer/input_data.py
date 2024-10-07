import numpy as np
import math
import h5py
import os

# ---------------------------- General Main Functions ----------------------------

def read_input_data(filename):
    """
    Read input data from a file, choosing the appropriate reader based on file extension.

    Parameters:
        filename (str): Path to the input file.

    Returns:
        tuple: Nodes, connectivity, solution data, and metadata extracted from the file.
        
    Raises:
        ValueError: If the file format is not supported.
    """
    extension = filename.split('.')[-1].lower()
    
    # CFmesh Format (COOLFluiD)
    if extension == "cfmesh":
        return read_cfmesh(filename)

    # pyfrs Format (PyFR)
    elif extension == "pyfrs":
        # Assume that the mesh file is in the same directory with the same base name and .msh extension
        mesh_filename = filename.replace('.pyfrs', '.msh')
        if not os.path.exists(mesh_filename):
            raise ValueError(f"PyFR solution file '{filename}' requires a corresponding mesh file '{mesh_filename}'.")
        return read_msh_and_pyfr(mesh_filename, filename)

    # Placeholder for New Format 1
    # elif extension == "newformat1":
    #     return read_newformat1(filename)

    # Placeholder for New Format 2
    # elif extension == "newformat2":
    #     # Custom implementation for new format 2
    #     return read_newformat2(filename)

    else:
        raise ValueError(f"Unsupported file format: {extension}")

def get_data_info(metadata, filename):
    """
    Extract mesh and solution parameters from metadata based on file extension.

    Parameters:
        metadata (dict): Metadata dictionary containing the data details.
        filename (str): Path to the file, used to determine handling based on extension.

    Returns:
        tuple: Dimension, number of equations, geometric order, solution order, number of elements, and element type.
        
    Raises:
        ValueError: If the file format is not supported.
    """
    extension = filename.split('.')[-1].lower()

    # CFmesh Format (COOLFluiD)
    if extension == "cfmesh":
        return get_info_cfmesh(metadata)

    # PyFR Format
    elif extension in ["pyfrm", "pyfrs"]:
        return get_info_pyfr(metadata)

    # Placeholder for New Format 1
    # elif extension == "newformat1":
    #     return get_info_newformat1(metadata)

    # Placeholder for New Format 2
    # elif extension == "newformat2":
    #     return get_info_newformat2(metadata)

    else:
        raise ValueError(f"Unsupported file format: {extension}")

# ---------------------------- Format-Specific Extraction Functions ----------------------------

# ----------- CFmesh Format (COOLFluiD) Functions -----------

def get_info_cfmesh(metadata):
    """
    Extract parameters from CFmesh metadata.

    Parameters:
        metadata (dict): Metadata dictionary specific to CFmesh.

    Returns:
        tuple: Dimension, number of equations, geometric order, solution order, number of elements, and element type.
    """
    dim = int(metadata.get('!NB_DIM', [0])[0])
    nbEqs = int(metadata.get('!NB_EQ', [0])[0])
    geoOrder = int(metadata.get('!GEOM_POLYORDER', [1])[0])
    solOrder = int(metadata.get('!SOL_POLYORDER', [1])[0])
    nbElem = int(metadata.get('!NB_ELEM', ['0'])[0])
    ElemType = metadata.get('!ELEM_TYPES', [''])[0].strip() 
    return dim, nbEqs, geoOrder, solOrder, nbElem, ElemType

def read_cfmesh(in_fname):
    """
    Read mesh data from a CFmesh file including nodes, element connectivity, and states.

    Parameters:
        in_fname (str): Path to the CFmesh file.

    Returns:
        tuple: Nodes, connectivity, solution data, and metadata from the CFmesh file.
    """
    metadata = {}
    connectivity = []
    element_states = []
    nodes = []
    states = []

    with open(in_fname, 'r') as in_file:
        line = 1
        while line:
            line = in_file.readline().strip()
            if not line:
                break  # end of file reached
            mData = line.split(' ')
            if len(mData) > 1:
                if mData[0] in metadata:
                    metadata[mData[0]].extend(mData[1:])
                else:
                    metadata[mData[0]] = mData[1:]
            if mData[0] == "!LIST_ELEM":
                nb_nodes_per_type = int(metadata['!NB_NODES_PER_TYPE'][0])
                while True:
                    n_line = in_file.tell()
                    line = in_file.readline().strip()
                    if not line:
                        break
                    elements = line.split(' ')
                    try:
                        elem_connectivity = np.array(elements[:nb_nodes_per_type]).astype(int)
                        state_indices = np.array(elements[nb_nodes_per_type:]).astype(int)
                        connectivity.append(elem_connectivity)
                        element_states.append(state_indices)
                    except ValueError:
                        in_file.seek(n_line)
                        break
            elif mData[0] == "!LIST_NODE":
                while True:
                    n_line = in_file.tell()
                    line = in_file.readline().strip()
                    if not line:
                        break
                    line = line.split(' ')
                    try:
                        array = np.array(line).astype(np.double)
                        nodes.append(array)
                    except ValueError:
                        in_file.seek(n_line)
                        break
            elif mData[0] == '!LIST_STATE':
                while True:
                    n_line = in_file.tell()
                    line = in_file.readline().strip()
                    if not line:
                        break
                    line = line.split(' ')
                    try:
                        array = np.array(line).astype(np.double)
                        states.append(array)
                    except ValueError:
                        in_file.seek(n_line)
                        break

    solution_data = [
        [states[idx] for idx in elem_indices]
        for elem_indices in element_states
    ]
    
    return nodes, connectivity, solution_data, metadata

# ----------- PyFR Format Functions -----------

def get_info_pyfr(metadata):
    """
    Extract parameters from PyFR metadata.

    Parameters:
        metadata (dict): Metadata dictionary specific to PyFR.

    Returns:
        tuple: Dimension, number of equations, geometric order, solution order, number of elements, and element type.
    """
    dim = metadata.get('dim', 3)  # Default to 3 if not available
    nbEqs = metadata.get('nbEqs', 0)
    geoOrder = metadata.get('geoOrder', 1)
    solOrder = metadata.get('solOrder', 1)
    nbElem = metadata.get('nbElem', 0)
    ElemType = metadata.get('ElemType', 'Unknown')
    return dim, nbEqs, geoOrder, solOrder, nbElem, ElemType

def read_msh_and_pyfr(mesh_filename, soln_filename):
    """
    Read mesh and solution data from Gmsh and PyFR files.

    Parameters:
        mesh_filename (str): Path to the Gmsh mesh file (.msh).
        soln_filename (str): Path to the PyFR solution file (.pyfrs).

    Returns:
        tuple: Nodes, connectivity, solution data, and metadata from the files.
    """
    nodes, connectivity, mesh_metadata = read_msh(mesh_filename)

    solution_data = None
    solOrder = 1
    nbEqs = 0
    element_type = mesh_metadata.get('ElemType')

    with h5py.File(soln_filename, 'r') as soln_file:
        dataset = next((soln_file[key] for key in soln_file.keys() if key.startswith('soln_')), None)
        if dataset is None:
            raise ValueError(f"No solution dataset found in '{soln_filename}'.")

        data_shape = dataset.shape
        nbEqs = data_shape[1]
        num_solution_points = data_shape[0]
        solOrder = get_solOrder_pyfr(element_type, num_solution_points)

        reorder_indices = get_reorder_indices(element_type, solOrder)
        solution_data = np.array(dataset).transpose(2, 0, 1)[:, reorder_indices, :]

    solution_metadata = {
        'solOrder': solOrder,
        'nbEqs': nbEqs,
    }

    metadata = {**mesh_metadata, **solution_metadata}

    return nodes, connectivity, solution_data, metadata

def read_msh(mesh_filename):
    """
    Read nodes and element connectivity from a Gmsh mesh file (.msh), including metadata.

    Parameters:
        mesh_filename (str): Path to the Gmsh file.

    Returns:
        tuple: Nodes, connectivity, and metadata from the Gmsh file.
    """
    nodes = []
    connectivity = []
    dim = None
    geoOrder = 1  # Default to 1 for linear elements
    ElemType = None
    nbElem = 0

    # Mapping Gmsh element type descriptors to readable element names and geometric orders
    element_type_map = {
        2: ('Triag', 1), # 3-node triangle (P1)
        3: ('Quad', 1),     # 4-node quadrangle (P1)
        4: ('Tetra', 1),    # 4-node tetrahedron (P1)
        5: ('Hexa', 1),     # 8-node hexahedron (P1)
        6: ('Prism', 1),    # 6-node prism (P1)
        9: ('Triag', 2), # 6-node second order triangle (P2)
        11: ('Tetra', 2),   # 10-node second order tetrahedron (P2)
        16: ('Quad', 2),    # 8-node second order quadrangle (P2)
        17: ('Hexa', 2)     # 20-node second order hexahedron (P2)
    }

    with open(mesh_filename, 'r') as msh_file:
        inside_nodes_section = False
        inside_elements_section = False

        for line in msh_file:
            # Detect start of nodes section
            if line.startswith('$Nodes'):
                inside_nodes_section = True
                continue
            if line.startswith('$EndNodes'):
                inside_nodes_section = False
                continue

            # Detect start of elements section
            if line.startswith('$Elements'):
                inside_elements_section = True
                continue
            if line.startswith('$EndElements'):
                inside_elements_section = False
                continue

            # Extract nodes
            if inside_nodes_section:
                parts = line.strip().split()
                if len(parts) == 4:  # Index, x, y, z
                    # TODO: Extend support for 3D
                    nodes.append([float(parts[1]), float(parts[2])])#, float(parts[3])])

            # Extract elements (connectivity)
            if inside_elements_section:
                parts = line.strip().split()
                if len(parts) > 4:
                    element_type = int(parts[1])
                    if element_type in element_type_map:
                        element_name, order = element_type_map[element_type]
                        ElemType = element_name
                        geoOrder = order

                        # Extract element connectivity (node indices)
                        num_tags = int(parts[2])
                        element_connectivity = [int(index) - 1 for index in parts[3 + num_tags:]]
                        connectivity.append(element_connectivity)
                        nbElem += 1

    # Determine dimension based on node coordinates (checking if z coordinates are used)
    # TODO: Extend support for 3D
    dim = 2 #3 if any(node[2] != 0.0 for node in nodes) else 2

    nodes_array = np.array(nodes)
    connectivity_array = np.array(connectivity, dtype=object)  # Use dtype=object for jagged arrays

    metadata = {
        'dim': dim,
        'geoOrder': geoOrder,
        'nbElem': nbElem,
        'ElemType': ElemType
    }

    return nodes_array, connectivity_array, metadata
# ---------------------------- Utility Functions ----------------------------

def get_reorder_indices(element_type, order):
    """
    Generate reorder indices for an element based on type and polynomial order.
    This is used to convert from PyFR solution files.

    Parameters:
        element_type (str): The type of the element (e.g., 'Quad', 'Triag').
        order (int): Polynomial order (e.g., P1, P2, P3).

    Returns:
        list: List of indices for reordering the solution points to match the user's convention.
    
    Raises:
        NotImplementedError: If the element type is not supported.
    """
    if element_type.lower() == 'quad':
        num_points = order + 1
        original_indices = np.arange(num_points ** 2).reshape(num_points, num_points)
        reordered_indices = original_indices.flatten('F').tolist()
        return reordered_indices
    elif element_type.lower() == 'triag':
        raise NotImplementedError("Reordering for 'Triag' elements is not yet implemented.")
    else:
        raise NotImplementedError(f"Reordering for element type '{element_type}' is not yet implemented.")

def get_solOrder_pyfr(element_type, num_solution_points):
    """
    Determine the polynomial order (solOrder) for an element type based on the number of solution points.

    Parameters:
        element_type (str): The type of the element (e.g., 'Quad', 'Triag').
        num_solution_points (int): The number of solution points per element.

    Returns:
        int: The polynomial order (solOrder).

    Raises:
        ValueError: If the polynomial order cannot be determined for the given element type and number of solution points.
    """
    if element_type.lower() == 'quad':
        solOrder = int(math.sqrt(num_solution_points)) - 1
        if (solOrder + 1) ** 2 != num_solution_points:
            raise ValueError(f"Invalid number of solution points ({num_solution_points}) for a Quad element.")
        return solOrder
    elif element_type.lower() == 'triag':
        solOrder = int((-3 + math.sqrt(1 + 8 * num_solution_points)) / 2)
        if (solOrder + 1) * (solOrder + 2) // 2 != num_solution_points:
            raise ValueError(f"Invalid number of solution points ({num_solution_points}) for a Triag element.")
        return solOrder
    else:
        raise NotImplementedError(f"Element type '{element_type}' is not supported for polynomial order determination.")

