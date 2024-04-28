import numpy as np

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
    if extension == "cfmesh":
        return read_cfmesh(filename)
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
    if extension == "cfmesh":
        return get_info_cfmesh(metadata)
    else:
        raise ValueError(f"Unsupported file format: {extension}")
        
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
    element_states = []  # Store state indices for each element
    nodes = []
    #geom_ents = []
    states = []  # This will store all state data temporarily
    #var_data = []

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
                nb_nodes_per_type = int(metadata['!NB_NODES_PER_TYPE'][0])  # Assumes there is only one type of elements
                state_indices = []
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

    solution_data = []
    for elem_indices in element_states:
        element_state_data = []
        for idx in elem_indices:
            element_state_data.append(states[idx])
    
        solution_data.append(element_state_data)

    return nodes, connectivity, solution_data, metadata
    

