import numpy as np

def read_input_data(filename):
    """
    General function to read input data, delegating to specific reader based on file extension.
    """
    extension = filename.split('.')[-1].lower()
    if extension == "cfmesh":
        return read_cfmesh(filename)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

def get_data_info(metadata, filename):
    """
    Extracts general mesh and solution parameters from metadata.
    """
    extension = filename.split('.')[-1].lower()
    if extension == "cfmesh":
        return get_info_cfmesh(metadata)
    else:
        raise ValueError(f"Unsupported file format: {extension}")
        
def get_info_cfmesh(metadata):
    """
    Extracts from CFmesh general mesh and solution parameters from metadata.
    """
    dim = int(metadata.get('!NB_DIM', [0])[0])
    nbEqs = int(metadata.get('!NB_EQ', [0])[0])
    geoOrder = int(metadata.get('!GEOM_POLYORDER', [1])[0])
    solOrder = int(metadata.get('!SOL_POLYORDER', [1])[0])
    nbElem = int(metadata.get('!NB_ELEM', ['0'])[0])
    ElemType = metadata.get('!ELEM_TYPES', [''])[0].strip() 
    return dim, nbEqs, geoOrder, solOrder, nbElem, ElemType

def read_cfmesh(in_fname):
    metadata = {}
    connectivity = []
    element_states = []  # Store state indices for each element
    nodes = []
    geom_ents = []
    states = []  # This will store all state data temporarily
    var_data = []

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
                        break  # end of section or file
                    elements = line.split(' ')
                    try:
                        # Slicing the line based on the number of nodes specified
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

    # Link state data to elements using indices stored in element_states
    # Initialize an empty list to store the linked state data for each element
    solution_data = []

    # Loop through each set of indices stored in element_states
    for elem_indices in element_states:
        # Initialize a temporary list to hold the states for the current element
        element_state_data = []
    
        # Loop through each index in the current set of elem_indices
        for idx in elem_indices:
            # Append the state data corresponding to the current index to the temporary list
            element_state_data.append(states[idx])
    
        # Append the list of state data for the current element to the solution_data list
        solution_data.append(element_state_data)

    return nodes, connectivity, solution_data, metadata
    

