import json
import numpy as np
from writer.MeshData import MeshData
from writer.input_data import read_input_data, get_data_info
from writer.cgns_operations import create_new_cgns_tree, save_cgns_file, create_base, create_zone, write_coordinates, write_connectivity, write_solution
from writer.cgns_utils import preprocess_coordinates, get_CG_elemtype
from writer.solution_processing import interpolate_to_nodes

def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

def main(config_file='config.json'):
    # Load the top-level configuration
    config = load_config(config_file)

    # Load the active configuration specified in the top-level config
    active_config = load_config(config['active_config'])

    # Load input data
    nodes, connectivity, solution_data, metadata = read_input_data(active_config['filename'])
    
    # Get general mesh and solution parameters 
    dim, nbEqs, geoOrder, solOrder, nbElem, elem_type = get_data_info(metadata, active_config['filename'])
    var_names = active_config['var_names']
    cell_dim = phys_dim = dim

    # Initialize MeshData
    mesh = MeshData(nodes, connectivity, elem_type, geoOrder, solOrder)

    # Create a new CGNS tree
    tree = create_new_cgns_tree()
    print("CGNS Tree created.")

    # Create CGNS base
    base_node = create_base(tree, "Base", cell_dim, phys_dim)
    if base_node is None:
        raise Exception("Failed to create base")

    # Upgrade mesh if necessary and preprocess coordinates
    mesh.upgrade_mesh()
    coordX, coordY, coordZ = preprocess_coordinates(mesh.nodes)

    # Define CGNS zone
    zone_size = np.array([[mesh.num_nodes, mesh.num_elements, 0]], dtype=np.int32)
    zone_type = "Unstructured"
    zone_node = create_zone(base_node, "Zone1", zone_size, zone_type)
    if zone_node is None:
        raise Exception("Failed to create zone")

    # Write mesh data to CGNS
    write_coordinates(zone_node, coordX, coordY, coordZ)
    
    element_name = mesh.elem_type
    CGelement_type = get_CG_elemtype(mesh.elem_type, mesh.geo_order)
    element_range = np.array([1, mesh.num_elements], dtype=np.int32)

    # Prepare connectivity for CGNS format
    output_connectivity = np.concatenate(mesh.connectivity)
    connectivity_node = write_connectivity(zone_node, element_name, CGelement_type, output_connectivity, element_range)
    if connectivity_node is None:
        raise Exception("Failed to add connectivity")

    # Define and write solution data
    output_solution_data = interpolate_to_nodes(mesh, solution_data, var_names, nbEqs)
    solution_name = "SolutionField"
    write_solution(zone_node, solution_name, output_solution_data)

    # Save the CGNS tree to a file
    save_cgns_file(tree, active_config['output_filename'])

if __name__ == "__main__":
    main()
