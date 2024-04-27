import numpy as np
from input_data import read_input_data, get_data_info
from cgns_operations import create_new_cgns_tree, save_cgns_file, create_base, create_zone, write_coordinates, write_connectivity, write_solution
from mesh_upgrade import upgrade_mesh
from solution_processing import interpolate_to_nodes
from cgns_utils import preprocess_coordinates, get_CG_elemtype

def main():
# -----------------------------------------------------
    filename = "./examples/example1.CFmesh"
    output_filename = "./examples/output1.cgns" 

    # Load input data
    nodes, connectivity, solution_data, metadata = read_input_data(filename)
    
    # Get general mesh and solution parameters 
    dim, nbEqs, geoOrder, solOrder, nbElem, ElemType = get_data_info(metadata, filename)
    var_names= ["rho", "rhoU", "rhoV", "rhoE"]
    cell_dim = dim
    phys_dim = dim
# -----------------------------------------------------
    # Start by creating a new CGNS tree (Open CGNS file)
    tree = create_new_cgns_tree()
    print("CGNS Tree created.")

    # Create CGNS base
    base_node = create_base(tree, "Base", cell_dim, phys_dim)
    if base_node is None:
        raise Exception("Failed to create base")

    # Check if mesh upgrade is needed and perform it
    upgraded_nodes, upgraded_connectivity, upgraded_geoOrder = upgrade_mesh(nodes, connectivity, solOrder, geoOrder, ElemType)

    # Define CGNS zone
    zone_size =  np.array([[len(upgraded_nodes), nbElem, 0]], dtype=np.int32)  # Specify the size of the zone
    zone_type = "Unstructured"
    zone_node = create_zone(base_node, "Zone1", zone_size, zone_type)
    if zone_node is None:
        raise Exception("Failed to create zone")

    # Write mesh data to CGNS
    coordX, coordY, coordZ = preprocess_coordinates(upgraded_nodes)
    write_coordinates(zone_node, coordX, coordY, coordZ)
    
    element_name = ElemType
    CGelement_type = get_CG_elemtype(ElemType, upgraded_geoOrder)
    element_range = np.array([1, nbElem], dtype=np.int32) 

    output_connectivity = np.concatenate(upgraded_connectivity) # Flatten the list of arrays into one single array

    connectivity_node = write_connectivity(zone_node, element_name, CGelement_type, output_connectivity, element_range)
    if connectivity_node is None:
        raise Exception("Failed to add connectivity")

    # Define and write solution data
    output_solution_data = interpolate_to_nodes(upgraded_nodes, upgraded_connectivity, solution_data, ElemType, upgraded_geoOrder, solOrder, var_names, nbEqs)
    solution_name = "SolutionField"
    write_solution(zone_node, solution_name, output_solution_data)

    # Save the tree to a file / Close CGNS file
    save_cgns_file(tree, output_filename)

if __name__ == "__main__":
    main()

