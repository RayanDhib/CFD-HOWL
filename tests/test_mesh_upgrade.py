# tests/test_mesh_upgrade.py
import unittest
import numpy as np
from unittest.mock import patch
from writer.mesh_upgrade import upgrade_geoOrder, build_node_tree, check_node_existence, preprocess_connectivity, compute_new_nodes_position, clean_data
from writer.utils import compute_geoShape_function

class TestMeshUpgrade(unittest.TestCase):

    def test_upgrade_geoOrder(self):
        # Mock data setup
        nodes = np.array([[0, 0], [1, 0], [0, 1]])
        connectivity = [[1, 2, 3]]
        old_geoOrder = 1
        new_geoOrder = 2
        ElemType = 'Triag'

        # Perform the function call
        upgraded_nodes, upgraded_connectivity = upgrade_geoOrder(nodes, connectivity, old_geoOrder, new_geoOrder, ElemType)
        expected_number_of_nodes = 6
        expected_connectivity_shape = (1,6)
        # Assertions to check the correct function behavior
        self.assertEqual(len(upgraded_nodes), expected_number_of_nodes)
        self.assertEqual(upgraded_connectivity.shape, expected_connectivity_shape)

    def test_check_node_existence(self):
        nodes = np.array([[0, 0], [1, 1]])
        tree = build_node_tree(nodes)
        new_pos = np.array([0, 0])
        exists, idx = check_node_existence(tree, new_pos)
        self.assertTrue(exists)
        self.assertEqual(idx, 0)

    def test_preprocess_connectivity(self):
        connectivity = [[0, 1, 2]]
        CGelement_type = 5 #CK.TRI_3
        processed = preprocess_connectivity(connectivity, CGelement_type)
        self.assertEqual(processed, [[1, 2, 3]])

    def test_compute_new_nodes_position(self):
        outputPntsMappedCoords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
        element_nodes = np.array([[0, 0], [1, 0], [0, 1]])
        geoShapeFuncs = [compute_geoShape_function(coord, 'Triag', 1) for coord in outputPntsMappedCoords]
        nbrNodesQ1 = 3
        new_positions = compute_new_nodes_position(outputPntsMappedCoords, element_nodes, geoShapeFuncs, nbrNodesQ1)
        self.assertEqual(new_positions.shape, (3, 2))

    def test_clean_data(self):
        nodes = np.array([[0, 0], [1, 0], [0, 0], [0, 1], [1, 1]])
        connectivity = np.array([[1, 2, 4], [1, 4, 5]])
        cleaned_nodes, cleaned_connectivity = clean_data(nodes, connectivity)
        self.assertEqual(len(cleaned_nodes), 4)
        self.assertTrue(np.array_equal(cleaned_connectivity, np.array([[1, 2, 3], [1, 3, 4]])))

if __name__ == '__main__':
    unittest.main()
