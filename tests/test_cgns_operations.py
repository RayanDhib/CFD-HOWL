# tests/test_cgns_operations.py
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from writer.cgns_operations import create_new_cgns_tree, save_cgns_file, create_base, create_zone, write_coordinates

class TestCgnsOperations(unittest.TestCase):

    @patch('CGNS.PAT.cgnslib.newCGNSTree')
    def test_create_new_cgns_tree(self, mock_new_tree):
        mock_new_tree.return_value = "new_tree"
        tree = create_new_cgns_tree()
        self.assertEqual(tree, "new_tree")

    @patch('CGNS.MAP.save')
    def test_save_cgns_file(self, mock_save):
        mock_tree = MagicMock()
        save_cgns_file(mock_tree, "output.cgns")
        mock_save.assert_called_with("output.cgns", mock_tree)

    @patch('CGNS.PAT.cgnslib.newCGNSBase')
    def test_create_base_success(self, mock_new_base):
        mock_tree = MagicMock()
        mock_new_base.return_value = "base_node"
        result = create_base(mock_tree, "Base1", 3, 3)
        mock_new_base.assert_called_with(mock_tree, "Base1", 3, 3)
        self.assertEqual(result, "base_node")

    @patch('CGNS.PAT.cgnslib.newCGNSBase')
    def test_create_base_failure(self, mock_new_base):
        mock_tree = MagicMock()
        mock_new_base.side_effect = Exception("Error creating base")
        result = create_base(mock_tree, "Base1", 3, 3)
        self.assertIsNone(result)

    @patch('CGNS.PAT.cgnslib.newZone')
    def test_create_zone_success(self, mock_new_zone):
        mock_base_node = MagicMock()
        zone_size = np.array([[10, 0, 0]])
        zone_type = "Unstructured"
        mock_new_zone.return_value = "zone_node"
        result = create_zone(mock_base_node, "Zone1", zone_size, zone_type)
        args, kwargs = mock_new_zone.call_args
        self.assertEqual(args[0], mock_base_node)
        self.assertEqual(args[1], "Zone1")
        np.testing.assert_array_equal(args[2], zone_size)
        self.assertEqual(args[3], "Unstructured")
        self.assertEqual(result, "zone_node")

    @patch('CGNS.PAT.cgnslib.newZone')
    def test_create_zone_failure(self, mock_new_zone):
        mock_base_node = MagicMock()
        mock_new_zone.side_effect = Exception("Error creating zone")
        result = create_zone(mock_base_node, "Zone1", np.array([[10, 0, 0]]), "Unstructured")
        self.assertIsNone(result)

    @patch('CGNS.PAT.cgnslib.newGridCoordinates')
    @patch('CGNS.PAT.cgnslib.newCoordinates')
    def test_write_coordinates_success(self, mock_new_coordinates, mock_new_grid_coordinates):
        mock_zone_node = MagicMock()
        mock_new_grid_coordinates.return_value = "grid_coordinates"
        coordX = np.array([1, 2, 3])
        coordY = np.array([4, 5, 6])
        coordZ = np.array([7, 8, 9])

        write_coordinates(mock_zone_node, coordX, coordY, coordZ)
        mock_new_grid_coordinates.assert_called_once_with(mock_zone_node, "GridCoordinates")
        calls = [
            unittest.mock.call(mock_zone_node, 'CoordinateX', coordX),
            unittest.mock.call(mock_zone_node, 'CoordinateY', coordY),
            unittest.mock.call(mock_zone_node, 'CoordinateZ', coordZ)
        ]
        mock_new_coordinates.assert_has_calls(calls, any_order=True)

    @patch('CGNS.PAT.cgnslib.newGridCoordinates')
    @patch('CGNS.PAT.cgnslib.newCoordinates')
    def test_write_coordinates_failure(self, mock_new_coordinates, mock_new_grid_coordinates):
        mock_zone_node = MagicMock()
        mock_new_coordinates.side_effect = Exception("Error writing coordinates")
        mock_new_grid_coordinates.side_effect = Exception("Error writing grid coordinates")
        
        with self.assertRaises(Exception) as context:
            write_coordinates(mock_zone_node, np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9]))
        
        self.assertTrue("Error writing grid coordinates" in str(context.exception))

if __name__ == '__main__':
    unittest.main()
