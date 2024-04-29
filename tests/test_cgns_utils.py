# tests/test_cgns_utils.py
import unittest
import numpy as np
from writer.cgns_utils import preprocess_coordinates, get_CG_elemtype, get_outputPnts_mapped_coords, get_cgns_cell_node_conn, CK

class TestCgnsUtils(unittest.TestCase):

    def test_preprocess_coordinates(self):
        nodes = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        coordX, coordY, coordZ = preprocess_coordinates(nodes)
        np.testing.assert_array_equal(coordX, [1, 4, 7])
        np.testing.assert_array_equal(coordY, [2, 5, 8])
        np.testing.assert_array_equal(coordZ, [3, 6, 9])

    def test_get_CG_elemtype_valid(self):
        element_type = 'Quad'
        geo_order = 1
        self.assertEqual(get_CG_elemtype(element_type, geo_order), CK.QUAD_4)

    def test_get_CG_elemtype_invalid(self):
        with self.assertRaises(ValueError):
            get_CG_elemtype('InvalidType', 99)

    def test_get_outputPnts_mapped_coords(self):
        element_type = CK.QUAD_4
        expected = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        np.testing.assert_array_almost_equal(get_outputPnts_mapped_coords(element_type), expected)

    def test_get_cgns_cell_node_conn_valid(self):
        self.assertEqual(get_cgns_cell_node_conn(CK.TRI_3), [0, 1, 2])

    def test_get_cgns_cell_node_conn_invalid(self):
        with self.assertRaises(ValueError):
            get_cgns_cell_node_conn('InvalidType')

if __name__ == '__main__':
    unittest.main()
