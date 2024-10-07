import numpy as np
from .cgns_utils import get_CG_elemtype
from .mesh_upgrade import preprocess_connectivity, upgrade_geoOrder, clean_data

class MeshData:
    def __init__(self, nodes, connectivity, elem_type, geo_order, sol_order):
        """
        Initialize the MeshData object with mesh nodes, connectivity, and other attributes.

        Parameters:
            nodes (np.array): Array of mesh nodes.
            connectivity (list): List representing mesh connectivity.
            elem_type (str): Type of elements in the mesh.
            geo_order (int): Geometric order of the mesh.
            sol_order (int): Solution order required for simulations.
        """
        self._nodes = nodes
        self._connectivity = connectivity
        self._elem_type = elem_type
        self._geo_order = geo_order
        self._sol_order = sol_order

    @property
    def nodes(self):
        """Returns the nodes of the mesh."""
        return self._nodes

    @property
    def connectivity(self):
        """Returns the connectivity of the mesh."""
        return self._connectivity

    @property
    def elem_type(self):
        """Returns the type of elements in the mesh."""
        return self._elem_type

    @property
    def geo_order(self):
        """Returns the geometric order of the mesh."""
        return self._geo_order

    @property
    def sol_order(self):
        """Returns the solution order of the mesh."""
        return self._sol_order

    @property
    def num_elements(self):
        """Returns the number of elements in the mesh."""
        return len(self._connectivity)

    @property
    def num_nodes(self):
        """Returns the number of nodes in the mesh."""
        return len(self._nodes)

    def upgrade_mesh(self):
        """
        Upgrade the mesh to a higher geometric order if necessary, based on the solution order.
        """
        self._connectivity = preprocess_connectivity(self._connectivity, get_CG_elemtype(self._elem_type, self._geo_order))
        
        if self._sol_order > self._geo_order:
            print("Upgrading mesh to higher geometric order.")
            upgraded_geo_order = min(self._sol_order, 4)  # Limiting the geometric order upgrade to 4
            old_geo_order = self._geo_order
            self._nodes, self._connectivity = upgrade_geoOrder(self._nodes, self._connectivity, self._geo_order, upgraded_geo_order, self._elem_type)
            self._geo_order = upgraded_geo_order

            if upgraded_geo_order == 3 and old_geo_order == 2:
                self._nodes, self._connectivity = clean_data(self._nodes, self._connectivity)
        print("Mesh upgrade done.")
        
    def __str__(self):
        """
        String representation of the MeshData object.
        """
        return (f"MeshData with {self.num_nodes} nodes, {self.num_elements} elements, "
                f"Element Type: {self.elem_type}, Geo Order: {self.geo_order}, Sol Order: {self.sol_order}")
