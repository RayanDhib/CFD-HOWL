import numpy as np

def compute_solShape_function(mapped_coord, element_type, sol_order):
    """
    Calculate solution shape functions based on element type and solution order.

    Parameters:
        mapped_coord (np.array): Mapped coordinates at which to evaluate the shape functions.
        element_type (str): Type of the finite element ('Triag', 'Quad', etc.).
        sol_order (int): Solution order, determining the polynomial degree of the shape functions.

    Returns:
        np.array: Shape functions evaluated at the given mapped coordinates.

    Raises:
        ValueError: If the element type is not supported.
    """
    if element_type == 'Triag':
        return compute_solShape_function_tri(mapped_coord, sol_order)
    elif element_type == 'Quad':
        return compute_solShape_function_quad(mapped_coord, sol_order)
    else:
        raise ValueError(f"Unsupported element type {element_type}")

def compute_geoShape_function(mapped_coord, element_type, geo_order):
    """
    Calculate geometric shape functions based on element type and geometric order.

    Parameters:
        mapped_coord (np.array): Mapped coordinates at which to evaluate the shape functions.
        element_type (str): Type of the finite element ('Triag', 'Quad', etc.).
        geo_order (int): Geometric order, determining the polynomial degree of the shape functions.

    Returns:
        np.array: Shape functions evaluated at the given mapped coordinates.

    Raises:
        ValueError: If the element type or geometric order is not supported.
    """
    if element_type == 'Triag':
        return compute_geoShape_function_tri(mapped_coord, geo_order)
    elif element_type == 'Quad':
        return compute_geoShape_function_quad(mapped_coord, geo_order)
    else:
        raise ValueError(f"Unsupported element type {element_type}")

# ----------------------------------------------------- Shape functions 
def compute_geoShape_function_tri(mapped_coord, geo_order):
    """
    Calculate triangular element geometric shape functions.

    Parameters:
        mapped_coord (np.array): Mapped coordinates for evaluation.
        geo_order (int): Geometric order of the polynomial.

    Returns:
        np.array: Evaluated shape functions.

    Raises:
        ValueError: If the geometric order is not supported.
    """
    if geo_order == 1:
        # Linear shape functions
        shape_func = np.array([1 - sum(mapped_coord), mapped_coord[0], mapped_coord[1]])
    elif geo_order == 2:
        # Quadratic shape functions
        shape_func = np.zeros(6)
        shape_func[0] = (1.0 - np.sum(mapped_coord)) * (2.0 * (1.0 - np.sum(mapped_coord)) - 1.0)
        shape_func[1] = mapped_coord[0] * (2.0 * mapped_coord[0] - 1.0)
        shape_func[2] = mapped_coord[1] * (2.0 * mapped_coord[1] - 1.0)
        shape_func[3] = 4.0 * mapped_coord[0] * (1.0 - np.sum(mapped_coord))
        shape_func[4] = 4.0 * mapped_coord[0] * mapped_coord[1]
        shape_func[5] = 4.0 * mapped_coord[1] * (1.0 - np.sum(mapped_coord))
    else:
        raise ValueError(f"Geo order {geo_order} for triangular elements not supported.")
    return shape_func

def compute_geoShape_function_quad(mapped_coord, geo_order):
    """
    Calculate quadrilateral element geometric shape functions.

    Parameters:
        mapped_coord (np.array): Mapped coordinates for evaluation, expected as [xi, eta].
        geo_order (int): Geometric order of the polynomial.

    Returns:
        np.array: Evaluated shape functions.

    Raises:
        ValueError: If the geometric order is not supported.
    """
    xi, eta = mapped_coord
    if geo_order == 1:
        # Linear shape functions for quadrilateral elements
        shape_func = np.array([
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta)
        ])
    elif geo_order == 2:
        # Quadratic shape functions for quadrilateral elements
        shape_func = np.zeros(9)
        xi2 = xi ** 2
        eta2 = eta ** 2
        xiEta = xi * eta

        shape_func[0] = 0.25 * (1.0 - xi)  * (1.0 - eta) * xiEta
        shape_func[1] = -0.25 * (1.0 + xi)  * (1.0 - eta) * xiEta
        shape_func[2] = 0.25 * (1.0 + xi)  * (1.0 + eta) * xiEta
        shape_func[3] = -0.25 * (1.0 - xi)  * (1.0 + eta) * xiEta
        shape_func[4] = -0.5  * (1.0 - xi2) * (1.0 - eta) * eta
        shape_func[5] = 0.5  * (1.0 + xi)  * (1.0 - eta2) * xi
        shape_func[6] = 0.5  * (1.0 - xi2) * (1.0 + eta) * eta
        shape_func[7] = -0.5  * (1.0 - xi)  * (1.0 - eta2) * xi
        shape_func[8] = (1.0 - xi2) * (1.0 - eta2)
    else:
        raise ValueError(f"Geo order {geo_order} for quadrilateral elements not supported.")
    return shape_func

def compute_solShape_function_tri(mapped_coord, sol_order):
    """
    Calculate triangular element solution shape functions based on the solution order.

    Parameters:
        mapped_coord (np.array): Local coordinates in the reference domain.
        sol_order (int): Order of the solution polynomial.

    Returns:
        np.array: Evaluated solution shape functions at the mapped coordinates.
    """
    exponents = get_sol_poly_exponents(sol_order, 'Triag')
    sol_pnts_local_coords = get_sol_pnts_local_coords(sol_order, 'Triag')
    coefs = get_solPolyCoefs(sol_pnts_local_coords, exponents)
    nbr_polys = len(coefs)
    shape_func = np.zeros(nbr_polys)

    for iPoly in range(nbr_polys):
        poly_value = 0
        for iTerm in range(nbr_polys):
            term = coefs[iPoly][iTerm]
            for iCoor, exp in enumerate(exponents[iTerm]):
                term *= np.power(mapped_coord[iCoor], exp)
            poly_value += term
        shape_func[iPoly] = poly_value
    return shape_func

def compute_solShape_function_quad(mapped_coord, sol_order):
    """
    Calculate quadrilateral element solution shape functions based on the solution order.

    Parameters:
        mapped_coord (np.array): Local coordinates in the reference domain [ksi, eta].
        sol_order (int): Order of the solution polynomial.

    Returns:
        np.array: Evaluated solution shape functions at the mapped coordinates.
    """
    exponents = get_sol_poly_exponents(sol_order, 'Quad') 
    sol_pnts_local_coords = get_sol_pnts_local_coords(sol_order,'Quad') 
    coefs = get_solPolyCoefs(sol_pnts_local_coords, exponents)      

    nbr_sol_pnts = int(len(coefs)**0.5)
    shape_func = np.zeros(len(coefs))
    ksi, eta = mapped_coord
    ksi_factors = np.zeros(nbr_sol_pnts)
    eta_factors = np.zeros(nbr_sol_pnts)
    for iSol in range(nbr_sol_pnts):
        ksiSol = sol_pnts_local_coords[iSol][0] 
        etaSol = sol_pnts_local_coords[iSol][0]
        ksi_factors[iSol] = 1
        eta_factors[iSol] = 1
        for iFac in range(nbr_sol_pnts):
            if iFac != iSol:
                ksiFac = sol_pnts_local_coords[iFac][0]
                etaFac = sol_pnts_local_coords[iFac][0]

                ksi_factors[iSol] *= (ksi - ksiFac) / (ksiSol - ksiFac)
                eta_factors[iSol] *= (eta - etaFac) / (etaSol - etaFac)

    iFunc = 0
    for iKsi in range(nbr_sol_pnts):
        for iEta in range(nbr_sol_pnts):
            shape_func[iFunc] = ksi_factors[iKsi] * eta_factors[iEta]
            iFunc += 1

    return shape_func
# Add more functions for other element types like Tetra, Hexa, etc.

# ----------------------------------------------------- Helper functions 
def get_solPolyCoefs(sol_pnts_local_coords, sol_poly_exponents):
    """
    Compute the coefficients for the solution polynomials based on the solution points' local coordinates and polynomial exponents.

    Parameters:
        sol_pnts_local_coords (np.array): Local coordinates of solution points in the reference domain.
        sol_poly_exponents (np.array): Exponents used in the polynomial basis functions.

    Returns:
        np.array: Coefficients of the solution polynomials.
    """
    nbr_sol_polys = len(sol_pnts_local_coords)
    dimensionality = sol_pnts_local_coords.shape[1]

    # Prepare the left-hand side (LHS) matrix for polynomial fitting
    lhs = np.ones((nbr_sol_polys, nbr_sol_polys))
    for i_poly in range(nbr_sol_polys):
        for i_term in range(nbr_sol_polys):
            for i_coor in range(dimensionality):
                lhs[i_poly, i_term] *= sol_pnts_local_coords[i_poly, i_coor] ** sol_poly_exponents[i_term, i_coor]

    # Invert the LHS matrix to solve for coefficients
    lhs_inv = np.linalg.inv(lhs)

    # Prepare the solution polynomial coefficients matrix
    sol_poly_coefs = lhs_inv.T  # Transpose to align indices with the expected output

    return sol_poly_coefs

def get_sol_poly_exponents(sol_order, elem_type):
    """
    Generate polynomial exponents for the specified element type and solution order.

    Parameters:
        sol_order (int): The polynomial order of the elements.
        elem_type (str): The type of element (e.g., 'Quad' or 'Triag').

    Returns:
        np.array: An array of exponents for each term in the polynomial basis.

    Raises:
        ValueError: If the element type is unsupported.
    """
    if elem_type == 'Quad':
        nbr_sol_pnts_1D = sol_order + 1
        exponents = [[i_ksi, i_eta] for i_ksi in range(nbr_sol_pnts_1D) for i_eta in range(nbr_sol_pnts_1D)]
    elif elem_type == 'Triag':
        exponents = [[i_ksi, i_eta] for i_ksi in range(sol_order + 1) for i_eta in range(sol_order + 1 - i_ksi)]
    else:
        raise ValueError("Unsupported element type")

    return np.array(exponents)

def get_sol_pnts_local_coords(sol_order, elem_type):
    """
    Generate local coordinates for solution points based on element type and polynomial order.
    
    Parameters:
        sol_order (int): The polynomial order.
        elem_type (str): The element type ('Triag' or 'Quad').

    Returns:
        np.array: An array of local coordinates for the solution points.

    Raises:
        ValueError: If the element type or solution order is unsupported.
    """
    if elem_type == 'Triag':
        if sol_order == 0:
            return np.array([[1/3, 1/3]])
        elif sol_order == 1:
            return np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
        elif sol_order == 2:
            return np.array([
                [0.091576213509780, 0.091576213509780],
                [0.816847572980440, 0.091576213509780],
                [0.091576213509780, 0.816847572980440],
                [0.445948490915964, 0.108103018168071],
                [0.108103018168071, 0.445948490915964],
                [0.445948490915964, 0.445948490915964]
            ])
        elif sol_order == 3:
            return np.array([
                [0.055564052669793, 0.055564052669793],
                [0.888871894660413, 0.055564052669793],
                [0.055564052669793, 0.888871894660413],
                [0.634210747745723, 0.070255540518384],
                [0.070255540518384, 0.634210747745723],
                [0.295533711735893, 0.634210747745723],
                [0.295533711735893, 0.070255540518384],
                [0.070255540518384, 0.295533711735893],
                [0.634210747745723, 0.295533711735893],
                [0.333333333333333, 0.333333333333333]
            ])
        elif sol_order == 4:
            return np.array([
                [0.035870877695734, 0.035870877695734],
                [0.928258244608533, 0.035870877695734],
                [0.035870877695734, 0.928258244608533],
                [0.241729395767967, 0.241729395767967],
                [0.516541208464066, 0.241729395767967],
                [0.241729395767967, 0.516541208464066],
                [0.474308787777079, 0.051382424445843],
                [0.051382424445843, 0.474308787777079],
                [0.474308787777079, 0.474308787777079],
                [0.751183631106484, 0.047312487011716],
                [0.047312487011716, 0.751183631106484],
                [0.201503881881800, 0.751183631106484],
                [0.201503881881800, 0.047312487011716],
                [0.047312487011716, 0.201503881881800],
                [0.751183631106484, 0.201503881881800]
            ])
    elif elem_type == 'Quad':
        # Quadrilateral elements local coordinates using Gauss-Legendre points
        points_1D = np.polynomial.legendre.leggauss(sol_order + 1)[0]
        return np.array(np.meshgrid(points_1D, points_1D)).T.reshape(-1, 2)[:, [1, 0]]
    else:
        raise ValueError(f"Unsupported element type '{elem_type}'")

    return np.array([])


