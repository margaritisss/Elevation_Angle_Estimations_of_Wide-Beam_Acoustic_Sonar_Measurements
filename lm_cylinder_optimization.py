"""
=============================================================================
Levenberg-Marquardt Optimization for Elliptic Cylinder Fitting
with On-Manifold SE(3) Optimization

Section 4.2.1 of Breux & Lapierre (2020):
    "To further improve our estimation, we use the Levenberg–Marquardt (LM)
     algorithm to obtain the shape which minimizes orthogonal distances to
     the elliptic cylinder. The estimation vector x ∈ ℝ⁸ representing our
     elliptic cylinder is composed of a 6D pose in SE(3) and the scales
     along the minor and major ellipse axis. Here, we use on-manifold
     optimization on SE(3), as described in [22]."

Reference [22]: Blanco, "A tutorial on SE(3) transformation parameterizations
                and on-manifold optimization" — Section 10.2

=============================================================================
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# =============================================================================
# SE(3) Lie Group Operations (from Blanco tutorial, Section 10)
# =============================================================================

def skew(v):
    """
    Skew-symmetric matrix from a 3-vector.
    [v]_× such that [v]_× @ w = v × w
    """
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ]
    ])


def se3_exp(xi):
    """
    Exponential map: se(3) → SE(3)
    
    Maps a 6-vector ξ = [ρ; φ] (translation increment; rotation increment)
    to a 4×4 SE(3) transformation matrix.
    
    This is the "boxplus" operation from Blanco Section 10.2:
        T_new = T_old @ exp(ξ)
    
    Using the closed-form formula (Eq. 9.26 in Blanco):
        exp(ξ) = [exp(φ^)  |  V·ρ ]
                 [   0     |   1  ]
    
    where V is the left Jacobian of SO(3).
    
    Parameters
    ----------
    xi : (6,) array
        Lie algebra element [ρ (3), φ (3)]
        - ρ: translation increment (will be multiplied by V)
        - φ: rotation increment (axis-angle, ||φ|| = angle)
    
    Returns
    -------
    T : (4, 4) array
        SE(3) transformation matrix
    """
    rho = xi[:3]   # translation part
    phi = xi[3:]   # rotation part (axis-angle)
    
    theta = np.linalg.norm(phi)
    
    if theta < 1e-10:
        # First-order approximation for small angles
        R = np.eye(3) + skew(phi)
        V = np.eye(3)
    else:
        # Rodrigues formula for rotation
        axis = phi / theta
        K = skew(axis)
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        
        # Left Jacobian V (Eq. 9.27 in Blanco)
        V = (np.eye(3) 
             + (1 - np.cos(theta)) / theta**2 * skew(phi)
             + (theta - np.sin(theta)) / theta**3 * (skew(phi) @ skew(phi)))
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    
    return T


def se3_log(T):
    """
    Logarithm map: SE(3) → se(3)
    
    Inverse of se3_exp.
    
    Parameters
    ----------
    T : (4, 4) array
        SE(3) transformation matrix
    
    Returns
    -------
    xi : (6,) array
        Lie algebra element [ρ; φ]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    
    # Rotation part: log(R) → axis-angle
    phi = Rotation.from_matrix(R).as_rotvec()
    theta = np.linalg.norm(phi)
    
    if theta < 1e-10:
        V_inv = np.eye(3)
    else:
        K = skew(phi / theta)
        # Inverse of left Jacobian
        V_inv = (np.eye(3) 
                 - 0.5 * skew(phi)
                 + (1 - theta * np.cos(theta/2) / (2 * np.sin(theta/2))) / theta**2 
                   * (skew(phi) @ skew(phi)))
    
    rho = V_inv @ t
    
    return np.concatenate([rho, phi])


# =============================================================================
# Orthogonal Distance to Elliptic Cylinder
# =============================================================================

def point_to_ellipse_distance(x, y, a, b, tol=1e-12, max_iter=20):
    """
    Compute the signed distance from point (x, y) to ellipse x²/a² + y²/b² = 1.
    
    Uses Newton's method to find the closest point on the ellipse.
    Positive distance = outside ellipse, negative = inside.
    
    This is non-trivial because the closest point requires solving a quartic.
    We use an iterative approach that's robust and accurate.
    
    Parameters
    ----------
    x, y : float
        Point coordinates in the ellipse plane
    a, b : float
        Semi-major and semi-minor axes (a >= b assumed)
    
    Returns
    -------
    dist : float
        Signed orthogonal distance to the ellipse
    """
    # Handle point at origin
    if abs(x) < tol and abs(y) < tol:
        return -min(a, b)  # Inside, distance to nearest point
    
    # Use symmetry: work in first quadrant
    x_abs, y_abs = abs(x), abs(y)
    
    # Initial guess: project radially onto ellipse
    r = np.sqrt((x_abs/a)**2 + (y_abs/b)**2)
    if r < tol:
        return -min(a, b)
    
    # Parametric angle estimate
    t = np.arctan2(a * y_abs, b * x_abs)
    
    # Newton iteration to find closest point
    for _ in range(max_iter):
        cos_t, sin_t = np.cos(t), np.sin(t)
        
        # Point on ellipse
        ex, ey = a * cos_t, b * sin_t
        
        # Vector from ellipse point to query point
        dx, dy = x_abs - ex, y_abs - ey
        
        # Tangent direction (derivative of ellipse parametrization)
        tx, ty = -a * sin_t, b * cos_t
        
        # Project residual onto tangent (should be zero at minimum)
        dot = dx * tx + dy * ty
        
        # Second derivative for Newton step
        ddot = (a**2 * sin_t**2 + b**2 * cos_t**2 
                + dx * (-a * cos_t) + dy * (-b * sin_t))
        
        if abs(ddot) < tol:
            break
        
        dt = -dot / ddot
        t = t + dt
        
        if abs(dt) < tol:
            break
    
    # Final closest point
    ex, ey = a * np.cos(t), b * np.sin(t)
    dist = np.sqrt((x_abs - ex)**2 + (y_abs - ey)**2)
    
    # Sign: positive if outside ellipse
    if (x_abs/a)**2 + (y_abs/b)**2 < 1:
        dist = -dist
    
    return dist


def orthogonal_distance_to_cylinder(p_world, T_cyl, a, b):
    """
    Compute the orthogonal distance from a 3D point to an elliptic cylinder.
    
    The cylinder is defined by:
        - T_cyl: 4×4 SE(3) matrix defining the cylinder's pose
          - Origin: center of the cylinder
          - Z-axis: cylinder axis direction
          - X-axis: major axis direction (semi-length a)
          - Y-axis: minor axis direction (semi-length b)
        - a, b: semi-major and semi-minor axes
    
    Parameters
    ----------
    p_world : (3,) array
        Point in world coordinates
    T_cyl : (4, 4) array
        SE(3) pose of the cylinder frame
    a, b : float
        Semi-major and semi-minor axes
    
    Returns
    -------
    d : float
        Orthogonal distance to cylinder surface (positive = outside)
    """
    # Transform point to cylinder's local frame
    R = T_cyl[:3, :3]
    t = T_cyl[:3, 3]
    
    # p_local = R^T @ (p_world - t)
    p_local = R.T @ (p_world - t)
    
    # In local frame: x, y are in the ellipse plane, z is along cylinder axis
    x_local, y_local, z_local = p_local
    
    # Distance is purely determined by (x, y) — z is ignored for an infinite cylinder
    dist = point_to_ellipse_distance(x_local, y_local, a, b)
    
    return dist


# =============================================================================
# Initial State Construction
# =============================================================================

def build_initial_state(pca_result, ellipse_result):
    """
    Build the initial SE(3) pose and scale parameters from your previous results.
    
    Parameters
    ----------
    pca_result : dict
        From find_cylinder_axes_on_scans(), contains:
        - 'center': (3,) robust RPCA centroid
        - 'axis_vector': (3,) cylinder axis
        - 'point_cloud': (N, 3) the 3D points
        
    ellipse_result : dict  
        From fit_elliptic_sections(), contains:
        - 'center_2d': (2,) ellipse center in PCA plane
        - 'semi_axes': (2,) [semi_major, semi_minor]
        - 'new_major': (3,) rotated major axis (aligned with ellipse)
        - 'new_minor': (3,) rotated minor axis (aligned with ellipse)
    
    Returns
    -------
    initial_state : dict
        - 'T': (4, 4) initial SE(3) pose
        - 'a': semi-major axis
        - 'b': semi-minor axis
        - 'point_cloud': (N, 3) points to fit
    """
    # 1. Get the cylinder frame axes
    #    Convention: Z = cylinder axis, X = major, Y = minor
    z_axis = pca_result['axis_vector'].copy()
    z_axis /= np.linalg.norm(z_axis)
    
    x_axis = ellipse_result['new_major'].copy()
    x_axis /= np.linalg.norm(x_axis)
    
    # Ensure orthogonality via Gram-Schmidt
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    # 2. Compute the 3D center position
    #    The 2D center offset is expressed in the (u, v) = (major, minor) basis
    #    from fit_elliptic_sections. But note that 'u' and 'v' in ellipse_results
    #    were the ORIGINAL major/minor before rotation. However, center_2d was
    #    computed in that coordinate system.
    #    
    #    We need to express the offset in world coordinates.
    u0, v0 = ellipse_result['center_2d']
    
    # The original (pre-rotation) major and minor axes from PCA
    # NOTE: ellipse_result['u'] and ['v'] are projected coordinates (N,), not unit vectors!
    original_major = pca_result['major_axis']  # (3,) unit vector
    original_minor = pca_result['minor_axis']  # (3,) unit vector
    
    # 3D offset from RPCA center to ellipse center
    center_offset = u0 * original_major + v0 * original_minor
    
    # Full 3D center
    center_3d = pca_result['center'] + center_offset
    
    # 3. Build the SE(3) transformation matrix
    #    Columns of R are the cylinder frame axes expressed in world frame
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center_3d
    
    # 4. Semi-axes
    a, b = ellipse_result['semi_axes']  # [semi_major, semi_minor]
    
    return {
        'T': T,
        'a': a,
        'b': b,
        'point_cloud': pca_result['point_cloud'],
    }


# =============================================================================
# Levenberg-Marquardt Optimization with On-Manifold SE(3) Updates
# =============================================================================

def cylinder_residuals(params, point_cloud, T_current):
    """
    Compute residuals (orthogonal distances) for all points.
    
    The optimization works by finding an increment ξ ∈ se(3) such that:
        T_new = T_current @ exp(ξ)
    
    minimizes the sum of squared distances.
    
    Parameters
    ----------
    params : (8,) array
        [ξ (6), a, b] where:
        - ξ = [ρ (3), φ (3)]: se(3) increment
        - a: semi-major axis
        - b: semi-minor axis
    point_cloud : (N, 3) array
        Points to fit
    T_current : (4, 4) array
        Current SE(3) estimate (fixed during this LM call)
    
    Returns
    -------
    residuals : (N,) array
        Orthogonal distance for each point
    """
    xi = params[:6]
    a = params[6]
    b = params[7]
    
    # Ensure positive semi-axes
    a = max(a, 1e-6)
    b = max(b, 1e-6)
    
    # Apply increment: T_new = T_current @ exp(ξ)
    T_new = T_current @ se3_exp(xi)
    
    # Compute distances
    N = point_cloud.shape[0]
    residuals = np.zeros(N)
    
    for i in range(N):
        residuals[i] = orthogonal_distance_to_cylinder(point_cloud[i], T_new, a, b)
    
    return residuals


def optimize_elliptic_cylinder(initial_state, max_nfev=1000, verbose=True):
    """
    Refine the elliptic cylinder estimate using Levenberg-Marquardt
    with on-manifold SE(3) optimization.
    
    This implements the approach from Section 10.2 of Blanco's tutorial:
    instead of optimizing the pose directly (which has issues with
    over-parameterization and singularities), we optimize the INCREMENT
    in the Lie algebra se(3), then apply it via the exponential map.
    
    Parameters
    ----------
    initial_state : dict
        From build_initial_state():
        - 'T': (4, 4) initial SE(3) pose
        - 'a': initial semi-major axis
        - 'b': initial semi-minor axis
        - 'point_cloud': (N, 3) points
    max_nfev : int
        Maximum number of function evaluations
    verbose : bool
        Print optimization progress
    
    Returns
    -------
    result : dict
        - 'T': (4, 4) optimized SE(3) pose
        - 'a': optimized semi-major axis
        - 'b': optimized semi-minor axis
        - 'center': (3,) cylinder center in world frame
        - 'axis': (3,) cylinder axis direction
        - 'major_axis': (3,) major axis direction
        - 'minor_axis': (3,) minor axis direction
        - 'rms_error': RMS of residuals
        - 'success': bool
    """
    T_current = initial_state['T'].copy()
    a0 = initial_state['a']
    b0 = initial_state['b']
    point_cloud = initial_state['point_cloud']
    
    if verbose:
        print(f"Starting LM optimization with {point_cloud.shape[0]} points")
        print(f"Initial semi-axes: a={a0:.4f}, b={b0:.4f}")
    
    # Initial parameters: ξ = 0 (no change to pose), current semi-axes
    x0 = np.array([0, 0, 0, 0, 0, 0, a0, b0])
    
    # Bounds: semi-axes must be positive
    lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 1e-6, 1e-6]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    
    # Run Levenberg-Marquardt
    result = least_squares(
        cylinder_residuals,
        x0,
        args=(point_cloud, T_current),
        method='lm',  # Levenberg-Marquardt
        max_nfev=max_nfev,
        verbose=2 if verbose else 0,
    )
    
    # Extract final parameters
    xi_final = result.x[:6]
    a_final = result.x[6]
    b_final = result.x[7]
    
    # Apply the final increment to get the optimized pose
    T_final = T_current @ se3_exp(xi_final)
    
    # Extract geometric quantities from T_final
    center = T_final[:3, 3]
    major_axis = T_final[:3, 0]  # X column = major axis
    minor_axis = T_final[:3, 1]  # Y column = minor axis
    cylinder_axis = T_final[:3, 2]  # Z column = cylinder axis
    
    # Compute RMS error
    final_residuals = cylinder_residuals(result.x, point_cloud, T_current)
    rms_error = np.sqrt(np.mean(final_residuals**2))
    
    if verbose:
        print(f"\nOptimization {'succeeded' if result.success else 'failed'}")
        print(f"Final semi-axes: a={a_final:.4f}, b={b_final:.4f}")
        print(f"RMS error: {rms_error:.6f}")
        print(f"Function evaluations: {result.nfev}")
    
    return {
        'T': T_final,
        'a': a_final,
        'b': b_final,
        'center': center,
        'axis': cylinder_axis,
        'major_axis': major_axis,
        'minor_axis': minor_axis,
        'rms_error': rms_error,
        'success': result.success,
        'n_iterations': result.nfev,
        'initial_a': a0,
        'initial_b': b0,
    }


# =============================================================================
# Wrapper for Processing Multiple Scans
# =============================================================================

def optimize_all_scans(pca_results, ellipse_results, verbose=True):
    """
    Run LM optimization on all scans.
    
    Parameters
    ----------
    pca_results : dict
        {scan_num: pca_dict} from find_cylinder_axes_on_scans
    ellipse_results : dict
        {scan_num: ellipse_dict} from fit_elliptic_sections
    verbose : bool
    
    Returns
    -------
    optimized_results : dict
        {scan_num: optimization_result}
    """
    optimized_results = {}
    
    for scan_num in ellipse_results.keys():
        if scan_num not in pca_results:
            if verbose:
                print(f"⚠ Scan {scan_num}: missing PCA results, skipping")
            continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing Scan {scan_num:04d}")
            print('='*60)
        
        # Build initial state
        initial_state = build_initial_state(
            pca_results[scan_num],
            ellipse_results[scan_num]
        )
        
        # Run optimization
        opt_result = optimize_elliptic_cylinder(initial_state, verbose=verbose)
        
        optimized_results[scan_num] = opt_result
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Optimized {len(optimized_results)} scan(s)")
        print('='*60)
    
    return optimized_results


# =============================================================================
# Comparison / Visualization Helper
# =============================================================================

def print_comparison(scan_num, ellipse_result, opt_result):
    """
    Print a comparison of before/after optimization.
    """
    print(f"\n--- Scan {scan_num:04d}: Before vs After LM Optimization ---")
    print(f"  Semi-major axis:  {ellipse_result['semi_axes'][0]:.4f}  →  {opt_result['a']:.4f}")
    print(f"  Semi-minor axis:  {ellipse_result['semi_axes'][1]:.4f}  →  {opt_result['b']:.4f}")
    print(f"  RMS error:        N/A  →  {opt_result['rms_error']:.6f}")
    print(f"  Cylinder center:  {opt_result['center']}")
    print(f"  Cylinder axis:    {opt_result['axis']}")
    print("-" * 55)
