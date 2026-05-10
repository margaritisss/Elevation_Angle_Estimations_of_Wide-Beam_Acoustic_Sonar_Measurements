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
    
    Using the closed-form formula (Eq. 9.29 in Blanco):
        exp(ξ) = [exp(φ^)  |  V·ρ ]
                 [   0     |   1  ]
    
    where V is defined in Eq. 9.30 (left Jacobian of SO(3)).
    
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
        
        # Left Jacobian V (Eq. 9.30 in Blanco)
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
    # Guard against degenerate semi-axes
    a = max(a, 1e-6)
    b = max(b, 1e-6)

    # Handle point at origin
    if abs(x) < tol and abs(y) < tol:
        return -min(a, b)  # Inside, distance to nearest point
    
    # Use symmetry: work in first quadrant
    x_abs, y_abs = abs(x), abs(y)
    
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
#
# Implements the iterative on-manifold approach from Blanco §10.2:
#
#   At each outer iteration:
#     1. Linearize around ε = 0  (Eq. 10.4)
#     2. Solve for ε⋆ via one LM solve
#     3. Update on the manifold:  T ← T · exp(ε⋆)   (Eq. 10.5)
#     4. Reset ε to 0 and repeat until convergence
#
# The state lives on SE(3) × ℝ², so the boxplus operator (Eq. 10.6)
# applies the exponential map to the SE(3) part and ordinary addition
# to the semi-axes (a, b) ∈ ℝ².
#
# We optimize log(a), log(b) instead of a, b directly.  This ensures
# a, b > 0 without bounds (which scipy 'lm' method does not support)
# and removes the non-differentiable max() clamp.
# =============================================================================

def _cylinder_residuals_inner(eps, point_cloud, T_current, log_a, log_b):
    """
    Residuals for a SINGLE inner LM step.

    The increment ε ∈ ℝ⁸ = [ξ(6), δ_log_a, δ_log_b] is always small
    because the outer loop resets it to zero after each manifold update.

    Parameters
    ----------
    eps : (8,) array
        [ξ (6), δ_log_a, δ_log_b]
        - ξ: se(3) increment (translation ρ, rotation φ)
        - δ_log_a, δ_log_b: additive increments to log(a), log(b)
    point_cloud : (N, 3) array
    T_current : (4, 4) array   — current linearization point (fixed)
    log_a, log_b : float       — current log-semi-axes (fixed)

    Returns
    -------
    residuals : (N,) array
    """
    xi = eps[:6]
    a = np.exp(log_a + eps[6])
    b = np.exp(log_b + eps[7])

    # Right-multiply convention:  T_new = T_current · exp(ξ)
    T_new = T_current @ se3_exp(xi)

    N = point_cloud.shape[0]
    residuals = np.zeros(N)
    for i in range(N):
        residuals[i] = orthogonal_distance_to_cylinder(point_cloud[i], T_new, a, b)

    return residuals


def optimize_elliptic_cylinder(initial_state, max_outer=50, tol=1e-8,
                               verbose=True): 
    """
    Refine the elliptic cylinder estimate using Levenberg-Marquardt
    with proper iterative on-manifold SE(3) optimization (Blanco §10.2).

    Each outer iteration performs ONE Gauss-Newton / LM step in the
    tangent space (the inner solver is limited to a single linearisation),
    then retracts back onto the manifold.

    Trust-region bounds on ε keep steps inside the region where the
    tangent-space approximation is valid and prevent degenerate solutions
    (collapsing semi-axes).

    Parameters
    ----------
    initial_state : dict
        From build_initial_state():
        - 'T': (4, 4) initial SE(3) pose
        - 'a': initial semi-major axis
        - 'b': initial semi-minor axis
        - 'point_cloud': (N, 3) points
    max_outer : int
        Maximum number of outer (manifold) iterations
    tol : float
        Convergence threshold on ‖ε⋆‖
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
    log_a = np.log(initial_state['a'])
    log_b = np.log(initial_state['b'])
    point_cloud = initial_state['point_cloud']

    if verbose:
        print(f"Starting on-manifold LM optimization with {point_cloud.shape[0]} points")
        print(f"Initial semi-axes: a={initial_state['a']:.4f}, b={initial_state['b']:.4f}")

    # ------------------------------------------------------------------
    # Trust-region bounds for each component of ε ∈ ℝ⁸
    #
    #   ε = [dx, dy, dz, ω₁, ω₂, ω₃, δ_log_a, δ_log_b]
    #
    # Translation (dx,dy,dz): allow shifts up to ±0.5 m per step
    #   (adjust if your data uses different units)
    # Rotation (ω₁,ω₂,ω₃):  allow up to ±15° ≈ 0.26 rad per step
    # Log-scales (δ_log_a, δ_log_b):  allow ±0.5 per step
    #   (this means a or b can at most change by a factor e^0.5 ≈ 1.65
    #    or shrink by factor e^{-0.5} ≈ 0.61 in a single step)
    # ------------------------------------------------------------------
    trans_bound = 0.5
    rot_bound   = 0.26      # ~15 degrees
    scale_bound = 0.5        # factor ∈ [0.61, 1.65] per step

    lower_bounds = np.array([
        -trans_bound, -trans_bound, -trans_bound,
        -rot_bound, -rot_bound, -rot_bound,
        -scale_bound, -scale_bound,
    ])
    upper_bounds = -lower_bounds

    total_nfev = 0
    converged = False
    prev_rms = np.inf

    for outer_iter in range(max_outer):
        # -----------------------------------------------------------------
        # Inner LM solve:  find small ε⋆ around ε = 0
        #   (Blanco Eq. 10.4:  ε⋆ ← argmin_ε S(x ⊞ ε)  with ε linearized at 0)
        #
        # We use method='trf' (Trust Region Reflective) instead of 'lm'
        # so that we can enforce the bounds on ε.
        # max_nfev is kept low to approximate a single GN/LM step.
        # -----------------------------------------------------------------
        eps0 = np.zeros(8)

        result = least_squares(
            _cylinder_residuals_inner,
            eps0,
            args=(point_cloud, T_current, log_a, log_b),
            method='trf',
            bounds=(lower_bounds, upper_bounds),
            max_nfev=50,        # few evaluations ≈ one linearised step
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            verbose=0,
        )
        total_nfev += result.nfev

        eps_star = result.x
        xi_star = eps_star[:6]
        d_log_a = eps_star[6]
        d_log_b = eps_star[7]

        # -----------------------------------------------------------------
        # Manifold update  (Blanco Eq. 10.5:  x ← x ⊞ ε⋆)
        #
        #   SE(3) part:  T ← T · exp(ξ⋆)        (right-multiply convention)
        #   ℝ² part:     log_a ← log_a + δ_log_a  (ordinary addition)
        #                log_b ← log_b + δ_log_b
        # -----------------------------------------------------------------
        T_current = T_current @ se3_exp(xi_star)
        log_a += d_log_a
        log_b += d_log_b

        # Compute current cost for reporting
        a_cur = np.exp(log_a)
        b_cur = np.exp(log_b)
        cur_residuals = np.array([
            orthogonal_distance_to_cylinder(point_cloud[i], T_current, a_cur, b_cur)
            for i in range(point_cloud.shape[0])
        ])
        rms = np.sqrt(np.mean(cur_residuals**2))

        eps_norm = np.linalg.norm(eps_star)

        if verbose:
            print(f"  outer iter {outer_iter+1:3d}  |  ‖ε⋆‖ = {eps_norm:.2e}  |"
                  f"  a = {a_cur:.4f}  b = {b_cur:.4f}  |  RMS = {rms:.6f}")

        # -----------------------------------------------------------------
        # Convergence check
        # -----------------------------------------------------------------
        if eps_norm < tol:
            converged = True
            if verbose:
                print(f"  Converged: ‖ε⋆‖ = {eps_norm:.2e} < tol = {tol:.1e}")
            break

        prev_rms = rms

    # Final values
    a_final = np.exp(log_a)
    b_final = np.exp(log_b)

    # Extract geometric quantities
    center = T_current[:3, 3]
    major_axis = T_current[:3, 0]
    minor_axis = T_current[:3, 1]
    cylinder_axis = T_current[:3, 2]

    # Final RMS
    final_residuals = np.array([
        orthogonal_distance_to_cylinder(point_cloud[i], T_current, a_final, b_final)
        for i in range(point_cloud.shape[0])
    ])
    rms_error = np.sqrt(np.mean(final_residuals**2))

    if verbose:
        status = "CONVERGED" if converged else "MAX ITERATIONS REACHED"
        print(f"\nOptimization {status} after {outer_iter+1} outer iterations")
        print(f"Total function evaluations: {total_nfev}")
        print(f"Final semi-axes: a={a_final:.4f}, b={b_final:.4f}")
        print(f"RMS error: {rms_error:.6f}")

    return {
        'T': T_current,
        'a': a_final,
        'b': b_final,
        'center': center,
        'axis': cylinder_axis,
        'major_axis': major_axis,
        'minor_axis': minor_axis,
        'rms_error': rms_error,
        'success': converged,
        'n_iterations': outer_iter + 1,
        'total_nfev': total_nfev,
        'initial_a': initial_state['a'],
        'initial_b': initial_state['b'],
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