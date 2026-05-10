# Gemini after the plan33 from oppus

def find_cylinder_axis(point_cloud, components):
    """
    Identify the cylinder axis among 3 principal vectors from RPCA.
    Section 4.2.1: tangent projections are minimal along the axis.
    """
    tangents = np.diff(point_cloud, axis=0)
    projection_sums = np.zeros(3)
    for k in range(3):
        projection_sums[k] = np.sum(np.abs(tangents @ components[k]))

    axis_index  = np.argmin(projection_sums)
    axis_vector = components[axis_index]
    return axis_index, axis_vector, projection_sums

def fit_ellipse_fitzgibbon(x, y):
    """
    Fit an ellipse to 2-D points using the direct least-squares
    method of Fitzgibbon, Pilu & Fisher (IEEE TPAMI, 1999).

    This implementation includes the recommended data normalization
    (translation + scaling) to prevent numerical instability, and 
    uses the theoretically correct eigenvector selection criterion 
    from Theorem 1.

    Parameters
    ----------
    x, y : (N,) arrays of 2-D point coordinates

    Returns
    -------
    coeffs : (6,) array  [a, b, c, d, e, f]  of the conic equation
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1. Normalize data for numerical stability (Section 3.1)
    # Translate centroid to origin
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x_centered = x - mean_x
    y_centered = y - mean_y

    # Scale so mean distance from origin is sqrt(2)
    distances = np.sqrt(x_centered**2 + y_centered**2)
    mean_dist = np.mean(distances)
    if mean_dist == 0:
        raise ValueError("All points are at the same location.")
    
    scale = np.sqrt(2) / mean_dist
    x_norm = x_centered * scale
    y_norm = y_centered * scale

    # Design matrix D = [x², xy, y², x, y, 1] using normalized coordinates
    D = np.column_stack([x_norm**2, x_norm*y_norm, y_norm**2, x_norm, y_norm, np.ones_like(x_norm)])

    # Scatter matrix S = Dᵀ D
    S = D.T @ D

    # Partition S into 3×3 blocks
    S1 = S[:3, :3]
    S2 = S[:3, 3:]
    S3 = S[3:, 3:]

    # Constraint matrix C₁ for 4ac − b² = 1
    C1 = np.array([[ 0,  0,  2],
                   [ 0, -1,  0],
                   [ 2,  0,  0]], dtype=float)
    C1_inv = np.linalg.inv(C1)

    # Reduced eigenproblem: (S1 − S2 S3⁻¹ S2ᵀ) a₁ = λ C₁ a₁
    S3_inv = np.linalg.inv(S3)
    M = C1_inv @ (S1 - S2 @ S3_inv @ S2.T)

    eigvals, eigvecs = np.linalg.eig(M)

    # 2. Select correct eigenvector (Theorem 1)
    # The generalized eigenproblem yields exactly one positive eigenvalue.
    valid = np.where(eigvals.real > 1e-8)[0]
    
    if len(valid) == 0:
        # Fallback if numerical issues lead to no strictly positive eigenvalues:
        cond = (4 * eigvecs[0] * eigvecs[2] - eigvecs[1]**2)
        valid = np.where(cond.real > 0)[0]
        if len(valid) == 0:
            raise RuntimeError("Ellipse fit failed — no valid eigenvector found")
        # Among valid, pick the one with largest real eigenvalue (closest to positive)
        best = valid[np.argmax(eigvals[valid].real)]
    else:
        # If there are multiple (due to noise), pick the largest one
        best = valid[np.argmax(eigvals[valid].real)]

    a1 = eigvecs[:, best].real          # [a, b, c]
    a2 = -S3_inv @ S2.T @ a1            # [d, e, f]
    
    a_n, b_n, c_n = a1
    d_n, e_n, f_n = a2

    # 3. Un-normalize the coefficients back to original coordinates
    a = a_n
    b = b_n
    c = c_n
    
    d = d_n / scale - 2 * a_n * mean_x - b_n * mean_y
    e = e_n / scale - 2 * c_n * mean_y - b_n * mean_x
    
    f = (f_n / (scale**2) 
         - (d_n * mean_x) / scale 
         - (e_n * mean_y) / scale 
         + a_n * (mean_x**2) 
         + b_n * mean_x * mean_y 
         + c_n * (mean_y**2))

    coeffs = np.array([a, b, c, d, e, f])
    
    return coeffs

def conic_to_ellipse_params(coeffs):
    """
    Convert conic coefficients  a x² + b xy + c y² + d x + e y + f = 0
    to geometric ellipse parameters.

    Returns
    -------
    cx, cy       : centre of the ellipse
    semi_major   : semi-major axis length
    semi_minor   : semi-minor axis length
    angle_rad    : rotation angle of the major axis (radians, from +x axis)
    """
    a, b, c, d, e, f = coeffs

    # Centre (solving ∂F/∂x = 0, ∂F/∂y = 0)
    denom = 4*a*c - b**2
    cx = (b*e - 2*c*d) / denom
    cy = (b*d - 2*a*e) / denom

    # Rotation angle of the ellipse axes
    angle = 0.5 * np.arctan2(b, a - c)

    # Semi-axes
    num   = (2 * (a*e**2 + c*d**2 + f*b**2 - b*d*e - 4*a*c*f))
    s1    = a + c
    s2    = np.sqrt((a - c)**2 + b**2)
    semi_major = np.sqrt(abs(num / (denom * (s1 - s2))))
    semi_minor = np.sqrt(abs(num / (denom * (s1 + s2))))

    # Ensure semi_major ≥ semi_minor
    if semi_major < semi_minor:
        semi_major, semi_minor = semi_minor, semi_major
        angle += np.pi / 2

    return cx, cy, semi_major, semi_minor, angle

# ─────────────────────────────────────────────────────────────
#  Step C:  Full pipeline — project, fit, rotate PCA base
# ─────────────────────────────────────────────────────────────

def fit_elliptic_cylinder(point_cloud, center, axis_vector,
                          major_pca, minor_pca):
    """
    Estimate the elliptic cross-section and rotate the PCA base
    to align with the ellipse axes (Section 4.2.1 of the paper).

    1. Project all 3-D points onto the 2-D plane spanned by
       (major_pca, minor_pca).
    2. Fit an ellipse to those projections [paper 33].
    3. Rotate the PCA base around the cylinder axis so that the
       new u/v axes coincide with the ellipse semi-axes.

    Parameters
    ----------
    point_cloud  : (N, 3)  3-D points
    center       : (3,)    robust centre from RPCA
    axis_vector  : (3,)    cylinder axis unit vector
    major_pca    : (3,)    1st non-axis principal vector (larger variance)
    minor_pca    : (3,)    2nd non-axis principal vector (smaller variance)

    Returns
    -------
    result : dict
        'semi_major'       : float — semi-major axis length
        'semi_minor'       : float — semi-minor axis length
        'ellipse_angle'    : float — rotation of ellipse in the PCA plane (rad)
        'center_2d'        : (2,)  — ellipse centre in the projection plane
        'center_3d'        : (3,)  — ellipse centre in world coordinates
        'conic_coeffs'     : (6,)  — conic equation coefficients
        'aligned_major'    : (3,)  — world-frame major axis of ellipse
        'aligned_minor'    : (3,)  — world-frame minor axis of ellipse
        'axis_vector'      : (3,)  — cylinder axis (unchanged)
    """
    # ── 1. Project 3-D points onto the 2-D PCA cross-section plane ──
    pts_centered = point_cloud - center
    u_coords = pts_centered @ major_pca    # projection onto major PCA vec
    v_coords = pts_centered @ minor_pca    # projection onto minor PCA vec

    # ── 2. Fit ellipse in (u, v) coordinates  [paper 33] ──
    coeffs = fit_ellipse_fitzgibbon(u_coords, v_coords)
    cx, cy, semi_a, semi_b, theta = conic_to_ellipse_params(coeffs)

    # ── 3. Rotate PCA base around cylinder axis by angle θ ──
    #    The fitted ellipse is rotated by θ in the (u, v) plane.
    #    We rotate the two PCA vectors by −θ around the cylinder axis
    #    so they align with the ellipse axes.
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    aligned_major = cos_t * major_pca + sin_t * minor_pca
    aligned_minor = -sin_t * major_pca + cos_t * minor_pca

    # Ensure unit vectors
    aligned_major /= np.linalg.norm(aligned_major)
    aligned_minor /= np.linalg.norm(aligned_minor)

    # Ellipse centre in world coordinates
    center_3d = center + cx * major_pca + cy * minor_pca

    return {
        'semi_major':    semi_a,
        'semi_minor':    semi_b,
        'ellipse_angle': theta,
        'center_2d':     np.array([cx, cy]),
        'center_3d':     center_3d,
        'conic_coeffs':  coeffs,
        'aligned_major': aligned_major,
        'aligned_minor': aligned_minor,
        'axis_vector':   axis_vector,
    }

def find_cylinder_axes_on_scans(scan_ids='all',
                                h5_path='HDF5_DATA/7_SCANS.h5',
                                K=3, trim_pct=0.50, seed=42,
                                verbose=True):
    """
    Run RPCA then identify the cylinder axis for each requested scan.

    Parameters
    ----------
    scan_ids  : int, list[int], or 'all'
    h5_path   : str
    K, trim_pct, seed : forwarded to rpca_on_scans
    verbose   : bool

    Returns
    -------
    results : dict { scan_number: { axis_vector, major_axis, minor_axis, ... } }
    """
    rpca_results = rpca_on_scans(
        scan_ids=scan_ids, h5_path=h5_path,
        K=K, trim_pct=trim_pct, seed=seed, verbose=verbose,
    )

    results = {}
    with h5py.File(h5_path, 'r') as f:
        for scan_num, rpca_data in rpca_results.items():
            key = f'{scan_num:04d}'
            pc  = f[f'SCANS/{key}/POINT_CLOUD'][:]

            axis_idx, axis_vec, proj_sums = find_cylinder_axis(
                pc, rpca_data['components']
            )

            other_idx = sorted([i for i in range(K) if i != axis_idx])
            # The two remaining vectors span the ellipse cross-section.
            # Order them by decreasing variance → major then minor.
            vr = rpca_data['variances']
            if vr[other_idx[0]] < vr[other_idx[1]]:
                other_idx = [other_idx[1], other_idx[0]]

            major_axis = rpca_data['components'][other_idx[0]]
            minor_axis = rpca_data['components'][other_idx[1]]

            results[scan_num] = {
                'axis_index':      axis_idx,
                'axis_vector':     axis_vec,
                'major_axis':      major_axis,
                'minor_axis':      minor_axis,
                'components':      rpca_data['components'],
                'center':          rpca_data['center'],
                'variances':       rpca_data['variances'],
                'projection_sums': proj_sums,
                'n_points':        rpca_data['n_points'],
            }

            if verbose:
                print(f"\n--- Scan {key} ({rpca_data['n_points']} pts) ---")
                print(f"  Projection sums : {proj_sums}")
                print(f"  Cylinder axis   : {axis_vec}  (component #{axis_idx})")
                print(f"  Major ellipse   : {major_axis}  (component #{other_idx[0]})")
                print(f"  Minor ellipse   : {minor_axis}  (component #{other_idx[1]})")

    if verbose:
        print(f"\n✓ Identified cylinder axis for {len(results)} scan(s)")

    return results



results = find_cylinder_axes_on_scans(scan_ids='all')


















# OPPUs
# ─────────────────────────────────────────────────────────────
#  Step A:  Cylinder axis identification (from before)
# ─────────────────────────────────────────────────────────────

def find_cylinder_axis(point_cloud, components):
    """Identify the cylinder axis (minimal tangent projection)."""
    tangents = np.diff(point_cloud, axis=0)
    proj_sums = np.array([np.sum(np.abs(tangents @ components[k]))
                          for k in range(3)])
    idx = np.argmin(proj_sums)
    return idx, components[idx], proj_sums

# ─────────────────────────────────────────────────────────────
#  Step B:  Fitzgibbon direct ellipse fit  [paper 33]
# ─────────────────────────────────────────────────────────────

def fit_ellipse_fitzgibbon(x, y):
    """
    Fit an ellipse to 2-D points using the direct least-squares
    method of Fitzgibbon, Pilu & Fisher (IEEE TPAMI, 1999).

    This implementation includes the recommended data normalization
    (translation + scaling) to prevent numerical instability, and 
    uses the theoretically correct eigenvector selection criterion 
    from Theorem 1.

    Parameters
    ----------
    x, y : (N,) arrays of 2-D point coordinates

    Returns
    -------
    coeffs : (6,) array  [a, b, c, d, e, f]  of the conic equation
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1. Normalize data for numerical stability (Section 3.1)
    # Translate centroid to origin
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x_centered = x - mean_x
    y_centered = y - mean_y

    # Scale so mean distance from origin is sqrt(2)
    distances = np.sqrt(x_centered**2 + y_centered**2)
    mean_dist = np.mean(distances)
    if mean_dist == 0:
        raise ValueError("All points are at the same location.")
    
    scale = np.sqrt(2) / mean_dist
    x_norm = x_centered * scale
    y_norm = y_centered * scale

    # Design matrix D = [x², xy, y², x, y, 1] using normalized coordinates
    D = np.column_stack([x_norm**2, x_norm*y_norm, y_norm**2, x_norm, y_norm, np.ones_like(x_norm)])

    # Scatter matrix S = Dᵀ D
    S = D.T @ D

    # Partition S into 3×3 blocks
    S1 = S[:3, :3]
    S2 = S[:3, 3:]
    S3 = S[3:, 3:]

    # Constraint matrix C₁ for 4ac − b² = 1
    C1 = np.array([[ 0,  0,  2],
                   [ 0, -1,  0],
                   [ 2,  0,  0]], dtype=float)
    C1_inv = np.linalg.inv(C1)

    # Reduced eigenproblem: (S1 − S2 S3⁻¹ S2ᵀ) a₁ = λ C₁ a₁
    S3_inv = np.linalg.inv(S3)
    M = C1_inv @ (S1 - S2 @ S3_inv @ S2.T)

    eigvals, eigvecs = np.linalg.eig(M)

    # 2. Select correct eigenvector (Theorem 1)
    # The generalized eigenproblem yields exactly one positive eigenvalue.
    valid = np.where(eigvals.real > 1e-8)[0]
    
    if len(valid) == 0:
        # Fallback if numerical issues lead to no strictly positive eigenvalues:
        cond = (4 * eigvecs[0] * eigvecs[2] - eigvecs[1]**2)
        valid = np.where(cond.real > 0)[0]
        if len(valid) == 0:
            raise RuntimeError("Ellipse fit failed — no valid eigenvector found")
        # Among valid, pick the one with largest real eigenvalue (closest to positive)
        best = valid[np.argmax(eigvals[valid].real)]
    else:
        # If there are multiple (due to noise), pick the largest one
        best = valid[np.argmax(eigvals[valid].real)]

    a1 = eigvecs[:, best].real          # [a, b, c]
    a2 = -S3_inv @ S2.T @ a1            # [d, e, f]
    
    a_n, b_n, c_n = a1
    d_n, e_n, f_n = a2

    # 3. Un-normalize the coefficients back to original coordinates
    a = a_n
    b = b_n
    c = c_n
    
    d = d_n / scale - 2 * a_n * mean_x - b_n * mean_y
    e = e_n / scale - 2 * c_n * mean_y - b_n * mean_x
    
    f = (f_n / (scale**2) 
         - (d_n * mean_x) / scale 
         - (e_n * mean_y) / scale 
         + a_n * (mean_x**2) 
         + b_n * mean_x * mean_y 
         + c_n * (mean_y**2))

    coeffs = np.array([a, b, c, d, e, f])
    
    return coeffs

def conic_to_ellipse_params(coeffs):
    """
    Convert conic coefficients  a x² + b xy + c y² + d x + e y + f = 0
    to geometric ellipse parameters.

    Returns
    -------
    cx, cy       : centre of the ellipse
    semi_major   : semi-major axis length
    semi_minor   : semi-minor axis length
    angle_rad    : rotation angle of the major axis (radians, from +x axis)
    """
    a, b, c, d, e, f = coeffs

    # Centre (solving ∂F/∂x = 0, ∂F/∂y = 0)
    denom = 4*a*c - b**2
    cx = (b*e - 2*c*d) / denom
    cy = (b*d - 2*a*e) / denom

    # Rotation angle of the ellipse axes
    angle = 0.5 * np.arctan2(b, a - c)

    # Semi-axes
    num   = (2 * (a*e**2 + c*d**2 + f*b**2 - b*d*e - 4*a*c*f))
    s1    = a + c
    s2    = np.sqrt((a - c)**2 + b**2)
    semi_major = np.sqrt(abs(num / (denom * (s1 - s2))))
    semi_minor = np.sqrt(abs(num / (denom * (s1 + s2))))

    # Ensure semi_major ≥ semi_minor
    if semi_major < semi_minor:
        semi_major, semi_minor = semi_minor, semi_major
        angle += np.pi / 2

    return cx, cy, semi_major, semi_minor, angle

# ─────────────────────────────────────────────────────────────
#  Step C:  Full pipeline — project, fit, rotate PCA base
# ─────────────────────────────────────────────────────────────

def fit_elliptic_cylinder(point_cloud, center, axis_vector,
                          major_pca, minor_pca):
    """
    Estimate the elliptic cross-section and rotate the PCA base
    to align with the ellipse axes (Section 4.2.1 of the paper).

    1. Project all 3-D points onto the 2-D plane spanned by
       (major_pca, minor_pca).
    2. Fit an ellipse to those projections [paper 33].
    3. Rotate the PCA base around the cylinder axis so that the
       new u/v axes coincide with the ellipse semi-axes.

    Parameters
    ----------
    point_cloud  : (N, 3)  3-D points
    center       : (3,)    robust centre from RPCA
    axis_vector  : (3,)    cylinder axis unit vector
    major_pca    : (3,)    1st non-axis principal vector (larger variance)
    minor_pca    : (3,)    2nd non-axis principal vector (smaller variance)

    Returns
    -------
    result : dict
        'semi_major'       : float — semi-major axis length
        'semi_minor'       : float — semi-minor axis length
        'ellipse_angle'    : float — rotation of ellipse in the PCA plane (rad)
        'center_2d'        : (2,)  — ellipse centre in the projection plane
        'center_3d'        : (3,)  — ellipse centre in world coordinates
        'conic_coeffs'     : (6,)  — conic equation coefficients
        'aligned_major'    : (3,)  — world-frame major axis of ellipse
        'aligned_minor'    : (3,)  — world-frame minor axis of ellipse
        'axis_vector'      : (3,)  — cylinder axis (unchanged)
    """
    # ── 1. Project 3-D points onto the 2-D PCA cross-section plane ──
    pts_centered = point_cloud - center
    u_coords = pts_centered @ major_pca    # projection onto major PCA vec
    v_coords = pts_centered @ minor_pca    # projection onto minor PCA vec

    # ── 2. Fit ellipse in (u, v) coordinates  [paper 33] ──
    coeffs = fit_ellipse_fitzgibbon(u_coords, v_coords)
    cx, cy, semi_a, semi_b, theta = conic_to_ellipse_params(coeffs)

    # ── 3. Rotate PCA base around cylinder axis by angle θ ──
    #    The fitted ellipse is rotated by θ in the (u, v) plane.
    #    We rotate the two PCA vectors by −θ around the cylinder axis
    #    so they align with the ellipse axes.
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    aligned_major = cos_t * major_pca + sin_t * minor_pca
    aligned_minor = -sin_t * major_pca + cos_t * minor_pca

    # Ensure unit vectors
    aligned_major /= np.linalg.norm(aligned_major)
    aligned_minor /= np.linalg.norm(aligned_minor)

    # Ellipse centre in world coordinates
    center_3d = center + cx * major_pca + cy * minor_pca

    return {
        'semi_major':    semi_a,
        'semi_minor':    semi_b,
        'ellipse_angle': theta,
        'center_2d':     np.array([cx, cy]),
        'center_3d':     center_3d,
        'conic_coeffs':  coeffs,
        'aligned_major': aligned_major,
        'aligned_minor': aligned_minor,
        'axis_vector':   axis_vector,
    }

# ─────────────────────────────────────────────────────────────
#  Step D:  Convenience wrapper for multiple scans
# ─────────────────────────────────────────────────────────────

def fit_elliptic_cylinder_on_scans(scan_ids='all',
                                   h5_path='HDF5_DATA/7_SCANS.h5',
                                   K=3, trim_pct=0.50, seed=42,
                                   verbose=True):
    """
    Full Algorithm 1, Line 2:  fitEllipticCylinder.

    For each scan:
      1. Run RPCA → 3 principal vectors  (already done)
      2. Identify cylinder axis           (tangent-projection criterion)
      3. Fit elliptic section             (Fitzgibbon [33])
      4. Rotate PCA base to align with ellipse axes

    Parameters
    ----------
    scan_ids  : int, list[int], or 'all'
    h5_path   : str
    K, trim_pct, seed : forwarded to rpca_on_scans
    verbose   : bool

    Returns
    -------
    results : dict { scan_number: fit_result_dict }
    """
    # ── A. Run RPCA ──
    rpca_results = rpca_on_scans(
        scan_ids=scan_ids, h5_path=h5_path,
        K=K, trim_pct=trim_pct, seed=seed, verbose=verbose,
    )

    results = {}
    with h5py.File(h5_path, 'r') as f:
        for scan_num, rpca_data in rpca_results.items():
            key = f'{scan_num:04d}'
            pc  = f[f'SCANS/{key}/POINT_CLOUD'][:]

            # ── B. Identify cylinder axis ──
            axis_idx, axis_vec, proj_sums = find_cylinder_axis(
                pc, rpca_data['components']
            )
            other_idx = sorted([i for i in range(K) if i != axis_idx])
            vr = rpca_data['variances']
            if vr[other_idx[0]] < vr[other_idx[1]]:
                other_idx = [other_idx[1], other_idx[0]]

            major_pca = rpca_data['components'][other_idx[0]]
            minor_pca = rpca_data['components'][other_idx[1]]

            # ── C. Fit ellipse and rotate base ──
            try:
                fit = fit_elliptic_cylinder(
                    point_cloud=pc,
                    center=rpca_data['center'],
                    axis_vector=axis_vec,
                    major_pca=major_pca,
                    minor_pca=minor_pca,
                )

                results[scan_num] = {
                    **fit,
                    'components':      rpca_data['components'],
                    'variances':       rpca_data['variances'],
                    'rpca_center':     rpca_data['center'],
                    'projection_sums': proj_sums,
                    'n_points':        rpca_data['n_points'],
                }

                if verbose:
                    print(f"\n--- Scan {key} ({rpca_data['n_points']} pts) ---")
                    print(f"  Cylinder axis       : {axis_vec}")
                    print(f"  PCA major (before)  : {major_pca}")
                    print(f"  PCA minor (before)  : {minor_pca}")
                    print(f"  Ellipse semi-major  : {fit['semi_major']:.4f}")
                    print(f"  Ellipse semi-minor  : {fit['semi_minor']:.4f}")
                    print(f"  Ellipse rotation    : {np.degrees(fit['ellipse_angle']):.2f}°")
                    print(f"  Aligned major axis  : {fit['aligned_major']}")
                    print(f"  Aligned minor axis  : {fit['aligned_minor']}")
                    print(f"  Ellipse centre (3D) : {fit['center_3d']}")

            except RuntimeError as e:
                if verbose:
                    print(f"\n⚠ Scan {key}: {e}")

    if verbose:
        print(f"\n✓ Fitted elliptic cylinder for {len(results)} scan(s)")

    return results





    # # Single scan
# results = fit_elliptic_cylinder_on_scans(scan_ids=3)

# # Several scans
# results = fit_elliptic_cylinder_on_scans(scan_ids=[1, 2, 3])

# All scans
results = fit_elliptic_cylinder_on_scans(scan_ids='all')

# Inspect one result
r = results[3]
print(f"Semi-major: {r['semi_major']:.3f}")
print(f"Semi-minor: {r['semi_minor']:.3f}")
print(f"Aligned major axis (world): {r['aligned_major']}")
print(f"Aligned minor axis (world): {r['aligned_minor']}")
print(f"Cylinder axis (world):      {r['axis_vector']}")