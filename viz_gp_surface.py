"""
Visualization of GP-estimated karst surface
============================================

Given a trained GP for a scan, this module:
  1. Samples a dense grid of (s, ψ) values
  2. Predicts the range ρ̄(s,ψ) ± σ(s,ψ) from the GP
  3. Converts each predicted point to 3D world coordinates
     (using the same quaternion chain as compute_point_cloud)
  4. Builds PyVista meshes for the mean surface and ±3σ bounds
  5. Overlays the original vertical sonar point cloud

Usage in your notebook:
    from gp_surface import train_gp_single_scan
    from viz_gp_surface import visualize_gp_surface

    gp_result, scan_data = train_gp_single_scan(57)
    visualize_gp_surface(57)          # loads from HDF5 + retrains
    # or:
    visualize_gp_surface(57, gp_result=gp_result)  # reuse existing
"""

import numpy as np
import h5py
import pyvista as pv
from math import cos, sin, pi


# ══════════════════════════════════════════════════════════════════════════════
# Quaternion helpers  (same convention as your notebook: [w, x, y, z])
# ══════════════════════════════════════════════════════════════════════════════

def _qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _sonar_to_world(rho, psi, robot_pos, robot_quat):
    """
    Convert a single vertical sonar measurement (rho, psi) to a 3D world
    point, using exactly the same quaternion chain as your
    compute_point_cloud function.

    Parameters
    ----------
    rho        : float, range (distance)
    psi        : float, sonar rotation angle (radians)
    robot_pos  : (3,) robot position [x, y, z]
    robot_quat : (4,) robot orientation [qw, qx, qy, qz]

    Returns
    -------
    point : (3,) world-frame 3D point
    """
    # Body-frame quaternion from sonar angle
    half_psi = psi / 2.0
    q_body = np.array([cos(half_psi), -sin(half_psi), 0.0, 0.0])

    # Frame alignment rotations
    rot_z = np.array([cos(pi/2), 0.0, 0.0, sin(pi/2)])
    q_body = _qmul(q_body, rot_z)

    rot_y = np.array([cos(pi/2), 0.0, sin(pi/2), 0.0])
    q_body = _qmul(q_body, rot_y)

    # Combined rotation: robot × body
    q_combined = _qmul(robot_quat, q_body)

    # Rotate the distance vector [0, 0, 0, rho]
    dist_quat = np.array([0.0, 0.0, 0.0, rho])
    rotated = _qmul(q_combined, _qmul(dist_quat, _qconj(q_combined)))

    # World-frame point
    return robot_pos + rotated[1:4]


def _sonar_to_world_batch(rho_array, psi_array, robot_pos, robot_quat):
    """
    Vectorized version: convert multiple (rho, psi) pairs at a single
    robot pose to world-frame 3D points.
    """
    n = len(rho_array)
    points = np.empty((n, 3))
    for j in range(n):
        points[j] = _sonar_to_world(rho_array[j], psi_array[j],
                                     robot_pos, robot_quat)
    return points


# ══════════════════════════════════════════════════════════════════════════════
# Generate the predicted 3D surface from the GP
# ══════════════════════════════════════════════════════════════════════════════

def generate_gp_surface(gp_result, scan_data,
                        n_s=40, n_psi=80,
                        sigma_mult=3.0):
    """
    Sample the GP on a dense (s, ψ) grid and convert predictions to 3D.

    Parameters
    ----------
    gp_result  : dict returned by train_gp_for_scan()
    scan_data  : dict with keys 'H', 's', 'pose_idx_min', 'T', 'a', 'b',
                 'M_v_rho', 'M_v_psi', 'M_v_pose_idx'
    n_s        : int, number of samples along the trajectory
    n_psi      : int, number of angular samples (full 2π)
    sigma_mult : float, confidence interval multiplier (default 3σ)

    Returns
    -------
    dict with keys:
        'mean_points'  : (n_s*n_psi, 3) — 3D points of the mean surface
        'upper_points' : (n_s*n_psi, 3) — 3D points of +σ surface
        'lower_points' : (n_s*n_psi, 3) — 3D points of −σ surface
        'sigma_values' : (n_s*n_psi,)   — σ at each grid point
        'rho_mean'     : (n_s*n_psi,)   — predicted mean range
        'rho_std'      : (n_s*n_psi,)   — predicted std of range
        'grid_s'       : (n_s,)         — sampled s values
        'grid_psi'     : (n_psi,)       — sampled ψ values
        'n_s'          : int
        'n_psi'        : int
    """
    from gp_surface import predict_surface

    H_scan       = scan_data['H']
    s_scan       = scan_data['s']
    pose_idx_min = scan_data['pose_idx_min']
    T            = scan_data['T']
    a            = scan_data['a']
    b            = scan_data['b']

    # ── Build the query grid ─────────────────────────────────────────
    grid_s   = np.linspace(s_scan.min(), s_scan.max(), n_s)
    grid_psi = np.linspace(-np.pi, np.pi, n_psi, endpoint=False)

    ss, pp = np.meshgrid(grid_s, grid_psi, indexing='ij')  # (n_s, n_psi)
    s_flat   = ss.ravel()
    psi_flat = pp.ravel()

    # ── GP prediction ────────────────────────────────────────────────
    rho_mean, rho_var = predict_surface(
        gp_result,
        s_query=s_flat,
        psi_query=psi_flat,
        T=T, a=a, b=b,
        H_scan=H_scan, s_scan=s_scan,
        pose_idx_min=pose_idx_min,
    )

    rho_std = np.sqrt(np.maximum(rho_var, 0.0))

    # Clamp ranges to be positive
    rho_upper = np.maximum(rho_mean + sigma_mult * rho_std, 0.01)
    rho_lower = np.maximum(rho_mean - sigma_mult * rho_std, 0.01)
    rho_mean  = np.maximum(rho_mean, 0.01)

    # ── Convert to 3D world coordinates ──────────────────────────────
    n_total = len(s_flat)
    mean_pts  = np.empty((n_total, 3))
    upper_pts = np.empty((n_total, 3))
    lower_pts = np.empty((n_total, 3))

    for i in range(n_total):
        # Find the closest trajectory pose for this s value
        idx = np.argmin(np.abs(s_scan - s_flat[i]))
        robot_pos  = H_scan[idx, :3]
        robot_quat = H_scan[idx, 3:]  # [qw, qx, qy, qz]

        psi_i = psi_flat[i]

        mean_pts[i]  = _sonar_to_world(rho_mean[i],  psi_i, robot_pos, robot_quat)
        upper_pts[i] = _sonar_to_world(rho_upper[i], psi_i, robot_pos, robot_quat)
        lower_pts[i] = _sonar_to_world(rho_lower[i], psi_i, robot_pos, robot_quat)

    return {
        'mean_points':  mean_pts,
        'upper_points': upper_pts,
        'lower_points': lower_pts,
        'sigma_values': rho_std,
        'rho_mean':     rho_mean,
        'rho_std':      rho_std,
        'grid_s':       grid_s,
        'grid_psi':     grid_psi,
        'n_s':          n_s,
        'n_psi':        n_psi,
    }


def _build_structured_mesh(points, n_s, n_psi):
    """
    Build a PyVista StructuredGrid from a (n_s*n_psi, 3) array of points
    arranged on a regular (s, ψ) grid.

    The grid wraps around in ψ (closing the tube), so we append the
    first column to stitch the seam.
    """
    # Reshape to (n_s, n_psi, 3)
    pts = points.reshape(n_s, n_psi, 3)

    # Close the tube by appending the first angular column
    pts_closed = np.concatenate([pts, pts[:, 0:1, :]], axis=1)  # (n_s, n_psi+1, 3)

    grid = pv.StructuredGrid(
        pts_closed[:, :, 0],
        pts_closed[:, :, 1],
        pts_closed[:, :, 2],
    )
    return grid


# ══════════════════════════════════════════════════════════════════════════════
# Main visualization function
# ══════════════════════════════════════════════════════════════════════════════

def visualize_gp_surface(scan_num,
                         h5_path='HDF5_DATA/7_SCANS.h5',
                         prior_key='INITIAL',
                         gp_result=None,
                         n_s=40,
                         n_psi=80,
                         sigma_mult=3.0,
                         show_bounds=True,
                         show_cylinder=True,
                         show_training_pts=True,
                         optimize_gp=True):
    """
    Visualize the GP-estimated surface for a given scan.

    Displays:
      - Green mesh:     GP mean surface
      - Blue mesh:      +3σ upper confidence bound
      - Dark blue mesh: −3σ lower confidence bound
      - Light-blue pts: Original vertical sonar point cloud
      - Orange mesh:    Elliptic cylinder prior (optional)

    Parameters
    ----------
    scan_num       : int, scan number to visualize
    h5_path        : str, path to HDF5 file
    prior_key      : str, 'INITIAL' or 'OPTIMIZED'
    gp_result      : dict, pre-trained GP result (if None, trains on the fly)
    n_s            : int, trajectory samples for the surface grid
    n_psi          : int, angular samples for the surface grid
    sigma_mult     : float, confidence bound multiplier (default 3)
    show_bounds    : bool, show ±σ surfaces
    show_cylinder  : bool, show the elliptic cylinder prior
    show_training_pts : bool, show the original sonar point cloud
    optimize_gp    : bool, optimize GP hyperparameters (if training)

    Returns
    -------
    gp_result   : the trained GP result dict (for reuse)
    surface_data: the generated surface dict
    """
    from gp_surface import train_gp_single_scan

    # ── 1. Load scan data ────────────────────────────────────────────
    sk = f"{scan_num:04d}"
    with h5py.File(h5_path, 'r') as f:
        scan = f[f'SCANS/{sk}']

        point_cloud  = scan['POINT_CLOUD'][:]
        M_v_rho      = scan['M_v_rho'][:]
        M_v_psi      = scan['M_v_psi'][:]
        M_v_pose_idx = scan['M_v_pose_idx'][:].astype(int)
        H_scan       = scan['H'][:]
        s_scan       = scan['s'][:]
        pose_idx_min = int(scan['pose_idx_min'][()])

        prior_path = f'SURFACE_PRIOR/{prior_key}'
        T = scan[f'{prior_path}/T'][:]
        a = float(scan[f'{prior_path}/a'][()])
        b = float(scan[f'{prior_path}/b'][()])

    scan_data = {
        'M_v_rho': M_v_rho, 'M_v_psi': M_v_psi,
        'M_v_pose_idx': M_v_pose_idx,
        'H': H_scan, 's': s_scan,
        'pose_idx_min': pose_idx_min,
        'T': T, 'a': a, 'b': b,
    }

    # ── 2. Train GP if not provided ──────────────────────────────────
    if gp_result is None:
        print(f"Training GP for scan {scan_num}...")
        gp_result, _ = train_gp_single_scan(
            scan_num, h5_path=h5_path,
            prior_key=prior_key, optimize=optimize_gp
        )
        if gp_result is None:
            print(f"Error: could not train GP for scan {scan_num}")
            return None, None

    # ── 3. Generate the predicted surface ────────────────────────────
    print(f"Generating surface grid ({n_s}×{n_psi} = {n_s*n_psi} points)...")
    surface_data = generate_gp_surface(
        gp_result, scan_data,
        n_s=n_s, n_psi=n_psi,
        sigma_mult=sigma_mult
    )

    # ── 4. Build PyVista meshes ──────────────────────────────────────
    mean_mesh = _build_structured_mesh(
        surface_data['mean_points'], n_s, n_psi
    )

    # Add the GP uncertainty (σ) as a scalar field on the mesh
    sigma_grid = surface_data['sigma_values'].reshape(n_s, n_psi)
    # Close the tube: append first column
    sigma_closed = np.concatenate([sigma_grid, sigma_grid[:, 0:1]], axis=1)
    mean_mesh.point_data['sigma'] = sigma_closed.ravel()

    if show_bounds:
        upper_mesh = _build_structured_mesh(
            surface_data['upper_points'], n_s, n_psi
        )
        lower_mesh = _build_structured_mesh(
            surface_data['lower_points'], n_s, n_psi
        )

    # ── 5. Build the prior cylinder mesh (optional) ──────────────────
    if show_cylinder:
        center = T[:3, 3]
        z_axis = T[:3, 2]
        projections = (point_cloud - center) @ z_axis
        h_min, h_max = np.min(projections), np.max(projections)
        height   = (h_max - h_min) * 1.1
        z_offset = (h_max + h_min) / 2.0

        cyl_mesh = pv.Cylinder(
            center=(0, 0, z_offset),
            direction=(0, 0, 1),
            radius=1.0,
            height=height,
            resolution=100,
        )
        cyl_mesh.points[:, 0] *= a
        cyl_mesh.points[:, 1] *= b
        cyl_mesh.transform(T, inplace=True)

    # ── 6. Render ────────────────────────────────────────────────────
    plotter = pv.Plotter()

    # Original sonar point cloud
    if show_training_pts and point_cloud.shape[0] > 0:
        plotter.add_points(
            point_cloud,
            color='red',
            point_size=5,
            render_points_as_spheres=True,
            opacity=0.9,
            label='Sonar measurements',
        )

    # Mean surface — colored by uncertainty
    plotter.add_mesh(
        mean_mesh,
        scalars='sigma',
        cmap='coolwarm',
        opacity=0.6,
        show_edges=False,
        label='GP mean surface',
        scalar_bar_args={'title': 'σ (uncertainty)', 'n_labels': 5},
    )

    # Confidence bounds
    if show_bounds:
        plotter.add_mesh(
            upper_mesh,
            color='steelblue',
            opacity=0.15,
            show_edges=False,
            label=f'+{sigma_mult:.0f}σ bound',
        )
        plotter.add_mesh(
            lower_mesh,
            color='navy',
            opacity=0.15,
            show_edges=False,
            label=f'−{sigma_mult:.0f}σ bound',
        )

    # Prior cylinder
    if show_cylinder:
        plotter.add_mesh(
            cyl_mesh,
            color='orange',
            opacity=0.15,
            show_edges=False,
            style='wireframe',
            line_width=1,
            label=f'Prior cylinder (a={a:.2f}, b={b:.2f})',
        )

    plotter.add_axes()
    plotter.add_legend(size=(0.35, 0.20), bcolor=(0.95, 0.95, 0.95))
    plotter.add_title(
        f'Scan {scan_num:04d} — GP Surface Estimation '
        f'(±{sigma_mult:.0f}σ)',
        font_size=11,
    )

    plotter.show(jupyter_backend='trame')

    return gp_result, surface_data


# ══════════════════════════════════════════════════════════════════════════════
# Contour visualization (2D slices — reproduces paper Figures 13/14)
# ══════════════════════════════════════════════════════════════════════════════

def visualize_gp_contours(scan_num,
                          h5_path='HDF5_DATA/7_SCANS.h5',
                          prior_key='INITIAL',
                          gp_result=None,
                          n_psi=200,
                          n_s_slices=5,
                          sigma_mult=3.0,
                          optimize_gp=True):
    """
    Plot 2D contour slices of the GP surface at several positions along
    the trajectory.  This is the equivalent of the ZY-plane contours
    shown in Figures 13 and 14 of the paper.

    For each slice (fixed s), plots:
      - The GP predicted cross-section (mean ± 3σ) in polar-like coords
      - The actual sonar measurements nearby
      - The prior cylinder cross-section

    Parameters
    ----------
    scan_num   : int
    n_psi      : int, angular resolution per slice
    n_s_slices : int, number of trajectory positions to slice at
    sigma_mult : float
    """
    import matplotlib.pyplot as plt
    from gp_surface import train_gp_single_scan, predict_surface

    # ── Load data ────────────────────────────────────────────────────
    sk = f"{scan_num:04d}"
    with h5py.File(h5_path, 'r') as f:
        scan = f[f'SCANS/{sk}']
        M_v_rho      = scan['M_v_rho'][:]
        M_v_psi      = scan['M_v_psi'][:]
        M_v_pose_idx = scan['M_v_pose_idx'][:].astype(int)
        H_scan       = scan['H'][:]
        s_scan       = scan['s'][:]
        pose_idx_min = int(scan['pose_idx_min'][()])

        prior_path = f'SURFACE_PRIOR/{prior_key}'
        T = scan[f'{prior_path}/T'][:]
        a = float(scan[f'{prior_path}/a'][()])
        b = float(scan[f'{prior_path}/b'][()])

    scan_data = {
        'M_v_rho': M_v_rho, 'M_v_psi': M_v_psi,
        'M_v_pose_idx': M_v_pose_idx,
        'H': H_scan, 's': s_scan,
        'pose_idx_min': pose_idx_min,
        'T': T, 'a': a, 'b': b,
    }

    # ── Train GP if needed ───────────────────────────────────────────
    if gp_result is None:
        gp_result, _ = train_gp_single_scan(
            scan_num, h5_path=h5_path,
            prior_key=prior_key, optimize=optimize_gp
        )

    # ── Select slice positions along the trajectory ──────────────────
    s_slices = np.linspace(s_scan.min(), s_scan.max(), n_s_slices + 2)[1:-1]
    psi_dense = np.linspace(-np.pi, np.pi, n_psi)

    # Map each measurement to its s value
    local_idx = np.clip(M_v_pose_idx - pose_idx_min, 0, len(s_scan) - 1)
    s_measurements = s_scan[local_idx]

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_s_slices, figsize=(5 * n_s_slices, 5),
                             subplot_kw={'projection': 'polar'})
    if n_s_slices == 1:
        axes = [axes]

    for ax, s_val in zip(axes, s_slices):
        # GP prediction at this slice
        s_query = np.full(n_psi, s_val)
        rho_mean, rho_var = predict_surface(
            gp_result,
            s_query=s_query, psi_query=psi_dense,
            T=T, a=a, b=b,
            H_scan=H_scan, s_scan=s_scan,
            pose_idx_min=pose_idx_min,
        )
        rho_std = np.sqrt(np.maximum(rho_var, 0.0))

        # Prior cylinder cross-section at this angle
        # For an ellipse: r(ψ) = ab / sqrt((b·cos ψ)² + (a·sin ψ)²)
        # But in our case the prior is computed via ray-cylinder intersection
        from gp_surface import compute_prior_range
        rho_prior = compute_prior_range(
            s_query=s_query, psi_query=psi_dense,
            T=T, a=a, b=b,
            H_scan=H_scan, s_scan=s_scan,
            pose_idx_min=pose_idx_min,
        )

        # Nearby measurements (within a band of s)
        s_range = (s_scan.max() - s_scan.min()) / (2 * n_s_slices)
        nearby = np.abs(s_measurements - s_val) < s_range
        psi_nearby = M_v_psi[nearby]
        rho_nearby = M_v_rho[nearby]

        # --- Plot ---
        # Mean surface
        ax.plot(psi_dense, rho_mean, 'g-', linewidth=2, label='GP mean')

        # Confidence band
        ax.fill_between(
            psi_dense,
            np.maximum(rho_mean - sigma_mult * rho_std, 0),
            rho_mean + sigma_mult * rho_std,
            alpha=0.25, color='green',
            label=f'±{sigma_mult:.0f}σ'
        )

        # Prior
        ax.plot(psi_dense, rho_prior, '--', color='orange', linewidth=1.5,
                label='Prior cylinder')

        # Sonar measurements
        if len(psi_nearby) > 0:
            ax.scatter(psi_nearby, rho_nearby, c='red', s=10, zorder=5,
                       label='Measurements', alpha=0.7)

        ax.set_title(f's = {s_val:.2f}', fontsize=11, pad=15)
        ax.legend(loc='upper right', fontsize=7)

    fig.suptitle(f'Scan {scan_num:04d} — GP Cross-Section Contours',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    return gp_result
