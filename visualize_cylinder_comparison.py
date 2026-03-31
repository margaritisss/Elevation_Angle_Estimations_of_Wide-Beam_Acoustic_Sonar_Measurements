# """
# Before / After visualization of the elliptic-cylinder LM optimization.

# Usage (in your notebook, after the ellipse fitting cell):
# ─────────────────────────────────────────────────────────
#     from visualize_cylinder_comparison import visualize_cylinder_comparison

#     visualize_cylinder_comparison(
#         scan_num       = 150,
#         cylinder_axis  = cylinder_axis,      # from find_cylinder_axes_on_scans
#         ellipse_results = ellipse_results,   # from fit_elliptic_sections
#     )
# """

# import numpy as np
# import pyvista as pv
# from lm_3 import (
#     build_initial_state,
#     optimize_elliptic_cylinder,
# )


# # ─────────────────────────────────────────────────────────────────────
# #  Helper: build a PyVista elliptic-cylinder mesh from (T, a, b)
# # ─────────────────────────────────────────────────────────────────────

# def _make_elliptic_cylinder_mesh(T, a, b, point_cloud):
#     """
#     Create a PyVista mesh for an elliptic cylinder defined by its
#     SE(3) pose T (4×4), semi-major axis a, and semi-minor axis b.

#     The cylinder height is computed from the point cloud so it spans
#     the data with a small margin.
#     """
#     center = T[:3, 3]
#     z_axis = T[:3, 2]

#     # Project points onto cylinder axis to find height bounds
#     projections = (point_cloud - center) @ z_axis
#     h_min, h_max = np.min(projections), np.max(projections)
#     height   = (h_max - h_min) * 1.1
#     z_offset = (h_max + h_min) / 2.0

#     # Unit-radius cylinder aligned with local Z
#     mesh = pv.Cylinder(
#         center=(0, 0, z_offset),
#         direction=(0, 0, 1),
#         radius=1.0,
#         height=height,
#         resolution=100,
#     )

#     # Deform into an ellipse
#     mesh.points[:, 0] *= a
#     mesh.points[:, 1] *= b

#     # Move to world frame
#     mesh.transform(T, inplace=True)

#     return mesh


# # ─────────────────────────────────────────────────────────────────────
# #  Public function
# # ─────────────────────────────────────────────────────────────────────

# def visualize_cylinder_comparison(scan_num, cylinder_axis, ellipse_results,
#                                   max_outer=20, inner_max_nfev=200, tol=1e-8):
#     """
#     For a given scan, run the on-manifold LM optimization and display
#     the BEFORE (purple) and AFTER (orange) elliptic cylinders together
#     with the sonar point cloud (light blue).

#     Parameters
#     ----------
#     scan_num : int
#         Scan number to process (must exist in both dicts).
#     cylinder_axis : dict
#         Output of find_cylinder_axes_on_scans  (keyed by scan number).
#     ellipse_results : dict
#         Output of fit_elliptic_sections        (keyed by scan number).
#     max_outer, inner_max_nfev, tol :
#         Forwarded to optimize_elliptic_cylinder.
#     """
#     # ── 1. Build the initial (before-optimization) state ──────────────
#     initial_state = build_initial_state(
#         cylinder_axis[scan_num],
#         ellipse_results[scan_num],
#     )

#     T_before = initial_state['T'].copy()
#     a_before = initial_state['a']
#     b_before = initial_state['b']
#     point_cloud = initial_state['point_cloud']

#     # ── 2. Run on-manifold LM optimization ────────────────────────────
#     opt_result = optimize_elliptic_cylinder(
#         initial_state,
#         max_outer=max_outer,
#         inner_max_nfev=inner_max_nfev,
#         tol=tol,
#         verbose=True,
#     )

#     T_after = opt_result['T']
#     a_after = opt_result['a']
#     b_after = opt_result['b']

#     # ── 3. Build the two cylinder meshes ──────────────────────────────
#     mesh_before = _make_elliptic_cylinder_mesh(T_before, a_before, b_before, point_cloud)
#     mesh_after  = _make_elliptic_cylinder_mesh(T_after,  a_after,  b_after,  point_cloud)

#     # ── 4. Render ─────────────────────────────────────────────────────
#     plotter = pv.Plotter()

#     # Point cloud
#     plotter.add_points(
#         point_cloud,
#         color='lightblue',
#         point_size=4.5,
#         render_points_as_spheres=True,
#         opacity=0.8,
#         label='Sonar points',
#     )

#     # Before cylinder — purple
#     plotter.add_mesh(
#         mesh_before,
#         color='purple',
#         opacity=0.30,
#         show_edges=False,
#         label=f'Before LM  (a={a_before:.3f}, b={b_before:.3f})',
#     )

#     # After cylinder — orange
#     plotter.add_mesh(
#         mesh_after,
#         color='orange',
#         opacity=0.40,
#         show_edges=False,
#         label=f'After LM   (a={a_after:.3f}, b={b_after:.3f})',
#     )

#     plotter.add_axes()
#     plotter.add_legend(size=(0.25, 0.15), bcolor=(0.95, 0.95, 0.95))
#     plotter.add_title(f'Scan {scan_num:04d} — Elliptic Cylinder Before / After LM', font_size=12)

#     plotter.show(jupyter_backend='trame')

#     return opt_result














"""
Before / After visualization of the elliptic-cylinder LM optimization.

Usage (in your notebook, after the ellipse fitting cell):
─────────────────────────────────────────────────────────
    from visualize_cylinder_comparison import visualize_cylinder_comparison

    visualize_cylinder_comparison(
        scan_num       = 150,
        cylinder_axis  = cylinder_axis,      # from find_cylinder_axes_on_scans
        ellipse_results = ellipse_results,   # from fit_elliptic_sections
    )
"""

import numpy as np
import pyvista as pv
from lm_3 import (
    build_initial_state,
    optimize_elliptic_cylinder,
)


# ─────────────────────────────────────────────────────────────────────
#  Helper: build a PyVista elliptic-cylinder mesh from (T, a, b)
# ─────────────────────────────────────────────────────────────────────

def _make_elliptic_cylinder_mesh(T, a, b, point_cloud):
    """
    Create a PyVista mesh for an elliptic cylinder defined by its
    SE(3) pose T (4×4), semi-major axis a, and semi-minor axis b.

    The cylinder height is computed from the point cloud so it spans
    the data with a small margin.
    """
    center = T[:3, 3]
    z_axis = T[:3, 2]

    # Project points onto cylinder axis to find height bounds
    projections = (point_cloud - center) @ z_axis
    h_min, h_max = np.min(projections), np.max(projections)
    height   = (h_max - h_min) * 1.1
    z_offset = (h_max + h_min) / 2.0

    # Unit-radius cylinder aligned with local Z
    mesh = pv.Cylinder(
        center=(0, 0, z_offset),
        direction=(0, 0, 1),
        radius=1.0,
        height=height,
        resolution=100,
    )

    # Deform into an ellipse
    mesh.points[:, 0] *= a
    mesh.points[:, 1] *= b

    # Move to world frame
    mesh.transform(T, inplace=True)

    return mesh


# ─────────────────────────────────────────────────────────────────────
#  Public function
# ─────────────────────────────────────────────────────────────────────

def visualize_cylinder_comparison(scan_num, cylinder_axis, ellipse_results,
                                  max_outer=50, tol=1e-8):
    """
    For a given scan, run the on-manifold LM optimization and display
    the BEFORE (purple) and AFTER (orange) elliptic cylinders together
    with the sonar point cloud (light blue).

    Parameters
    ----------
    scan_num : int
        Scan number to process (must exist in both dicts).
    cylinder_axis : dict
        Output of find_cylinder_axes_on_scans  (keyed by scan number).
    ellipse_results : dict
        Output of fit_elliptic_sections        (keyed by scan number).
    max_outer, tol :
        Forwarded to optimize_elliptic_cylinder.
    """
    # ── 1. Build the initial (before-optimization) state ──────────────
    initial_state = build_initial_state(
        cylinder_axis[scan_num],
        ellipse_results[scan_num],
    )

    T_before = initial_state['T'].copy()
    a_before = initial_state['a']
    b_before = initial_state['b']
    point_cloud = initial_state['point_cloud']

    # ── 2. Run on-manifold LM optimization ────────────────────────────
    opt_result = optimize_elliptic_cylinder(
        initial_state,
        max_outer=max_outer,
        tol=tol,
        verbose=True,
    )

    T_after = opt_result['T']
    a_after = opt_result['a']
    b_after = opt_result['b']

    # ── 3. Build the two cylinder meshes ──────────────────────────────
    mesh_before = _make_elliptic_cylinder_mesh(T_before, a_before, b_before, point_cloud)
    mesh_after  = _make_elliptic_cylinder_mesh(T_after,  a_after,  b_after,  point_cloud)

    # ── 4. Render ─────────────────────────────────────────────────────
    plotter = pv.Plotter()

    # Point cloud
    plotter.add_points(
        point_cloud,
        color='lightblue',
        point_size=4.5,
        render_points_as_spheres=True,
        opacity=0.8,
        label='Sonar points',
    )

    # Before cylinder — purple
    plotter.add_mesh(
        mesh_before,
        color='purple',
        opacity=0.30,
        show_edges=False,
        label=f'Before LM  (a={a_before:.3f}, b={b_before:.3f})',
    )

    # After cylinder — orange
    plotter.add_mesh(
        mesh_after,
        color='orange',
        opacity=0.40,
        show_edges=False,
        label=f'After LM   (a={a_after:.3f}, b={b_after:.3f})',
    )

    plotter.add_axes()
    plotter.add_legend(size=(0.25, 0.15), bcolor=(0.95, 0.95, 0.95))
    plotter.add_title(f'Scan {scan_num:04d} — Elliptic Cylinder Before / After LM', font_size=12)

    plotter.show(jupyter_backend='trame')

    return opt_result