import h5py
import numpy as np
from rpca import rpca_on_scans

def find_cylinder_axis(point_cloud, components):
    """
    Identify the cylinder axis among 3 principal vectors from RPCA.
    Section 4.2.1: tangent projections are minimal along the axis.
    """
    tangents = np.diff(point_cloud, axis=0)
    projection_sums = np.zeros(3)
    for k in range(3):
        projection_sums[k] = np.sum(np.abs(tangents @ components[k])**2)

    axis_index  = np.argmin(projection_sums)
    axis_vector = components[axis_index]
    return axis_index, axis_vector, projection_sums

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
        K=K, trim_pct=trim_pct, seed=seed, verbose=False,
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
                'point_cloud':      pc,
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


