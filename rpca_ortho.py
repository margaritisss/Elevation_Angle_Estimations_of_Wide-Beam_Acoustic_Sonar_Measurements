"""
Robust Principal Component Analysis via Trimmed Grassmann Average (TGA).

Implementation based on:
    Hauberg et al., "Scalable Robust Principal Component Analysis Using
    Grassmann Averages," IEEE TPAMI, vol. 38, no. 11, Nov. 2016.

This module provides:
    - tga_component  : compute a single robust principal component
    - tga_rpca       : compute K robust components (full TGA pipeline)
    - rpca_on_scans  : apply TGA RPCA to point clouds stored in 7_SCANS.h5
"""

import numpy as np
import h5py
import os


# ──────────────────────────────────────────────────────────────────────
#  Core TGA algorithm
# ──────────────────────────────────────────────────────────────────────

def _trimmed_weighted_mean(weights, vectors, trim_pct=0.50): 
    """
    Element-wise trimmed weighted mean (Algorithm detail from §3.3).

    For each dimension d independently, remove the trim_pct fraction of
    the smallest and the trim_pct fraction of the largest weighted values,
    then average the remaining middle portion.

    When trim_pct = 0.50 this reduces to the element-wise weighted median
    (maximally robust, as in §4 of the paper).

    Parameters
    ----------
    weights  : (N,) array of weights  w_n = ||x_n||
    vectors  : (N, D) sign-corrected unit vectors  a_n * u_n
    trim_pct : float in [0, 0.5], fraction to trim from each tail

    Returns
    -------
    m_rob : (D,) robust mean vector (un-normalised)
    """
    N, D = vectors.shape # number of points, dimensionality
    # Weighted vectors:  w_n * a_n * u_n
    wv = weights[:, None] * vectors  # (N, D)

    n_trim = int(np.floor(trim_pct * N)) # number of points to trim from each tail
    if n_trim == 0: 
        # No trimming — ordinary weighted mean
        return wv.mean(axis=0)

    # For P=50% (median), use numpy's fast median
    if n_trim >= N // 2:
        return np.median(wv, axis=0)

    m_rob = np.empty(D)
    for d in range(D):
        col = wv[:, d]
        idx = np.argsort(col)
        # Keep the middle portion (trim smallest and largest)
        kept = col[idx[n_trim: N - n_trim]]
        m_rob[d] = kept.mean()
    return m_rob

def tga_component(X, trim_pct=0.50, max_iter=500, tol=1e-8, rng=None):
    """
    Compute a single robust principal component via the Trimmed Grassmann
    Average algorithm (§3.3, Algorithm "Robust Grassmann Average").

    Parameters
    ----------
    X        : (N, D) zero-centred data matrix
    trim_pct : float, trimming percentage per tail (default 0.50 = median)
    max_iter : int, maximum number of iterations
    tol      : float, convergence threshold on ||q_i - q_{i-1}||
    rng      : numpy Generator (for reproducibility)

    Returns
    -------
    q   : (D,) unit vector — the robust principal component
    lam : float — robust "variance" captured  Σ |x_n^T q|
    """
    if rng is None:
        rng = np.random.default_rng()

    N, D = X.shape
    norms = np.linalg.norm(X, axis=1)        # w_n = ||x_n||

    # Remove zero-norm rows (degenerate)
    mask = norms > 1e-15
    X_clean = X[mask]
    norms   = norms[mask]
    N       = X_clean.shape[0]
    if N == 0:
        return np.zeros(D), 0.0

    U = X_clean / norms[:, None]             # u_n = x_n / ||x_n||   (N, D)
    w = norms                                # weights

    # Random initialisation on the unit sphere
    q = rng.standard_normal(D)
    q /= np.linalg.norm(q)

    for _ in range(max_iter):
        # Step 1: sign assignment   a_n = sign(u_n^T q)
        dots = U @ q                         # (N,)
        signs = np.sign(dots)
        signs[signs == 0] = 1.0              # break ties

        # Step 2: trimmed robust mean
        aU = signs[:, None] * U              # sign-corrected unit vectors
        m_rob = _trimmed_weighted_mean(w, aU, trim_pct)

        # Step 3: normalise
        m_norm = np.linalg.norm(m_rob)
        if m_norm < 1e-15:
            break
        q_new = m_rob / m_norm

        # Check convergence
        if np.linalg.norm(q_new - q) < tol:
            q = q_new
            break
        q = q_new

    # Robust "variance" captured by this component: Σ |x_n^T q|
    lam = np.sum(np.abs(X_clean @ q))

    return q, lam

def tga_rpca(X, K=3, trim_pct=0.50, max_iter=500, tol=1e-8, seed=42):
    """
    Full TGA–RPCA: compute K robust principal components with deflation.

    Steps (from the paper, §3.2.1 & §3.3):
        1. Subtract the element-wise median (robust centre).
        2. Compute the leading component via TGA.
        3. Deflate: X̂ = X − (X q) q^T   (Eq. 15)
        4. Repeat for K components.
        5. Post-hoc reorder by decreasing captured variance.

    Parameters
    ----------
    X        : (N, D) raw point cloud data
    K        : int, number of components to compute (default 3 for 3D)
    trim_pct : float, trimming fraction per tail (default 0.50)
    max_iter : int
    tol      : float
    seed     : int, random seed for reproducibility

    Returns
    -------
    result : dict with keys:
        'components' : (K, D) array — robust principal components (rows)
        'center'     : (D,)  array — robust centre (element-wise median)
        'variances'  : (K,)  array — captured variance per component
        'X_centred'  : (N, D) centred data used for the decomposition
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    K = min(K, D)  # cannot have more components than dimensions

    # Step 1: robust centre — element-wise median (§4, practical notes)
    center = np.median(X, axis=0)
    X_c = X - center

    components = []
    variances  = []
    X_res = X_c.copy()

    for k in range(K):
        q, lam = tga_component(X_res, trim_pct=trim_pct,
                               max_iter=max_iter, tol=tol, rng=rng)

        # ── Gram-Schmidt orthogonalisation against previous components ──
        # TGA + deflation does NOT algebraically guarantee orthogonality
        # (unlike eigen-decomposition in classical PCA).  We enforce it
        # explicitly:  subtract from q its projection onto every earlier
        # component, then re-normalise.
        for prev_q in components:
            q = q - np.dot(q, prev_q) * prev_q
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-15:
            # q collapsed — data has fewer than K effective dimensions
            break
        q = q / q_norm

        # Recompute captured variance with the orthogonalised direction
        lam = np.sum(np.abs(X_res @ q))

        components.append(q)
        variances.append(lam)

        # Deflation (Eq. 15):  X̂ = X − (X q) q^T
        projections = X_res @ q                  # (N,)
        X_res = X_res - np.outer(projections, q)

    components = np.array(components)   # (K, D)
    variances  = np.array(variances)    # (K,)

    # Step 5: post-hoc reorder by decreasing variance
    order = np.argsort(-variances)
    components = components[order]
    variances  = variances[order]

    return {
        'components': components,
        'center':     center,
        'variances':  variances,
        'X_centred':  X_c,
    }


# ──────────────────────────────────────────────────────────────────────
#  Convenience wrapper for the scan-based HDF5 workflow
# ──────────────────────────────────────────────────────────────────────

def rpca_on_scans(scan_ids='all',
                  h5_path='HDF5_DATA/7_SCANS.h5',
                  K=3, trim_pct=0.50, seed=42, verbose=True):
    """
    Apply TGA RPCA to point clouds stored in 7_SCANS.h5.

    Parameters
    ----------
    scan_ids : int, list of ints, or 'all'
        Which scans to process.
        - int        → single scan   (e.g. 1)
        - list[int]  → several scans (e.g. [1, 2, 5])
        - 'all'      → every scan in the file
    h5_path  : str, path to the HDF5 file
    K        : int, number of components (default 3)
    trim_pct : float, trimming percentage (default 0.50)
    seed     : int, random seed
    verbose  : bool

    Returns
    -------
    results : dict  { scan_number: rpca_result_dict }
        Each rpca_result_dict contains:
            'components' : (K, 3) principal components
            'center'     : (3,) robust centre
            'variances'  : (K,) captured variance
            'n_points'   : int, number of points in the cloud
    """
    results = {}

    with h5py.File(h5_path, 'r') as f:
        scans_grp = f['SCANS']
        all_keys = sorted(scans_grp.keys())   # ['0001', '0002', ...]

        # Resolve scan_ids → list of string keys
        if scan_ids == 'all':
            keys_to_process = all_keys
        elif isinstance(scan_ids, int):
            keys_to_process = [f'{scan_ids:04d}']
        elif isinstance(scan_ids, (list, tuple)):
            keys_to_process = [f'{sid:04d}' for sid in scan_ids]
        else:
            raise ValueError("scan_ids must be int, list of ints, or 'all'")

        for key in keys_to_process:
            if key not in scans_grp:
                if verbose:
                    print(f"  ⚠ Scan {key} not found — skipping")
                continue

            pc = scans_grp[key]['POINT_CLOUD'][:]   # (n_points, 3)
            n_pts = pc.shape[0]

            if n_pts < K:
                if verbose:
                    print(f"  ⚠ Scan {key}: only {n_pts} points, need ≥ {K} — skipping")
                continue

            rpca = tga_rpca(pc, K=K, trim_pct=trim_pct, seed=seed)

            scan_num = int(key)
            results[scan_num] = {
                'components': rpca['components'],
                'center':     rpca['center'],
                'variances':  rpca['variances'],
                'n_points':   n_pts,
            }

            # --- UPDATED PRINT BLOCK ---
            if verbose:
                print(f"--- Summary for Scan {key} ---")
                print(f"Number of points used: {n_pts}")
                print(f"Robust Center (Median):\n{rpca['center']}")
                print()
                print(f"Captured Variances:\n{rpca['variances']}")
                print()
                print(f"The {K} Principal Vectors (Components Matrix):\n{rpca['components']}\n")

    # Optional: You can remove or keep this final confirmation
    if verbose:
        print(f"✓ Processed {len(results)} scan(s)")

    return results