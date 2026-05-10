"""
Algorithm 1 — Line 4:   for each ψ_h_i, ρ_h_ij  →  Estimation
Algorithm 2 — Lines 1–33: Elevation Angle MLE

=============================================================================
OPTIMIZED implementation fixing two critical issues:

1. PERFORMANCE:  The original had O(n_h × n_q × n_r) Python loops with
   expensive per-element Rotation/Slerp/GP calls.  This version:
     - Precomputes ALL sonar-plane normals and projections once
     - Vectorizes the segment search
     - Batches GP predictions (one call per θ sample, not per segment)
     - Caches Rotation objects

2. CORRECTNESS:  The vertical sonar plane normal n^v must match the actual
   quaternion chain in compute_point_cloud.  The default [1,0,0] was wrong.
   This version derives n_v_local by examining what axis the vertical sonar
   rotation ψ^v sweeps around — after the frame alignment rotations
   (Rot_Z by π/2, Rot_Y by π/2), the scan-plane normal in body frame
   becomes [0, 1, 0] (the Y-axis), not [1, 0, 0].
=============================================================================
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from math import cos, sin, pi


# ═══════════════════════════════════════════════════════════════════════════
# Quaternion helpers  — same [w, x, y, z] convention as the notebook
# ═══════════════════════════════════════════════════════════════════════════

def _qmul(q1, q2):
    """Hamilton product [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _qconj(q):
    """Conjugate of [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


# ═══════════════════════════════════════════════════════════════════════════
# Derive the correct n_v_local from the actual quaternion chain
# ═══════════════════════════════════════════════════════════════════════════

def _derive_vertical_sonar_plane_normal():
    """
    Determine the vertical sonar scan-plane normal in the BODY frame
    by examining the quaternion chain used in compute_point_cloud.

    The chain is:
      Q_body(ψ) = Q_rot(ψ) ⊗ Rot_Z(π/2) ⊗ Rot_Y(π/2)

    where Q_rot(ψ) = [cos(ψ/2), -sin(ψ/2), 0, 0] rotates about X.

    As ψ varies, Q_body sweeps a ray in a plane.  The normal to that
    plane (in body frame) is what we need.

    Method: compute the ray direction at ψ=0 and ψ=π/2 — the cross
    product gives the plane normal.
    """
    def _ray_direction_body(psi):
        psi_half = psi / 2.0
        Q = np.array([cos(psi_half), -sin(psi_half), 0.0, 0.0])
        Rot_Z = np.array([cos(pi/2), 0.0, 0.0, sin(pi/2)])
        Q = _qmul(Q, Rot_Z)
        Rot_Y = np.array([cos(pi/2), 0.0, sin(pi/2), 0.0])
        Q = _qmul(Q, Rot_Y)
        # Rotate unit vector [0,0,0,1]
        unit = np.array([0.0, 0.0, 0.0, 1.0])
        rotated = _qmul(Q, _qmul(unit, _qconj(Q)))
        return rotated[1:4]

    d0 = _ray_direction_body(0.0)
    d1 = _ray_direction_body(np.pi / 2.0)
    normal = np.cross(d0, d1)
    norm = np.linalg.norm(normal)
    if norm > 1e-10:
        normal /= norm
    return normal

# Compute once at module load
_N_V_LOCAL_DEFAULT = _derive_vertical_sonar_plane_normal()


# ═══════════════════════════════════════════════════════════════════════════
# g^h :  Horizontal sonar measurement → 3D world point   (Equation 1 / 20)
# ═══════════════════════════════════════════════════════════════════════════

def g_h(rho_h, theta, psi_h, robot_pos, robot_quat):
    """
    Map horizontal sonar measurement to 3D world point  (Eq. 1/20).

    Uses the same quaternion chain as compute_point_cloud but adds
    the elevation angle θ via the decomposition:
      horizontal_range = ρ · cos(θ)   (in the sonar scan plane)
      vertical_range   = ρ · sin(θ)   (perpendicular to scan plane)
    """
    psi_half = psi_h / 2.0
    Q_psi = np.array([cos(psi_half), -sin(psi_half), 0.0, 0.0])

    Rot_Z = np.array([cos(pi / 2), 0.0, 0.0, sin(pi / 2)])
    Q_body = _qmul(Q_psi, Rot_Z)

    Rot_Y = np.array([cos(pi / 2), 0.0, sin(pi / 2), 0.0])
    Q_body = _qmul(Q_body, Rot_Y)

    Q_global = _qmul(np.asarray(robot_quat, dtype=float), Q_body)

    # Horizontal component: ρ·cos(θ) along the beam direction at θ=0
    rho_horiz = rho_h * np.cos(theta)
    Impact_Dist = np.array([0.0, 0.0, 0.0, rho_horiz])
    Q_temp = _qmul(Q_global, _qmul(Impact_Dist, _qconj(Q_global)))
    horiz_vec = Q_temp[1:4]

    # Vertical component: ρ·sin(θ) along the beam aperture direction
    # This is the sonar plane normal in world frame
    qw, qx, qy, qz = robot_quat
    R_robot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    elev_dir = R_robot @ _N_V_LOCAL_DEFAULT
    vert_vec = rho_h * np.sin(theta) * elev_dir

    return robot_pos + horiz_vec + vert_vec


# ═══════════════════════════════════════════════════════════════════════════
# Precompute trajectory normals and rotation matrices (done ONCE per scan)
# ═══════════════════════════════════════════════════════════════════════════

class _TrajectoryCache:
    """
    Precomputes and caches all trajectory-dependent quantities:
      - Rotation matrices R_i for every pose
      - Sonar plane normals n^v_i in world frame
      - Positions τ_i
      - SLERP interpolators for each segment
    """
    def __init__(self, H_scan, s_scan, n_v_local=None):
        self.H = H_scan
        self.s = s_scan
        self.n_r = H_scan.shape[0]
        self.n_v_local = n_v_local if n_v_local is not None else _N_V_LOCAL_DEFAULT

        # Precompute rotation matrices and normals
        self.positions = H_scan[:, :3]  # (n_r, 3)
        self.R_mats = np.zeros((self.n_r, 3, 3))
        self.normals = np.zeros((self.n_r, 3))
        self.rotations = []

        for i in range(self.n_r):
            qw, qx, qy, qz = H_scan[i, 3:]
            rot = Rotation.from_quat([qx, qy, qz, qw])
            self.rotations.append(rot)
            R = rot.as_matrix()
            self.R_mats[i] = R
            self.normals[i] = R @ self.n_v_local

    def find_segments(self, x_point):
        """
        Vectorized segment search (Equation 22).
        Returns list of segment start indices.
        """
        # Compute projections for ALL poses at once
        diffs = x_point[np.newaxis, :] - self.positions  # (n_r, 3)
        projs = np.sum(self.normals * diffs, axis=1)     # (n_r,)

        # Sign change between consecutive poses
        signs = projs[:-1] * projs[1:]  # (n_r-1,)
        return np.where(signs < 0)[0].tolist()

    def interpolate_pose(self, i, t_hat):
        """Interpolate pose at segment [i, i+1] with parameter t_hat."""
        tau = self.positions[i] + t_hat * (self.positions[i+1] - self.positions[i])

        # SLERP
        slerp = Slerp([0, 1], Rotation.concatenate([
            self.rotations[i], self.rotations[i+1]
        ]))
        r_hat = slerp(t_hat)
        q_xyzw = r_hat.as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        return np.concatenate([tau, q_wxyz]), r_hat.as_matrix()

    def find_t(self, i, x_point):
        """
        Find interpolation parameter t̂ for segment [i, i+1].
        Linear root-finding (Eq. 28/30).
        """
        f0 = np.dot(self.normals[i],   x_point - self.positions[i])
        f1 = np.dot(self.normals[i+1], x_point - self.positions[i+1])

        denom = f1 - f0
        if abs(denom) < 1e-15:
            return 0.5
        t_hat = -f0 / denom
        return np.clip(t_hat, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Virtual vertical sonar parameters   (Equations 31–34)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_virtual_params(x_point, tau_hat, R_hat, s_scan, i, t_hat):
    """
    Equations 31–34: compute ŝ, ψ̂^v, ρ̂^v from the interpolated pose.
    Takes pre-computed R_hat to avoid recomputation.
    """
    # Eq. 32: express in vertical sonar frame
    x_local = R_hat.T @ (x_point - tau_hat)

    # Eq. 33: virtual rotation angle
    psi_hat = np.arctan2(x_local[1], x_local[0])

    # Eq. 34: virtual range
    rho_hat = np.linalg.norm(x_local)

    # Eq. 31: curvilinear abscissa
    if i + 1 < len(s_scan):
        s_hat = s_scan[i] + t_hat * (s_scan[i + 1] - s_scan[i])
    else:
        s_hat = s_scan[i]

    return s_hat, psi_hat, rho_hat


# ═══════════════════════════════════════════════════════════════════════════
# Log-likelihood   (Equation 36)
# ═══════════════════════════════════════════════════════════════════════════

def _log_likelihood(rho_hat_v, rho_bar_star, sigma2_star):
    """Equation (36): Gaussian log-likelihood."""
    if sigma2_star <= 1e-20:
        return -np.inf
    residual = rho_bar_star - rho_hat_v
    return -(residual**2 / (2.0 * sigma2_star)
             + 0.5 * np.log(sigma2_star)
             + 0.5 * np.log(2.0 * np.pi))


# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM 2  —  Optimized implementation
# ═══════════════════════════════════════════════════════════════════════════

def elevation_angle_mle(gp_result, scan_data,
                        psi_h_k, rho_h_k, pose_idx_h_k,
                        beam_width=np.radians(35), n_q=200,
                        n_v_local=None, _cache=None):
    """
    Algorithm 2: Elevation Angle MLE for a single horizontal measurement.

    Parameters
    ----------
    _cache : _TrajectoryCache, optional — pass to avoid recomputing
             trajectory normals/rotations for every measurement.
    """
    from gp_surface import predict_surface

    H_scan       = scan_data['H']
    s_scan       = scan_data['s']
    pose_idx_min = scan_data['pose_idx_min']
    T            = scan_data['T']
    a            = scan_data['a']
    b            = scan_data['b']
    n_r          = H_scan.shape[0]

    # Build or reuse trajectory cache
    if _cache is None:
        _cache = _TrajectoryCache(H_scan, s_scan, n_v_local)

    # Robot pose for this measurement
    local_idx = int(pose_idx_h_k - pose_idx_min)
    local_idx = max(0, min(local_idx, n_r - 1))
    robot_pos  = H_scan[local_idx, :3]
    robot_quat = H_scan[local_idx, 3:]

    # Line 1
    theta_min = -beam_width / 2.0
    theta_max =  beam_width / 2.0
    delta_theta = (theta_max - theta_min) / n_q

    Theta_hat     = []
    theta_samples = np.zeros(n_q)
    l_max_all     = np.full(n_q, -np.inf)

    # Collect ALL (s_hat, psi_hat, rho_hat) pairs for batch GP prediction
    query_list = []  # list of (q_idx, s_hat, psi_hat_v, rho_hat_v)

    # ── PASS 1: Compute g^h, find segments, interpolate, get virtual params ──
    for q in range(n_q):
        theta_q = theta_min + (q + 0.5) * delta_theta
        theta_samples[q] = theta_q

        # Line 4: x(θ_q)
        x = g_h(rho_h_k, theta_q, psi_h_k, robot_pos, robot_quat)

        # Lines 6–11: segment selection (vectorized)
        I_segments = _cache.find_segments(x)

        if not I_segments:
            continue

        # For each segment, compute virtual params
        best_s = None
        best_psi = None
        best_rho = None
        best_dist = np.inf  # pick segment whose interpolated pose is closest

        for i in I_segments:
            t_hat = _cache.find_t(i, x)
            eta_hat, R_hat = _cache.interpolate_pose(i, t_hat)
            s_hat, psi_hat_v, rho_hat_v = _compute_virtual_params(
                x, eta_hat[:3], R_hat, s_scan, i, t_hat
            )
            # Among multiple segments, pick the closest one
            dist = np.linalg.norm(x - eta_hat[:3])
            if dist < best_dist:
                best_dist = dist
                best_s = s_hat
                best_psi = psi_hat_v
                best_rho = rho_hat_v

        if best_s is not None:
            query_list.append((q, best_s, best_psi, best_rho))

    # ── PASS 2: Batch GP prediction ──────────────────────────────────────
    if query_list:
        q_indices  = np.array([ql[0] for ql in query_list])
        s_queries  = np.array([ql[1] for ql in query_list])
        psi_queries = np.array([ql[2] for ql in query_list])
        rho_hats    = np.array([ql[3] for ql in query_list])

        # ONE call to predict_surface for ALL θ samples that found segments
        try:
            rho_mean, rho_var = predict_surface(
                gp_result,
                s_query=s_queries,
                psi_query=psi_queries,
                T=T, a=a, b=b,
                H_scan=H_scan, s_scan=s_scan,
                pose_idx_min=pose_idx_min,
            )

            # Compute log-likelihoods
            for idx_in_batch, q in enumerate(q_indices):
                l_q = _log_likelihood(
                    rho_hats[idx_in_batch],
                    rho_mean[idx_in_batch],
                    rho_var[idx_in_batch]
                )
                l_max_all[q] = l_q
        except Exception as e:
            pass  # All stay at -inf

    # ── PASS 3: Detect local maxima (Eq. 19, lines 30–32) ────────────────
    for q in range(2, n_q):
        if (l_max_all[q] < l_max_all[q - 1]) and \
           (l_max_all[q - 1] > l_max_all[q - 2]) and \
           np.isfinite(l_max_all[q - 1]):
            Theta_hat.append(theta_samples[q - 1])

    return {
        'Theta_hat':       Theta_hat,
        'log_likelihoods': l_max_all,
        'theta_samples':   theta_samples,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM 1, LINE 4
# ═══════════════════════════════════════════════════════════════════════════

def estimate_elevation_angles_for_scan(gp_result, scan_data,
                                        beam_width=np.radians(35),
                                        n_q=200, n_v_local=None,
                                        verbose=True):
    """
    Algorithm 1, Line 4: for each (ψ^h_i, ρ^h_ij), call Algorithm 2.
    """
    M_h_M        = scan_data['M_h_M']
    M_h_pose_idx = scan_data['M_h_pose_idx']
    n_h = M_h_M.shape[1]

    # ── Build the trajectory cache ONCE for the whole scan ───────────────
    cache = _TrajectoryCache(
        scan_data['H'], scan_data['s'], n_v_local
    )

    all_results = []

    if verbose:
        print(f"  Processing {n_h} horizontal measurements...")
        print(f"  Trajectory: {cache.n_r} poses")
        print(f"  n_v_local (body frame): {cache.n_v_local}")

    for k in range(n_h):
        rho_h_k = M_h_M[0, k]
        psi_h_k = M_h_M[2, k]

        if rho_h_k <= 0 or not np.isfinite(rho_h_k):
            continue

        pose_idx_h_k = int(M_h_pose_idx[k])

        result = elevation_angle_mle(
            gp_result=gp_result,
            scan_data=scan_data,
            psi_h_k=psi_h_k,
            rho_h_k=rho_h_k,
            pose_idx_h_k=pose_idx_h_k,
            beam_width=beam_width,
            n_q=n_q,
            n_v_local=n_v_local,
            _cache=cache,
        )

        result['measurement_idx'] = k
        result['rho_h'] = rho_h_k
        result['psi_h'] = psi_h_k
        all_results.append(result)

        if verbose and ((k + 1) % 50 == 0 or (k + 1) == n_h):
            n_found = len(result['Theta_hat'])
            # Count how many θ samples found segments
            n_with_segments = np.sum(np.isfinite(result['log_likelihoods']))
            print(f"    [{k+1}/{n_h}]  ψ_h={np.degrees(psi_h_k):6.1f}°  "
                  f"ρ_h={rho_h_k:.2f}m  →  {n_found} angle(s)  "
                  f"({n_with_segments}/{n_q} samples with segments)")

    if verbose:
        total = sum(len(r['Theta_hat']) for r in all_results)
        n_with = sum(1 for r in all_results if len(r['Theta_hat']) > 0)
        print(f"  ✓ Done: {total} elevation angles from "
              f"{n_with}/{len(all_results)} measurements")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE + DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_single_measurement(gp_result, scan_data, k=0,
                                  beam_width=np.radians(35), n_q=50,
                                  n_v_local=None):
    """
    Run Algorithm 2 on a single measurement with verbose diagnostics.
    Useful for debugging why no angles are found.
    """
    M_h_M = scan_data['M_h_M']
    M_h_pose_idx = scan_data['M_h_pose_idx']

    rho_h_k = M_h_M[0, k]
    psi_h_k = M_h_M[2, k]
    pose_idx_h_k = int(M_h_pose_idx[k])

    H_scan = scan_data['H']
    s_scan = scan_data['s']
    pose_idx_min = scan_data['pose_idx_min']

    cache = _TrajectoryCache(H_scan, s_scan, n_v_local)

    local_idx = int(pose_idx_h_k - pose_idx_min)
    local_idx = max(0, min(local_idx, H_scan.shape[0] - 1))
    robot_pos  = H_scan[local_idx, :3]
    robot_quat = H_scan[local_idx, 3:]

    print(f"=== Diagnostic for measurement k={k} ===")
    print(f"  ψ_h = {np.degrees(psi_h_k):.1f}°,  ρ_h = {rho_h_k:.2f} m")
    print(f"  Robot pos = {robot_pos}")
    print(f"  Robot quat = {robot_quat}")
    print(f"  n_v_local = {cache.n_v_local}")
    print(f"  Trajectory: {cache.n_r} poses")
    print(f"  s range: [{s_scan[0]:.3f}, {s_scan[-1]:.3f}]")
    print()

    # Test a few θ values
    theta_min = -beam_width / 2.0
    theta_max =  beam_width / 2.0
    test_thetas = np.linspace(theta_min, theta_max, min(n_q, 10))

    for theta_q in test_thetas:
        x = g_h(rho_h_k, theta_q, psi_h_k, robot_pos, robot_quat)
        segments = cache.find_segments(x)
        print(f"  θ={np.degrees(theta_q):+6.2f}°  x={x}  "
              f"segments={segments}")

        if segments:
            i = segments[0]
            t_hat = cache.find_t(i, x)
            eta_hat, R_hat = cache.interpolate_pose(i, t_hat)
            s_hat, psi_hat_v, rho_hat_v = _compute_virtual_params(
                x, eta_hat[:3], R_hat, s_scan, i, t_hat
            )
            print(f"         t̂={t_hat:.4f}  ŝ={s_hat:.4f}  "
                  f"ψ̂^v={np.degrees(psi_hat_v):.1f}°  ρ̂^v={rho_hat_v:.3f}")

    print()
    print("Now running full MLE...")
    result = elevation_angle_mle(
        gp_result=gp_result,
        scan_data=scan_data,
        psi_h_k=psi_h_k,
        rho_h_k=rho_h_k,
        pose_idx_h_k=pose_idx_h_k,
        beam_width=beam_width,
        n_q=n_q,
        n_v_local=n_v_local,
        _cache=cache,
    )

    n_finite = np.sum(np.isfinite(result['log_likelihoods']))
    print(f"  Samples with segments: {n_finite}/{n_q}")
    print(f"  Angles found: {result['Theta_hat']}")
    if n_finite > 0:
        valid = result['log_likelihoods'][np.isfinite(result['log_likelihoods'])]
        print(f"  Log-likelihood range: [{valid.min():.2f}, {valid.max():.2f}]")

    return result


def run_elevation_mle_on_scan(scan_num, h5_path='HDF5_DATA/7_SCANS.h5',
                               prior_key='INITIAL',
                               beam_width=np.radians(35),
                               n_q=200, n_v_local=None,
                               verbose=True):
    """
    Full pipeline: load data → train GP → run Algorithm 2 on all M^h.
    """
    import h5py
    from gp_surface import train_gp_single_scan

    if verbose:
        print(f"{'='*60}")
        print(f"Elevation Angle MLE — Scan {scan_num:04d}")
        print(f"{'='*60}")
        print(f"  Derived n_v_local = {_N_V_LOCAL_DEFAULT}")

    # Step 1: Train GP
    if verbose:
        print(f"\n[1/2] Training GP surface model...")
    gp_result, scan_data_raw = train_gp_single_scan(
        scan_num, h5_path=h5_path, prior_key=prior_key
    )
    if gp_result is None:
        if verbose:
            print("  ⚠ GP training failed. Aborting.")
        return None, None, None

    # Step 2: Load horizontal sonar data
    sk = f"{scan_num:04d}"
    with h5py.File(h5_path, 'r') as f:
        scan = f[f'SCANS/{sk}']
        M_h_M        = scan['M_h_M'][:]
        M_h_pose_idx = scan['M_h_pose_idx'][:].astype(int)

    scan_data = {
        'H':            scan_data_raw['H'],
        's':            scan_data_raw['s'],
        'pose_idx_min': scan_data_raw['pose_idx_min'],
        'T':            scan_data_raw['T'],
        'a':            scan_data_raw['a'],
        'b':            scan_data_raw['b'],
        'M_h_M':        M_h_M,
        'M_h_pose_idx': M_h_pose_idx,
    }

    # Step 3: Run Algorithm 2
    if verbose:
        print(f"\n[2/2] Running Elevation Angle MLE (Algorithm 2)...")
    mle_results = estimate_elevation_angles_for_scan(
        gp_result=gp_result,
        scan_data=scan_data,
        beam_width=beam_width,
        n_q=n_q,
        n_v_local=n_v_local,
        verbose=verbose,
    )

    if verbose:
        total = sum(len(r['Theta_hat']) for r in mle_results)
        n_with = sum(1 for r in mle_results if len(r['Theta_hat']) > 0)
        print(f"\n{'='*60}")
        print(f"Scan {scan_num:04d} Summary:")
        print(f"  Horizontal measurements : {len(mle_results)}")
        print(f"  With elevation angle(s) : {n_with}")
        print(f"  Total angles found      : {total}")
        print(f"{'='*60}")

    return mle_results, gp_result, scan_data


if __name__ == '__main__':
    import sys
    scan_num = int(sys.argv[1]) if len(sys.argv) > 1 else 57
    h5_path  = sys.argv[2] if len(sys.argv) > 2 else 'HDF5_DATA/7_SCANS.h5'
    results, gp, sd = run_elevation_mle_on_scan(
        scan_num, h5_path=h5_path, verbose=True
    )
    if results:
        print("\nExample results (first 5):")
        for r in results[:5]:
            k = r['measurement_idx']
            angles = [f"{np.degrees(a):.2f}°" for a in r['Theta_hat']]
            print(f"  M[{k}]:  ψ_h={np.degrees(r['psi_h']):.1f}°  "
                  f"ρ_h={r['rho_h']:.2f}m  →  θ̂ = {angles}")