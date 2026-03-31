"""
Algorithm 1 — Line 3:  f ← GP(s, M^v, f_p)

Gaussian Process Regression for Surface Estimation
===================================================

This module trains a GP per scan to model the karst surface as:

    f(s, ψ^v) → ρ^v

The GP learns the **residual** between the actual measured range ρ^v and the
range predicted by the elliptic cylinder prior f_p.  After training, the GP
can predict the expected range (with uncertainty) at any query point (s*, ψ*).

Kernel (Equation 14 of the paper):
    K([s, ψ], [s', ψ']) = K_s(s, s') · K_ψ(ψ, ψ')

- K_s : Matérn-5/2 on the curvilinear abscissa (Euclidean distance)
- K_ψ : Matérn-5/2 on the sonar rotation angle (chordal distance)

The chordal distance is handled by embedding ψ as [cos(ψ), sin(ψ)] and
using a standard Euclidean kernel on the 2D embedding, which is equivalent
to using chordal distance:  d_chord(ψ, ψ') = 2|sin((ψ-ψ')/2)|
    = ||[cos ψ, sin ψ] - [cos ψ', sin ψ']||₂

References
----------
- Paper Section 4.1 (Environment Model)
- Paper Section 4.2.2 (Gaussian Process Regression and Kernel Choices)
- Paper Equation (14)
"""

import numpy as np
import h5py
import GPy


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PRIOR FUNCTION  f_p(s, ψ)
# ══════════════════════════════════════════════════════════════════════════════

def compute_prior_range(s_query, psi_query, T, a, b, H_scan, s_scan, pose_idx_min):
    """
    Compute the prior range f_p(s, ψ) for one or more query points.

    For a given position along the trajectory (identified by curvilinear
    abscissa s) and a vertical sonar rotation angle ψ, this function returns
    the distance from the robot (in the vertical sonar plane) to the fitted
    elliptic cylinder.

    Parameters
    ----------
    s_query     : (n,) curvilinear abscissa values for each query point
    psi_query   : (n,) sonar rotation angles (radians) for each query point
    T           : (4, 4) SE(3) pose of the elliptic cylinder (world frame)
    a           : float, semi-major axis of the elliptic cross-section
    b           : float, semi-minor axis of the elliptic cross-section
    H_scan      : (m, 7) local trajectory poses [x, y, z, qw, qx, qy, qz]
    s_scan      : (m,) curvilinear abscissa for each pose in H_scan
    pose_idx_min: int, not used here but kept for API consistency

    Returns
    -------
    rho_prior : (n,) predicted ranges from the prior elliptic cylinder
    """
    from scipy.spatial.transform import Rotation

    n = len(s_query)
    rho_prior = np.zeros(n)

    # Inverse of the cylinder pose — to transform points into cylinder frame
    T_inv = np.linalg.inv(T)

    for i in range(n):
        # 1. Find the closest trajectory pose for this s value
        idx = np.argmin(np.abs(s_scan - s_query[i]))
        pose = H_scan[idx]

        robot_pos  = pose[:3]
        qw, qx, qy, qz = pose[3:]
        R_robot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

        # 2. The vertical sonar emits a ray at angle ψ in its local frame.
        #    Direction in robot body frame (assuming vertical sonar aligned
        #    such that ψ rotates in the sonar's XY plane):
        #    We use the same convention as compute_point_cloud:
        #    The sonar measurement direction in body frame after frame
        #    alignment rotations corresponds to a direction parameterized by ψ.
        psi = psi_query[i]
        # Ray direction in robot body frame (from the sonar rotation convention)
        # Based on gv mapping: [r·cos(θ)·cos(ψ), r·cos(θ)·sin(ψ), r·sin(θ)]
        # For vertical sonar, θ≈0, so direction ≈ [cos(ψ), sin(ψ), 0]
        # But the frame alignment rotations in your code transform this.
        # We'll compute the direction from the quaternion chain used in
        # compute_point_cloud, with unit distance.
        from math import cos, sin, pi

        theta2rad = psi / 2
        q_body = np.array([cos(theta2rad), -sin(theta2rad), 0, 0])

        # Frame alignment rotations (same as your compute_point_cloud)
        rot_z = np.array([cos(pi/2), 0, 0, sin(pi/2)])
        q_body = _quat_multiply(q_body, rot_z)

        rot_y = np.array([cos(pi/2), 0, sin(pi/2), 0])
        q_body = _quat_multiply(q_body, rot_y)

        # Combined quaternion: robot orientation × body frame rotation
        q_robot = np.array([qw, qx, qy, qz])
        q_combined = _quat_multiply(q_robot, q_body)

        # Direction = rotate unit vector [0, 0, 0, 1] (as pure quaternion)
        unit_dist = np.array([0.0, 0.0, 0.0, 1.0])
        q_conj = _quat_conjugate(q_combined)
        rotated = _quat_multiply(q_combined, _quat_multiply(unit_dist, q_conj))
        ray_dir = rotated[1:4]  # world-frame direction (unit length)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # 3. Find intersection of this ray with the elliptic cylinder.
        #    Transform ray origin and direction into cylinder local frame.
        origin_local = T_inv[:3, :3] @ robot_pos + T_inv[:3, 3]
        dir_local    = T_inv[:3, :3] @ ray_dir

        # In cylinder local frame: X-axis = major (a), Y-axis = minor (b),
        # Z-axis = cylinder axis.
        # Elliptic cross-section equation: (x/a)² + (y/b)² = 1
        # Ray: P(t) = origin_local + t * dir_local
        # Substitute: ((ox + t·dx)/a)² + ((oy + t·dy)/b)² = 1

        ox, oy = origin_local[0], origin_local[1]
        dx, dy = dir_local[0], dir_local[1]

        A_coeff = (dx/a)**2 + (dy/b)**2
        B_coeff = 2 * (ox*dx/a**2 + oy*dy/b**2)
        C_coeff = (ox/a)**2 + (oy/b)**2 - 1.0

        discriminant = B_coeff**2 - 4*A_coeff*C_coeff

        if A_coeff < 1e-12 or discriminant < 0:
            # Ray is parallel to cylinder axis or doesn't intersect
            # Use a fallback: Euclidean distance to cylinder center in XY
            rho_prior[i] = np.sqrt((a * np.cos(psi))**2 + (b * np.sin(psi))**2)
        else:
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-B_coeff + sqrt_disc) / (2 * A_coeff)
            t2 = (-B_coeff - sqrt_disc) / (2 * A_coeff)

            # Pick the nearest positive intersection
            candidates = [t for t in [t1, t2] if t > 0]
            if candidates:
                rho_prior[i] = min(candidates)
            else:
                # Both behind the robot — use absolute minimum
                rho_prior[i] = min(abs(t1), abs(t2))

    return rho_prior


def _quat_multiply(q1, q2):
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conjugate(q):
    """Conjugate of quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TRAIN GP FOR A SINGLE SCAN
# ══════════════════════════════════════════════════════════════════════════════

def train_gp_for_scan(s_scan, M_v_rho, M_v_psi, M_v_pose_idx,
                      T, a, b, H_scan, pose_idx_min,
                      optimize=True, verbose=False):
    """
    Train a Gaussian Process to model the karst surface for one scan.

    Implements Algorithm 1 Line 3:  f ← GP(s, M^v, f_p)

    The GP models the residual:  y = ρ^v - f_p(s, ψ^v)

    Parameters
    ----------
    s_scan       : (m,) curvilinear abscissa for each trajectory pose
    M_v_rho      : (n,) measured ranges from the vertical sonar
    M_v_psi      : (n,) sonar rotation angles (radians)
    M_v_pose_idx : (n,) global pose indices for each measurement
    T            : (4,4) SE(3) pose of the fitted elliptic cylinder
    a            : float, semi-major axis
    b            : float, semi-minor axis
    H_scan       : (m, 7) trajectory poses during this scan
    pose_idx_min : int, offset to convert global → local pose index
    optimize     : bool, whether to optimize hyperparameters via MLE
    verbose      : bool, print progress info

    Returns
    -------
    dict with keys:
        'model'     : the trained GPy model
        'X_train'   : (n, 3) training inputs [s, cos(ψ), sin(ψ)]
        'Y_train'   : (n, 1) training outputs (residuals)
        'rho_prior' : (n,) prior range values used
        'kernel'    : the GPy kernel object
    """
    n = len(M_v_rho)
    if n < 3:
        if verbose:
            print(f" ⚠ Skipping: only {n} measurements (need ≥ 3)")
        return None

    # ── Step 1: Map each measurement to its curvilinear abscissa ──────
    local_idx      = (M_v_pose_idx - pose_idx_min).astype(int)
    local_idx      = np.clip(local_idx, 0, len(s_scan) - 1)
    s_measurements = s_scan[local_idx]

    # ── Step 2: Compute prior range for each measurement ─────────────
    rho_prior = compute_prior_range(
        s_query=s_measurements,
        psi_query=M_v_psi,
        T=T, a=a, b=b,
        H_scan=H_scan, s_scan=s_scan,
        pose_idx_min=pose_idx_min
    )

    # ── Step 3: Compute residuals (GP targets) ───────────────────────
    residuals = M_v_rho - rho_prior

    # ── Step 4: Build training inputs ────────────────────────────────
    # Embed ψ as [cos(ψ), sin(ψ)] to handle the circular topology
    # via chordal distance (Equation 15 of the paper)
    X_train = np.column_stack([
        s_measurements,      # column 0: curvilinear abscissa
        np.cos(M_v_psi),     # column 1: cos(ψ)
        np.sin(M_v_psi),     # column 2: sin(ψ)
    ])

    Y_train = residuals.reshape(-1, 1)

    # ── Step 5: Build the product kernel (Equation 14) ───────────────
    # K_s: Matérn-5/2 on the curvilinear abscissa (dim 0)
    kern_s = GPy.kern.Matern52(
        input_dim=1,
        active_dims=[0],
        name='K_s'
    )

    # K_ψ: Matérn-5/2 on the angle embedding [cos(ψ), sin(ψ)] (dims 1,2)
    # Using Euclidean distance on the 2D circle embedding is equivalent
    # to chordal distance on the angle (Equation 15)
    kern_psi = GPy.kern.Matern52(
        input_dim=2,
        active_dims=[1, 2],
        name='K_psi'
    )

    # Product kernel (Equation 14)
    kernel = kern_s * kern_psi

    # ── Step 6: Create and train the GP model ────────────────────────
    model = GPy.models.GPRegression(X_train, Y_train, kernel)

    # Set reasonable initial noise variance
    model.Gaussian_noise.variance = 0.01

    if optimize:
        try:
            model.optimize(messages=verbose, max_iters=200)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Optimization warning: {e}")
            # Model still usable with initial hyperparameters

    if verbose:
        print(f"  GP trained: {n} points")
        print(f"    K_s  lengthscale = {kern_s.lengthscale.values[0]:.4f}, "
              f"variance = {kern_s.variance.values[0]:.4f}")
        print(f"    K_ψ  lengthscale = {kern_psi.lengthscale.values[0]:.4f}, "
              f"variance = {kern_psi.variance.values[0]:.4f}")
        print(f"    Noise variance  = {model.Gaussian_noise.variance.values[0]:.6f}")

    return {
        'model':     model,
        'X_train':   X_train,
        'Y_train':   Y_train,
        'rho_prior': rho_prior,
        'kernel':    kernel,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PREDICT WITH TRAINED GP
# ══════════════════════════════════════════════════════════════════════════════

def predict_surface(gp_result, s_query, psi_query,
                    T, a, b, H_scan, s_scan, pose_idx_min):
    """
    Predict surface range at query points using the trained GP.

    Returns the full predictive distribution:
        ρ^v* ~ N(ρ̄*, σ²*)
    where ρ̄* = f_p(s*, ψ*) + GP_mean(s*, ψ*)

    This is exactly what is needed for Equation (35)/(36) in the paper
    to evaluate the likelihood of candidate elevation angles.

    Parameters
    ----------
    gp_result   : dict returned by train_gp_for_scan()
    s_query     : (n,) curvilinear abscissa for query points
    psi_query   : (n,) sonar rotation angles (radians) for query points
    T, a, b     : elliptic cylinder parameters
    H_scan      : (m, 7) trajectory poses
    s_scan      : (m,) curvilinear abscissa for trajectory
    pose_idx_min: int

    Returns
    -------
    rho_mean : (n,) predicted mean range  ρ̄*  (prior + GP mean)
    rho_var  : (n,) predicted variance    σ²*  (from GP posterior)
    """
    model = gp_result['model']

    # Build query inputs with the same encoding as training
    X_query = np.column_stack([
        s_query,
        np.cos(psi_query),
        np.sin(psi_query),
    ])

    # GP prediction (returns residual mean and variance)
    residual_mean, residual_var = model.predict(X_query)

    # Compute prior range at query points
    rho_prior = compute_prior_range(
        s_query=s_query,
        psi_query=psi_query,
        T=T, a=a, b=b,
        H_scan=H_scan, s_scan=s_scan,
        pose_idx_min=pose_idx_min
    )

    # Full prediction: prior + learned residual
    rho_mean = rho_prior + residual_mean.flatten()
    rho_var  = residual_var.flatten()

    return rho_mean, rho_var


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BATCH PROCESSING:  TRAIN GP FOR ALL SCANS
# ══════════════════════════════════════════════════════════════════════════════

def train_gp_all_scans(h5_path='HDF5_DATA/7_SCANS.h5',
                       prior_key='INITIAL',
                       optimize=True,
                       verbose=True):
    """
    Train a GP for every scan in the HDF5 file.

    Parameters
    ----------
    h5_path   : str, path to the HDF5 file containing scan data
    prior_key : str, 'INITIAL' or 'OPTIMIZED' — which cylinder prior to use
    optimize  : bool, whether to optimize GP hyperparameters
    verbose   : bool, print progress

    Returns
    -------
    gp_models : dict, scan_number → gp_result (as returned by train_gp_for_scan)
    """
    gp_models = {}

    with h5py.File(h5_path, 'r') as f:
        scans_grp = f['SCANS']
        scan_keys = sorted(scans_grp.keys())

        if verbose:
            print(f"Training GPs for {len(scan_keys)} scans...")
            print(f"Using '{prior_key}' cylinder prior\n")

        for i, sk in enumerate(scan_keys):
            scan = scans_grp[sk]
            scan_num = int(sk)

            # ── Load vertical sonar data ──────────────────────────
            M_v_rho      = scan['M_v_rho'][:]
            M_v_psi      = scan['M_v_psi'][:]
            M_v_pose_idx = scan['M_v_pose_idx'][:].astype(int)

            # ── Load trajectory and curvilinear abscissa ──────────
            H_scan       = scan['H'][:]
            s_scan       = scan['s'][:]
            pose_idx_min = int(scan['pose_idx_min'][()])

            # ── Load surface prior (elliptic cylinder) ────────────
            prior_path = f'SURFACE_PRIOR/{prior_key}'
            if prior_path not in scan:
                if verbose and (i + 1) % 50 == 0:
                    print(f"  Scan {sk}: no {prior_key} prior, skipping")
                continue

            T = scan[f'{prior_path}/T'][:]
            a = float(scan[f'{prior_path}/a'][()])
            b = float(scan[f'{prior_path}/b'][()])

            # ── Train the GP ──────────────────────────────────────
            if verbose:
                print(f"  Scan {sk} ({len(M_v_rho)} measurements)...", end=" ")

            result = train_gp_for_scan(
                s_scan=s_scan,
                M_v_rho=M_v_rho,
                M_v_psi=M_v_psi,
                M_v_pose_idx=M_v_pose_idx,
                T=T, a=a, b=b,
                H_scan=H_scan,
                pose_idx_min=pose_idx_min,
                optimize=optimize,
                verbose=False,
            )

            if result is not None:
                gp_models[scan_num] = result
                if verbose:
                    m = result['model']
                    ks = m.kern.K_s
                    kp = m.kern.K_psi
                    print(f"✓  ls_s={ks.lengthscale.values[0]:.3f}, "
                          f"ls_ψ={kp.lengthscale.values[0]:.3f}, "
                          f"noise={m.Gaussian_noise.variance.values[0]:.5f}")
            else:
                if verbose:
                    print("⚠ skipped (too few points)")

    if verbose:
        print(f"\n✓ Trained {len(gp_models)} GP models out of {len(scan_keys)} scans")

    return gp_models


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SAVE / LOAD GP HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

def save_gp_params(gp_models, h5_path='HDF5_DATA/7_SCANS.h5'):
    """
    Save trained GP hyperparameters to the HDF5 file.

    The full GP model cannot be serialized to HDF5, but we save the
    learned hyperparameters so the GP can be reconstructed later.
    """
    with h5py.File(h5_path, 'a') as f:
        saved = 0
        for scan_num, result in gp_models.items():
            sk = f"{scan_num:04d}"
            if f"SCANS/{sk}" not in f:
                continue

            grp = f[f"SCANS/{sk}"]

            # Remove old GP data if re-running
            if 'GP_SURFACE' in grp:
                del grp['GP_SURFACE']

            gp_grp = grp.create_group('GP_SURFACE')

            model = result['model']

            # Save kernel hyperparameters
            gp_grp.create_dataset('param_array', data=model.param_array)
            gp_grp.create_dataset('param_names',
                                  data=np.array([str(n) for n in model.parameter_names()],
                                                dtype='S100'))

            # Save training data (needed to reconstruct the GP)
            gp_grp.create_dataset('X_train', data=result['X_train'],
                                  compression='gzip')
            gp_grp.create_dataset('Y_train', data=result['Y_train'],
                                  compression='gzip')
            gp_grp.create_dataset('rho_prior', data=result['rho_prior'],
                                  compression='gzip')
            saved += 1

    print(f"✓ Saved GP parameters for {saved} scans to {h5_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CONVENIENCE:  TRAIN ONE SCAN (for interactive use / debugging)
# ══════════════════════════════════════════════════════════════════════════════

def train_gp_single_scan(scan_num, h5_path='HDF5_DATA/7_SCANS.h5',
                         prior_key='INITIAL', optimize=True):
    """
    Train a GP for a single scan — convenience wrapper for debugging.

    Parameters
    ----------
    scan_num  : int, the scan number to process
    h5_path   : str, path to HDF5 file
    prior_key : str, 'INITIAL' or 'OPTIMIZED'
    optimize  : bool

    Returns
    -------
    gp_result : dict with 'model', 'X_train', 'Y_train', 'rho_prior', 'kernel'
    scan_data : dict with the raw scan data for further use
    """
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

    gp_result = train_gp_for_scan(
        s_scan=s_scan,
        M_v_rho=M_v_rho, M_v_psi=M_v_psi,
        M_v_pose_idx=M_v_pose_idx,
        T=T, a=a, b=b,
        H_scan=H_scan,
        pose_idx_min=pose_idx_min,
        optimize=optimize,
        verbose=True,
    )

    return gp_result, scan_data


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Example usage
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    h5_path = sys.argv[1] if len(sys.argv) > 1 else 'HDF5_DATA/7_SCANS.h5'

    # Train a single scan for testing
    print("=" * 60)
    print("Training GP for a single scan (scan 57)...")
    print("=" * 60)
    gp_result, scan_data = train_gp_single_scan(57, h5_path=h5_path)

    if gp_result is not None:
        print("\nModel summary:")
        print(gp_result['model'])

        # Quick self-prediction test
        rho_mean, rho_var = predict_surface(
            gp_result,
            s_query=scan_data['s'][:5],
            psi_query=scan_data['M_v_psi'][:5],
            T=scan_data['T'], a=scan_data['a'], b=scan_data['b'],
            H_scan=scan_data['H'], s_scan=scan_data['s'],
            pose_idx_min=scan_data['pose_idx_min'],
        )
        print(f"\nPrediction test (first 5 points):")
        print(f"  Predicted mean:  {rho_mean}")
        print(f"  Predicted std:   {np.sqrt(rho_var)}")
        print(f"  Actual measured: {scan_data['M_v_rho'][:5]}")
