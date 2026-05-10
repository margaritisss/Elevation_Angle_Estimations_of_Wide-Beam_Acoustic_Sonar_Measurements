"""
=============================================================================
Algorithm 2 (Breux & Lapierre 2020) — Elevation Angle MLE
LINES 1–5 ONLY  (preparation and per-θ_q geometry)
=============================================================================

This module implements ONLY the first five lines of Algorithm 2:

    1: Θ̂ ← ∅,   Δ_θ ← (θ_max − θ_min) / n_q
    2: for q do
    3:     θ_q ← θ_{q-1} + Δ_θ
    4:     x ← g^h([ρ_k^h, θ_q, ψ_k^h], k)             ▷ Equation (20)
    5:     I ← ∅

It is structured so that lines 6–33 can be appended later without rewriting
anything that's here.

INPUTS expected (mirrors the paper's notation):

    f               trained GP surface model        (gp_result from gp_surface.py)
    H_scan          (m, 7) trajectory of this scan  [x, y, z, qw, qx, qy, qz]
    s_scan          (m,)   curvilinear abscissa for each pose in H_scan
    M_h             (3, n_h)  rows = [ρ_h, θ_h (NaN), ψ_h]
    pose_idx_h      (n_h,)    σ^h(k) — *global* pose index for each ping
    pose_idx_min    int       so that local_idx = pose_idx_h - pose_idx_min
    p_H             (7,)      pose of horizontal sonar in robot body frame
                              [x, y, z, qw, qx, qy, qz]   (extrinsic calibration)
    beam_width      float     vertical aperture of the horizontal sonar (rad)
    n_q             int       number of θ samples (paper uses 200)
    k               int       which horizontal ping to process (0-indexed)

OUTPUTS (per ping k):

    A list of length n_q.  Element q is a dict with:
        'theta_q' : float          — the candidate elevation angle (rad)
        'x'       : (3,) np.array  — candidate 3D world point
        'I'       : list           — empty (filled in lines 6–11)

Notes on g^h (Equation 1 of the paper):
---------------------------------------
The paper defines:
    g^h([ρ, θ, ψ], k) =  η_{σ^h(k)}  ⊕  p^H  ⊕  [ρ cos θ cos ψ,
                                                  ρ cos θ sin ψ,
                                                  ρ sin θ]

Here ⊕ is the SE(3) pose-point composition.  Concretely:
    1) build local point in {S^H}:    p_local = [ρcθcψ, ρcθsψ, ρsθ]
    2) lift to body frame {B}:        p_body  = R(q^H) p_local + τ^H
    3) lift to world frame {0}:       x       = R(q_robot) p_body + τ_robot

⚠️  IMPORTANT — coordinate convention:
    Your existing `compute_point_cloud` (used for the *vertical* sonar) applies
    two extra alignment rotations (Z by π/2, then Y by π/2) before the final
    quaternion rotation.  Those rotations encode how the sonar is mounted on
    the robot.  For the *horizontal* sonar you need the analogous extrinsic
    calibration p^H.  In this module we assume p^H is supplied as a pose
    [x, y, z, qw, qx, qy, qz] in the body frame, with the convention that the
    sonar's XY plane (where ψ rotates and where θ=0 lies) is already encoded
    in p^H.  If your data uses a different convention, adapt `g_h` below or
    bake the alignment into p^H once and for all.
=============================================================================
"""

import numpy as np
from scipy.spatial.transform import Rotation

# ─────────────────────────────────────────────────────────────────────────────
# SE(3) helpers — match the conventions used in Algorithm_1.ipynb
# (quaternion stored as [qw, qx, qy, qz], scalar-first)
# ─────────────────────────────────────────────────────────────────────────────

def _pose_to_R_t(pose):
    """
    Split a 7-vector pose into a 3×3 rotation matrix and a 3-vector translation.

    pose = [x, y, z, qw, qx, qy, qz]   (scalar-first quaternion)
    """
    t = np.asarray(pose[:3], dtype=float)
    qw, qx, qy, qz = pose[3:]
    # scipy uses scalar-LAST: [qx, qy, qz, qw]
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return R, t


def _compose_pose_point(pose, p_local):
    """
    SE(3) pose ⊕ point  composition:    p_out = R(pose) @ p_local + t(pose)

    This is exactly the ⊕ operator used in Equation (1) of the paper for the
    pose-point case.
    """
    R, t = _pose_to_R_t(pose)
    return R @ p_local + t


# ─────────────────────────────────────────────────────────────────────────────
# Line 4 — Equation (20):  x ← g^h([ρ_k^h, θ_q, ψ_k^h], k)
# ─────────────────────────────────────────────────────────────────────────────

def g_h(rho, theta, psi, k, H_scan, p_H, pose_idx_h, pose_idx_min):
    """
    Map a horizontal-sonar spherical measurement to a 3D world point.

    Implements Equation (1) / Equation (20):

        x = η_{σ^h(k)}  ⊕  p^H  ⊕  [ρ cos θ cos ψ,
                                     ρ cos θ sin ψ,
                                     ρ sin θ]

    Parameters
    ----------
    rho           : float  — measured range of ping k  (ρ_k^h)
    theta         : float  — HYPOTHESIZED elevation angle θ_q  (rad)
    psi           : float  — measured azimuth of ping k  (ψ_k^h, rad)
    k             : int    — index of the horizontal ping
    H_scan        : (m, 7) — trajectory poses for this scan (local, 0-indexed)
    p_H           : (7,)   — sonar→body extrinsic [x,y,z,qw,qx,qy,qz]
    pose_idx_h    : (n_h,) — *global* pose index per ping (σ^h(k))
    pose_idx_min  : int    — global→local pose offset for this scan

    Returns
    -------
    x : (3,) np.array — candidate 3D world-frame point
    """
    # --- Step 1: Build the spherical-coordinate point in the sonar frame {S^H}
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(psi),   np.sin(psi)
    p_local = rho * np.array([cos_t * cos_p,
                              cos_t * sin_p,
                              sin_t])

    # --- Step 2:  p^H ⊕ p_local      (sonar frame → body frame)
    p_body = _compose_pose_point(p_H, p_local)

    # --- Step 3:  η_{σ^h(k)} ⊕ p_body      (body frame → world frame)
    local_idx = int(pose_idx_h[k]) - int(pose_idx_min)
    local_idx = max(0, min(local_idx, H_scan.shape[0] - 1))   # safety clamp
    eta_k = H_scan[local_idx]
    x = _compose_pose_point(eta_k, p_body)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2 — Lines 1 through 5 (per single horizontal ping k)
# ─────────────────────────────────────────────────────────────────────────────

def run_lines_1_to_5(M_h, pose_idx_h, pose_idx_min, H_scan,
                     p_H, k, beam_width, n_q):
    """
    Execute lines 1–5 of Algorithm 2 for ONE horizontal sonar measurement.

    Algorithm 2 lines:
        1: Θ̂ ← ∅,   Δ_θ ← (θ_max − θ_min) / n_q
        2: for q do
        3:     θ_q ← θ_{q-1} + Δ_θ
        4:     x ← g^h([ρ_k^h, θ_q, ψ_k^h], k)
        5:     I ← ∅

    Parameters
    ----------
    M_h           : (3, n_h)   horizontal sonar matrix.  Rows: [ρ, θ(NaN), ψ]
    pose_idx_h    : (n_h,)     σ^h — global pose index of each ping
    pose_idx_min  : int        global→local pose offset for this scan
    H_scan        : (m, 7)     trajectory poses during this scan
    p_H           : (7,)       horizontal sonar pose in body frame
    k             : int        which ping to process (0-indexed)
    beam_width    : float      vertical aperture of the h-sonar (rad).
                               θ_max = +b/2,  θ_min = −b/2
    n_q           : int        number of θ samples

    Returns
    -------
    state : dict with the following keys
        'Theta_hat' : list    — empty set Θ̂ (line 1, will be filled in line 31)
        'Delta_theta': float  — the angular step Δ_θ (line 1)
        'theta_min' : float
        'theta_max' : float
        'rho_k'     : float   — the measured range ρ_k^h
        'psi_k'     : float   — the measured azimuth ψ_k^h
        'samples'   : list of length n_q.  Each element is a dict:
            {
              'q'      : int,
              'theta_q': float,         # line 3
              'x'      : np.array(3,),  # line 4
              'I'      : list,          # line 5  (empty for now)
            }
    """
    # ────────────────────────────────────────────────────────────────────
    # Read the k-th horizontal ping
    # ────────────────────────────────────────────────────────────────────
    rho_k = float(M_h[0, k])     # range
    # M_h[1, k] is θ — currently NaN, this is what we are estimating
    psi_k = float(M_h[2, k])     # azimuth

    # ────────────────────────────────────────────────────────────────────
    # LINE 1:  Θ̂ ← ∅,   Δ_θ ← (θ_max − θ_min) / n_q
    # ────────────────────────────────────────────────────────────────────
    Theta_hat   = []                    # empty set, to be filled later (line 31)
    theta_max   =  beam_width / 2.0     # +b/2
    theta_min   = -beam_width / 2.0     # −b/2
    Delta_theta = (theta_max - theta_min) / n_q

    # ────────────────────────────────────────────────────────────────────
    # LINE 2: for q do
    # ────────────────────────────────────────────────────────────────────
    samples = []

    # The recurrence in line 3 is θ_q = θ_{q-1} + Δ_θ.  With Θ̂ initialized
    # at line 1 and the loop starting at q=1, the first value is therefore
    # θ_1 = θ_min + Δ_θ.  We follow that exactly.
    theta_prev = theta_min

    for q in range(1, n_q + 1):
        # ────────────────────────────────────────────────────────────────
        # LINE 3:  θ_q ← θ_{q-1} + Δ_θ
        # ────────────────────────────────────────────────────────────────
        theta_q   = theta_prev + Delta_theta
        theta_prev = theta_q

        # ────────────────────────────────────────────────────────────────
        # LINE 4:  x ← g^h([ρ_k^h, θ_q, ψ_k^h], k)        ▷ Equation (20)
        # ────────────────────────────────────────────────────────────────
        x = g_h(rho=rho_k, theta=theta_q, psi=psi_k, k=k,
                H_scan=H_scan, p_H=p_H,
                pose_idx_h=pose_idx_h, pose_idx_min=pose_idx_min)

        # ────────────────────────────────────────────────────────────────
        # LINE 5:  I ← ∅
        #   (set of trajectory segment indices where x falls in the
        #    vertical sonar plane — to be filled by lines 6–11)
        # ────────────────────────────────────────────────────────────────
        I = []

        samples.append({
            'q'       : q,
            'theta_q' : theta_q,
            'x'       : x,
            'I'       : I,
        })

    return {
        'Theta_hat'   : Theta_hat,
        'Delta_theta' : Delta_theta,
        'theta_min'   : theta_min,
        'theta_max'   : theta_max,
        'rho_k'       : rho_k,
        'psi_k'       : psi_k,
        'samples'     : samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — load a scan from HDF5 and run lines 1–5 for one ping
# ─────────────────────────────────────────────────────────────────────────────

def run_lines_1_to_5_from_h5(scan_num, k, p_H,
                             h5_path    = 'HDF5_DATA/7_SCANS.h5',
                             beam_width = np.radians(35.0),
                             n_q        = 200):
    """
    Convenience wrapper: pull a scan straight from the HDF5 file and run
    lines 1–5 for the k-th horizontal ping.

    Parameters
    ----------
    scan_num   : int    — scan number (e.g. 57)
    k          : int    — index of the horizontal ping
    p_H        : (7,)   — horizontal-sonar extrinsic in body frame
                          [x, y, z, qw, qx, qy, qz]
    h5_path    : str
    beam_width : float  — radians (default 35°)
    n_q        : int

    Returns
    -------
    state      : dict (same as run_lines_1_to_5)
    scan_data  : dict with the raw scan arrays (handy for downstream lines)
    """
    import h5py

    sk = f"{scan_num:04d}"
    with h5py.File(h5_path, 'r') as f:
        scan = f[f'SCANS/{sk}']

        H_scan       = scan['H'][:]
        s_scan       = scan['s'][:]
        pose_idx_min = int(scan['pose_idx_min'][()])
        M_h          = scan['M_h_M'][:]                    # shape (3, n_h)
        pose_idx_h   = scan['M_h_pose_idx'][:].astype(int) # shape (n_h,)

    scan_data = {
        'H_scan'      : H_scan,
        's_scan'      : s_scan,
        'pose_idx_min': pose_idx_min,
        'M_h'         : M_h,
        'pose_idx_h'  : pose_idx_h,
    }

    state = run_lines_1_to_5(
        M_h=M_h, pose_idx_h=pose_idx_h, pose_idx_min=pose_idx_min,
        H_scan=H_scan, p_H=p_H, k=k,
        beam_width=beam_width, n_q=n_q,
    )

    return state, scan_data


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Synthetic sanity check: identity p^H, identity robot pose, ρ=1, ψ=0
    # → for θ=0 we should get x ≈ [1, 0, 0]
    H_scan     = np.array([[0, 0, 0, 1, 0, 0, 0]])     # one identity pose
    p_H        = np.array([0, 0, 0, 1, 0, 0, 0])       # identity extrinsic
    pose_idx_h = np.array([0])                         # ping 0 → pose 0
    M_h        = np.array([[1.0],     # ρ
                           [np.nan],  # θ unknown
                           [0.0]])    # ψ

    state = run_lines_1_to_5(
        M_h=M_h, pose_idx_h=pose_idx_h, pose_idx_min=0,
        H_scan=H_scan, p_H=p_H, k=0,
        beam_width=np.radians(35.0), n_q=5,
    )
    print(f"Δ_θ        = {state['Delta_theta']:.4f}  rad")
    print(f"θ samples  = {[round(s['theta_q'], 4) for s in state['samples']]}")
    print(f"x at θ_q≈0 = {state['samples'][len(state['samples']) // 2]['x']}")
    # Expect the middle sample's x to be near [1, 0, 0]