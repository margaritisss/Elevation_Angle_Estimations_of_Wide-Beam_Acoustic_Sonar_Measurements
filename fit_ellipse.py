"""
=============================================================================|
Implements the elliptic cross-section fitting step described in Section 4.2.1|
of Breux & Lapierre (2020), using the Direct Least Squares ellipse fitting   |
method of Fitzgibbon, Pilu & Fisher (1999) — paper [33].                     |

The ONLY public function is:

    fit_elliptic_sections(scan_pca_results)

It takes as input the dict produced by `find_cylinder_axes_on_scans`
(which already contains, per scan, the two non-axis PCA vectors and the
point cloud) and returns, per scan, the fitted elliptic section.

References
-----------------------------------------------------------------------------
  [33] A. Fitzgibbon, M. Pilu, R. B. Fisher,
       "Direct Least Square Fitting of Ellipses,"
       IEEE Trans. PAMI, vol. 21, no. 5, pp. 476–480, May 1999.
=============================================================================
"""

# ---------------------------------------------------------------------------
# fit_elliptic_sections  —  the single public function
# ---------------------------------------------------------------------------

import numpy as np
from scipy.linalg import eig

def fit_elliptic_sections(scan_pca_results):
    """
    Estimate the elliptic cross-section for every scan's point cloud.

    This is the direct Python translation of the paper's pipeline
    (Section 4.2.1 + Fitzgibbon et al. 1999, Fig. 4):

      For each scan:
        1. Centre the 3-D point cloud at the robust RPCA centroid.
        2. Project all points onto the 2-D plane spanned by the two
           non-axis PCA vectors  (major_axis, minor_axis).
        3. Build the N×6 design matrix   D  with rows [u², uv, v², u, v, 1].
        4. Build the 6×6 scatter matrix  S = Dᵀ D.
        5. Build the 6×6 ellipse-specific constraint matrix C
           (encodes  4ac − b² = 1,  Eq. 4 of [33]).
        6. Solve the generalised eigensystem  S·a = λ·C·a   (Eq. 7).
        7. Select the eigenvector whose generalised eigenvalue is the
           ONLY NEGATIVE finite one  — exactly as the 6-line Matlab code
           in [33] does with  find(geval < 0 & ~isinf(geval)).
        8. Extract the ellipse orientation angle θ in the 2-D PCA plane.
        9. Rotate (major_axis, minor_axis) around the cylinder axis by θ
           so that the PCA base aligns with the true ellipse axes
           (last sentence of Section 4.2.1).

    Parameters
    ----------
    scan_pca_results : dict  { scan_number (int) : pca_dict }
        Output of `find_cylinder_axes_on_scans`.  Each pca_dict must contain:
            'point_cloud'  : (N, 3)  3-D world-frame points for this scan
            'center'       : (3,)    robust RPCA centroid
            'major_axis'   : (3,)    first non-cylinder PCA unit vector
            'minor_axis'   : (3,)    second non-cylinder PCA unit vector
            'axis_vector'  : (3,)    cylinder axis unit vector

    Returns
    -------
    results : dict  { scan_number (int) : result_dict }
        Each result_dict contains:
            'a'          : (6,)   algebraic coefficients [a,b,c,d,e,f]
            'center_2d'  : (2,)   ellipse centre in the PCA plane  (u₀, v₀)
            'semi_axes'  : (2,)   (semi_major, semi_minor)  lengths
            'angle_rad'  : float  ellipse orientation angle in the PCA plane
            'new_major'  : (3,)   major-axis 3-D unit vector (ellipse-aligned)
            'new_minor'  : (3,)   minor-axis 3-D unit vector (ellipse-aligned)
            'u'          : (N,)   projected u-coordinates  (diagnostic)
            'v'          : (N,)   projected v-coordinates  (diagnostic)
    """
    results = {}

    for scan_num, pca in scan_pca_results.items():

        point_cloud   = pca['point_cloud']   # (N, 3)
        center        = pca['center']        # (3,)
        major_axis    = pca['major_axis']    # (3,)
        minor_axis    = pca['minor_axis']    # (3,)
        cylinder_axis = pca['axis_vector']   # (3,)

        N = point_cloud.shape[0]
        if N < 6:
            # Cannot fit an ellipse with fewer than 6 points
            continue

        # ------------------------------------------------------------------
        # Step 1 — centre the point cloud at the RPCA centroid
        # ------------------------------------------------------------------
        pts = point_cloud - center           # (N, 3)

        # ------------------------------------------------------------------
        # Step 2 — project onto the 2-D plane spanned by (major, minor)
        #   u_i = (p_i − center) · major_axis
        #   v_i = (p_i − center) · minor_axis
        # ------------------------------------------------------------------
        u = pts @ major_axis                 # (N,)
        v = pts @ minor_axis                 # (N,)

        # ------------------------------------------------------------------
        # Step 3 — design matrix  D  (paper Fig. 4, first line)
        #   Each row:  [u², u·v, v², u, v, 1]
        # ------------------------------------------------------------------
        D = np.column_stack([
            u * u,          # u²
            u * v,          # u·v
            v * v,          # v²
            u,              # u
            v,              # v
            np.ones(N),     # 1
        ])                  # shape (N, 6)

        # ------------------------------------------------------------------
        # Step 4 — scatter matrix  S = Dᵀ D   (paper Fig. 4, third line)
        # ------------------------------------------------------------------
        S = D.T @ D         # (6, 6)

        # ------------------------------------------------------------------
        # Step 5 — constraint matrix  C   (paper Eq. 4 and Fig. 4)
        #
        #   Constraint:  4ac − b² = 1   ⟺   aᵀ C a = 1
        #
        #   With a = [a, b, c, d, e, f]  (indices 0..5):
        #       aᵀ C a = 4·a[0]·a[2] − a[1]²
        #   =>  C[0,2] = C[2,0] = 2,   C[1,1] = −1,   rest = 0.
        #
        #   In the Matlab listing:
        #       C(6,6)=0; C(1,3)=-2; C(2,2)=1; C(3,1)=-2;
        #   Note: Matlab is 1-indexed and the sign convention there
        #   encodes −(4ac − b²) = 1  (i.e. b² − 4ac = 1), which is why
        #   the Matlab code looks for the *negative* eigenvalue.
        #   Here we use the sign from Eq. 4 of the paper (4ac − b² = 1),
        #   which makes the ellipse solution the *positive* eigenvalue in
        #   scipy — BUT, to stay byte-for-byte faithful to the 6-line
        #   Matlab code, we adopt the SAME Matlab sign (b² − 4ac = 1) so
        #   that we can search for the negative eigenvalue just like Matlab.
        # ------------------------------------------------------------------
        C = np.zeros((6, 6))
        C[0, 2] = -2.0    # Matlab: C(1,3) = -2  (1-indexed → 0-indexed)
        C[2, 0] = -2.0    # Matlab: C(3,1) = -2
        C[1, 1] =  1.0    # Matlab: C(2,2) =  1

        # ------------------------------------------------------------------
        # Step 6 — solve  S·a = λ·C·a   (paper Eq. 7 and Matlab Fig. 4)
        #
        #   scipy.linalg.eig(A, B) solves  A·v = λ·B·v,
        #   i.e.  S·a = λ·C·a  — identical to Matlab's eig(S, C).
        # ------------------------------------------------------------------
        eigenvalues, eigenvectors = eig(S, C)

        # ------------------------------------------------------------------
        # Step 7 — select the unique elliptical solution
        #
        #   The 6-line Matlab code (paper Fig. 4):
        #       [NegR, NegC] = find(geval < 0 & ~isinf(geval));
        #       a = gevec(:, NegC);
        #
        #   With the Matlab sign convention (b² − 4ac = 1) adopted above,
        #   the ellipse solution is the ONLY NEGATIVE finite eigenvalue.
        # ------------------------------------------------------------------
        real_mask = np.isreal(eigenvalues) & np.isfinite(eigenvalues)
        ev_real   = eigenvalues[real_mask].real
        eg_real   = eigenvectors[:, real_mask].real

        neg_mask  = ev_real < 0
        if not np.any(neg_mask):
            # Degenerate point set — skip this scan
            continue

        # Among (potentially multiple) negative eigenvalues pick the one
        # with the largest magnitude — this is the unique "most elliptic"
        # solution and mirrors Matlab's behaviour when only one is present.
        neg_indices = np.where(neg_mask)[0]
        best_idx    = neg_indices[np.argmin(ev_real[neg_indices])]
        a           = eg_real[:, best_idx]   # (6,)

        # ------------------------------------------------------------------
        # Step 8 — extract the ellipse orientation angle θ in the PCA plane
        #
        #   Standard conic-to-geometric conversion:
        #       θ = ½ · arctan2(b, a − c)
        #   where a = a[0], b = a[1], c = a[2].
        #
        #   Then decide which axis is the major one using the eigenvalues of
        #   the 2×2 shape matrix  [[a, b/2],[b/2, c]].
        # ------------------------------------------------------------------
        a0, b0, c0, d0, e0, f0 = a

        # centre of the fitted ellipse in the 2-D PCA plane
        denom    = b0**2 - 4.0 * a0 * c0       # < 0 for an ellipse
        if abs(denom) < 1e-14:
            continue                            # numerical degenerate
        u0_ell   = (2.0 * c0 * d0 - b0 * e0) / denom
        v0_ell   = (2.0 * a0 * e0 - b0 * d0) / denom

        # semi-axis lengths
        disc     = np.sqrt(max((a0 - c0)**2 + b0**2, 0.0))
        num      = 2.0 * (a0 * e0**2 + c0 * d0**2 - b0 * d0 * e0
                          + denom * f0)
        val1     = num * ((a0 + c0) + disc)
        val2     = num * ((a0 + c0) - disc)
        if val1 < 0 or val2 < 0:
            continue                            # not a real ellipse
        semi1    = abs(np.sqrt(val1) / denom)
        semi2    = abs(np.sqrt(val2) / denom)
        semi_major = max(semi1, semi2)
        semi_minor = min(semi1, semi2)

        # orientation: θ = ½ arctan2(b, a − c)
        if abs(a0 - c0) < 1e-12:
            theta = np.pi / 4.0 if b0 > 0 else -np.pi / 4.0
        else:
            theta = 0.5 * np.arctan2(b0, a0 - c0)

        # eigenvalues of the 2×2 shape matrix to decide which axis is major
        lam1 = 0.5 * ((a0 + c0) + disc)
        lam2 = 0.5 * ((a0 + c0) - disc)
        if abs(lam1) > abs(lam2):
            theta = theta + np.pi / 2.0

        # wrap into (−π/2, π/2]
        theta = (theta + np.pi / 2.0) % np.pi - np.pi / 2.0

        # ------------------------------------------------------------------
        # Step 9 — rotate PCA basis around the cylinder axis by θ
        #
        #   "The PCA base is then rotated around the cylinder axis to be
        #    aligned with the ellipse axis."  (Section 4.2.1)
        #
        #   Rodrigues' rotation formula:
        #       R(θ, ax)·v = cos θ·v  + sin θ·(ax × v)  + (1−cos θ)(ax·v)·ax
        # ------------------------------------------------------------------
        ax    = cylinder_axis / np.linalg.norm(cylinder_axis)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        def _rodrigues(vec):
            return (cos_t * vec
                    + sin_t * np.cross(ax, vec)
                    + (1.0 - cos_t) * np.dot(ax, vec) * ax)

        new_major = _rodrigues(major_axis)
        new_minor = _rodrigues(minor_axis)
        new_major /= np.linalg.norm(new_major)
        new_minor /= np.linalg.norm(new_minor)

        # ------------------------------------------------------------------
        # Store result for this scan
        # ------------------------------------------------------------------
        results[scan_num] = {
            'a'         : a,
            'center_2d' : np.array([u0_ell, v0_ell]),
            'semi_axes' : np.array([semi_major, semi_minor]),
            'angle_rad' : theta,
            'new_major' : new_major,
            'new_minor' : new_minor,
            'u'         : u,
            'v'         : v,
        }

    return results