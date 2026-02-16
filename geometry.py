# -*- coding: utf-8 -*-
"""
Geometric utilities, IRLS refinement, and KDE consensus.
This module implements the geometric core of the Semi-Calibrated T-Linkage pipeline,
specifically Step 1 (Refinement), Step 2 (Focal Estimation), and Step 3 (KDE) 
as described in Section 4.1 of the paper.
"""
import numpy as np
from numba import njit

try:
    import poselib
except Exception as e:
    print(f"[WARNING] poselib cannot be imported, some geometric optimizations will be skipped: {e}")
    poselib = None

# ============================================================================
# GEOMETRIC UTILITIES & ROBUST ESTIMATION (Refers to Paper Section 4.1, Step 1)
# ============================================================================
@njit(cache=True, fastmath=True)
def _sampson_sqrt_res_numba(F, p1, p2):
    """
    Computes the square root of the Sampson distance for a given Fundamental matrix.
    Refers to the geometric error term d_S(x, x', F) in Equation (3) of the paper.
    """
    N = p1.shape[0]
    res = np.empty(N, dtype=np.float64)
    f00, f01, f02 = F[0,0], F[0,1], F[0,2]
    f10, f11, f12 = F[1,0], F[1,1], F[1,2]
    f20, f21, f22 = F[2,0], F[2,1], F[2,2]
    for i in range(N):
        x1, y1 = p1[i,0], p1[i,1]
        x2, y2 = p2[i,0], p2[i,1]
        Fx1_0 = f00*x1 + f01*y1 + f02
        Fx1_1 = f10*x1 + f11*y1 + f12
        Fx1_2 = f20*x1 + f21*y1 + f22
        Ftx2_0 = f00*x2 + f10*y2 + f20
        Ftx2_1 = f01*x2 + f11*y2 + f21
        num = x2*Fx1_0 + y2*Fx1_1 + Fx1_2
        den = Fx1_0**2 + Fx1_1**2 + Ftx2_0**2 + Ftx2_1**2 + 1e-15
        res[i] = num / np.sqrt(den)
    return res

@njit(cache=True)
def _irls_tukey_weights_numba(r, c=4.685):
    """
    Calculates weights using Tukey's biweight loss function.
    Refers to the robust loss function rho(.) used in Equation (3) of the paper 
    during the non-linear refinement phase.
    """
    r_abs = np.abs(r)
    med_r = np.median(r)
    mad_diff = np.abs(r - med_r)
    mad = np.median(mad_diff) * 1.4826 + 1e-12
    w = np.empty_like(r)
    for i in range(len(r)):
        val = np.abs(r[i]) / max(1e-12, mad)
        if val >= c: w[i] = 0.0
        else: w[i] = (1.0 - (val/c)**2)**2
    return w

def _project_rank2(F):
    """
    Enforces the rank-2 constraint on the Fundamental matrix.
    Refers to the det(F) = 0 condition in Equation (1) of the paper.
    """
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0
    return U @ np.diag(S) @ Vt

def _weighted_eight_point(x1, x2, w):
    """
    Weighted 8-point algorithm for Fundamental Matrix estimation.
    Used within the IRLS loop (Algorithm 1, Line 4: Refine).
    """
    X = np.stack([
        x2[:,0]*x1[:,0], x2[:,0]*x1[:,1], x2[:,0],
        x2[:,1]*x1[:,0], x2[:,1]*x1[:,1], x2[:,1],
        x1[:,0],         x1[:,1],         np.ones(len(x1))
    ], axis=1)
    W_diag = np.diag(w)
    XtWX = X.T @ W_diag @ X
    _, _, Vt = np.linalg.svd(XtWX)
    f = Vt[-1]
    F = f.reshape(3,3)
    return _project_rank2(F)

def eight_point_unweighted(p1c8, p2c8):
    """
    Standard unweighted 8-point algorithm on Minimal Sample Sets (MSS).
    Refers to Algorithm 1, Line 3: F <- EightPoint(s_k).
    """
    w = np.ones(p1c8.shape[0], float)
    return _weighted_eight_point(p1c8, p2c8, w)

def refine_F_nonlinear(p1c, p2c, F_init, inlier_mask, thr_px, alpha=1.0, max_outer=5, min_delta=0.01):
    """
    Iteratively Reweighted Least Squares (IRLS) optimization to minimize the 
    Sampson distance against the detected inlier set.
    Refers to Section 4.1 Step 1 and Algorithm 1, Line 4: F <- Refine(F, IRLS-Sampson, tau).
    Minimizes Equation (3).
    """
    F = _project_rank2(F_init.astype(float))
    inl = inlier_mask.copy()
    if inl.sum() < 8: return F, inl

    # Optional internal geometric refinement using PoseLib if available
    if poselib is not None and hasattr(poselib, "refine_fundamental"):
        try:
            p1_cont = np.ascontiguousarray(p1c[inl])
            p2_cont = np.ascontiguousarray(p2c[inl])
            Fp, _ = poselib.refine_fundamental(p1_cont, p2_cont, F)
            if Fp is not None and np.isfinite(Fp).all():
                F = _project_rank2(Fp)
        except: pass

    prev_inl = inl.copy()
    for _ in range(max_outer):
        r = _sampson_sqrt_res_numba(F, p1c[inl], p2c[inl])
        if r.size == 0: break
        w = _irls_tukey_weights_numba(r)
        F = _weighted_eight_point(p1c[inl], p2c[inl], w)
        r_all = _sampson_sqrt_res_numba(F, p1c, p2c)
        base_tau = (alpha * (thr_px**2))
        mad_val = np.median(np.abs(r_all - np.median(r_all)))
        mad_tau = (1.4826 * mad_val)**2 * 4.0
        tau = np.nanmin([base_tau, mad_tau]) if np.isfinite(mad_tau) and mad_tau>0 else base_tau
        r2 = r_all**2
        new_inl = r2 <= tau
        changed = np.sum(prev_inl ^ new_inl) / max(1, np.sum(prev_inl | new_inl))
        inl, prev_inl = new_inl, new_inl
        if changed < min_delta: break
    return F, inl

def choose_weights(Fc, f_prior1, f_prior2, weights_list, max_iters=120):
    """
    Wrapper around the robust solver by Kocur et al. [2024] to extract focal lengths.
    Refers to Section 4.1 Step 2 (Semi-Calibrated Hypothesis Generation) 
    and Algorithm 1, Line 5: f1, f2 <- EstimateFocal(F).
    Solves the Kruppa equations under the semi-calibrated assumption (Eq. 4).
    """
    if poselib is None: return None
    best = None
    for w in weights_list:
        cam1, cam2, it = poselib.focals_from_fundamental_iterative(
            Fc,
            {'model':'SIMPLE_PINHOLE','width':-1,'height':-1,'params':[float(f_prior1),0.0,0.0]},
            {'model':'SIMPLE_PINHOLE','width':-1,'height':-1,'params':[float(f_prior2),0.0,0.0]},
            max_iters=int(max_iters), weights=np.asarray(w, dtype=float)
        )
        f1 = float(cam1.focal())
        f2 = float(cam2.focal())
        cand = dict(w=w, f1=f1, f2=f2, it=int(it), key=(int(it), abs(f1 - f2)))
        if (best is None) or (cand["key"] < best["key"]): best = cand
    return best

# ============================================================================
# KDE & FOCAL CONSENSUS (Refers to Paper Section 4.1, Step 3)
# ============================================================================
def kde_1d(values, grid, bw=None):
    """
    Kernel Density Estimation (KDE) to find the dominant focal length.
    Refers to Section 4.1 Step 3 and Algorithm 1, Line 11: \hat{p}(f) <- KDE(f_vals).
    Implements Equation (6) using a Gaussian kernel and Scott's Rule for bandwidth.
    """
    vals=np.asarray(values,float)
    vals=vals[np.isfinite(vals)]
    if vals.size==0: return np.zeros_like(grid), np.nan
    
    std=np.std(vals); n=len(vals)
    if bw is None:
        # Bandwidth selection via Scott's Rule (as mentioned in Section 4.1 Step 3)
        bw = 1.06 * std * (n ** (-1/5)) if n>1 and std>0 else 1.0
        bw = max(bw,1e-9)
        
    diff=(grid[:,None] - vals[None,:]) / bw
    pdf=np.mean(np.exp(-0.5*diff**2)/(np.sqrt(2*np.pi)*bw), axis=1)
    return pdf,bw

def find_peaks_1d(y, prominence=0.05, distance=10):
    """
    Finds the peaks in the KDE probability density function to determine f_peak.
    Refers to Algorithm 1, Line 12: f_peak <- argmax \hat{p}(f).
    """
    from scipy.signal import find_peaks, peak_widths
    if np.max(y)<=0: return np.array([]), np.array([])
    idx,props=find_peaks(y,prominence=prominence*np.max(y),distance=distance)
    widths,_,_,_=peak_widths(y,idx,rel_height=0.5)
    return idx,widths