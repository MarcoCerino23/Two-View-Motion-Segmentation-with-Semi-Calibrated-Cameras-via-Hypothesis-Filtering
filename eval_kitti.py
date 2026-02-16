# -*- coding: utf-8 -*-
"""
Main evaluation script for the KITTI dataset (Autonomous Driving).
Physically Consistent T-Linkage (Full Continuous Version).
"""

import os, glob
import numpy as np
import cv2
import time
from sklearn.metrics import adjusted_rand_score

# Custom local imports
from semicalibrated_tlinkage import sampson_errors_numba, run_tlinkage_numba_heap
from geometry import refine_F_nonlinear, eight_point_unweighted, choose_weights, kde_1d, find_peaks_1d, poselib
from metrics import hungarian_me, false_negative_rate, save_cluster_plot

if poselib is not None:
    print("[INFO] PoseLib detected: Focal filter ENABLED.")
else:
    print("[WARNING] PoseLib not found! Running without focal physical constraints.")

np.set_printoptions(precision=3, suppress=True)

# ============================================================================
# GENERAL PARAMETERS
# ============================================================================
DATASET_ROOT = "dataset_kitti"
OUTPUT_DIR   = "output_improved_kitti_final"

# Algorithm Parameters for OUTDOOR (KITTI)
NUM_SCENES          = 200   
MSS_SIZE            = 8
NUM_MSS_SAMPLES     = 5000  
RNG_SEED            = 42

# Specific Geometric Tuning for KITTI
BASE_THR_PX         = 3.0   
TAU                 = BASE_THR_PX**2
CUTOFF              = 5.0 * TAU

MAX_REL_F12_DIFF    = 0.15  
FOCAL_PEAK_REL_RANGE = 0.25 

MAX_ITERS_SOLVER    = 100

# Preliminary dataset check
if not os.path.exists(DATASET_ROOT):
    sub = os.path.join(DATASET_ROOT, "dataset_processed")
    if os.path.exists(sub): DATASET_ROOT = sub


# ============================================================================
# PER-SCENE PROCESSING PIPELINE
# ============================================================================
def process_kitti_scene(scene_dir):
    rng = np.random.default_rng(RNG_SEED)
    scene_name = os.path.basename(scene_dir)
    print(f"\nProcessing {scene_name} ...")

    # 1. Load Features (.npz)
    npz_path = os.path.join(scene_dir, "features_and_ground_truth.npz")
    if not os.path.exists(npz_path): return None
    try:
        data = np.load(npz_path)
        p1 = data['points1']; p2 = data['points2']; labels_gt = data['labels']
    except: return None
    
    N = len(p1)
    if N < MSS_SIZE: return None

    # 2. Load Images (View 1 and View 2)
    img1 = cv2.imread(os.path.join(scene_dir, "render0.png"))
    img2 = cv2.imread(os.path.join(scene_dir, "render1.png"))
    
    # Fallback dimensions (Standard KITTI resolution)
    if img1 is None: H, W = 375, 1242 
    else: H, W = img1.shape[:2]

    scene_out = os.path.join(OUTPUT_DIR, scene_name)
    os.makedirs(scene_out, exist_ok=True)

    # 3. Geometric Setup
    cx, cy = (W-1)/2, (H-1)/2
    f_prior = float(max(H, W)) 
    p1c = p1 - [cx, cy]; p2c = p2 - [cx, cy]
    indices_pool = np.arange(N)
    
    t_start = time.time()

    # --- MSS SAMPLING & FOCAL CHECK ---
    F_models = []; fmed_list = []; focal_ok = []
    
    for _ in range(NUM_MSS_SAMPLES):
        idx = rng.choice(indices_pool, size=MSS_SIZE, replace=False)
        F0 = eight_point_unweighted(p1c[idx], p2c[idx])
        if not np.isfinite(F0).all(): continue
        
        inl0 = np.zeros(N, bool); inl0[idx] = True
        F_ref, inl_ref = refine_F_nonlinear(p1c, p2c, F0, inl0, thr_px=BASE_THR_PX)
        
        if inl_ref.sum() < 8 or not np.isfinite(F_ref).all(): continue
        
        # Focal Filter
        if poselib is None:
            F_models.append(F_ref); fmed_list.append(f_prior); focal_ok.append(True)
            continue
            
        try:
            # Note: KITTI uses a specific constrained weights list
            bestw = choose_weights(F_ref, f_prior, f_prior, 
                                   [np.array([1e-3,1,1e-3,1]), np.array([5e-4,2,5e-4,2])],
                                   max_iters=MAX_ITERS_SOLVER)
            if bestw is None: continue # Discard degenerate model
                
            f1, f2 = int(bestw["f1"]), int(bestw["f2"])
        except: continue
        
        fmed = 0.5 * (f1 + f2)
        if fmed <= 0: continue
        
        rel = abs(f1 - f2) / fmed
        focal_ok.append(rel <= MAX_REL_F12_DIFF)
        F_models.append(F_ref)
        fmed_list.append(fmed)

    F_models = np.array(F_models, dtype=object)
    fmed_arr = np.array(fmed_list, float)
    focal_ok = np.array(focal_ok, bool)
    M_total = len(F_models)

    # --- KDE CONSENSUS ---
    valid_peak = np.isfinite(fmed_arr) & focal_ok
    if valid_peak.sum() > 0:
        vals = fmed_arr[valid_peak]
        fmin, fmax = np.percentile(vals, [1, 99])
        grid = np.linspace(fmin, fmax, 512)
        kde, _ = kde_1d(vals, grid)
        idx_pk,_ = find_peaks_1d(kde, prominence=0.05, distance=10)
        f_peak = float(grid[idx_pk[np.argmax(kde[idx_pk])]]) if len(idx_pk) > 0 else np.median(vals)
        
        F_use = []
        for Fi, fm in zip(F_models, fmed_arr):
            if np.isfinite(fm) and abs(fm - f_peak)/f_peak <= FOCAL_PEAK_REL_RANGE:
                F_use.append(Fi)
        if not F_use: F_use = F_models
    else:
        F_use = F_models

    M = len(F_use)
    if M == 0: return None

    # --- CONTINUOUS T-LINKAGE ---
    PS = np.zeros((N, M), dtype=np.float64)
    for i in range(M):
        errs = sampson_errors_numba(F_use[i].astype(np.float64), p1c, p2c)
        weights = np.exp(-errs / TAU)
        weights[errs > CUTOFF] = 0.0
        PS[:, i] = weights
    
    pred = run_tlinkage_numba_heap(PS)
    elapsed_time = time.time() - t_start

    # --- METRICS & PLOTTING ---
    K_total = pred.max() + 1
    valid_clusters = [c for c in range(K_total) if (pred==c).sum() >= MSS_SIZE]
    pred_eval = pred.copy()
    pred_eval[~np.isin(pred_eval, valid_clusters)] = -1

    ME = hungarian_me(pred_eval, labels_gt)
    ACC = 1.0 - ME
    FNR = false_negative_rate(pred_eval, labels_gt)
    ARI = adjusted_rand_score(labels_gt, pred_eval)
    
    # Save Separate Plots for View 1 and View 2
    if img1 is not None:
        save_cluster_plot(img1, p1, pred_eval, os.path.join(scene_out, "clusters_view1.png"))
    if img2 is not None:
        save_cluster_plot(img2, p2, pred_eval, os.path.join(scene_out, "clusters_view2.png"))

    print(f"  Result: ME={ME:.3f} | FNR={FNR:.3f} | M_in={M_total} -> M_out={M} | Time={elapsed_time:.2f}s")
    
    return {"scene": scene_name, "ME": ME, "ACC": ACC, "FNR": FNR, "ARI": ARI, "TIME": elapsed_time, "DISCARDED": M_total - M}

# ============================================================================
# BATCH EXECUTION
# ============================================================================
if __name__ == "__main__":
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"Output Dir:   {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_dirs = sorted([d for d in glob.glob(os.path.join(DATASET_ROOT, "kitti_*")) if os.path.isdir(d)])
    sel_dirs = all_dirs[:NUM_SCENES]

    if len(sel_dirs) == 0:
        print("[ERROR] No scenes found! Check if the directory contains 'kitti_XXXXXX' subfolders.")
    else:
        results = []
        for d in sel_dirs:
            res = process_kitti_scene(d)
            if res: results.append(res)

        if results:
            me_vals = np.array([r["ME"] for r in results])
            fnr_vals = np.array([r["FNR"] for r in results])
            time_vals = np.array([r["TIME"] for r in results])
            
            print("\n" + "="*50)
            print("📊 KITTI RESULTS SUMMARY")
            print(f"Scenes Processed: {len(results)}")
            print(f"Mean ME:  {me_vals.mean():.4f} ± {me_vals.std():.4f}")
            print(f"Mean FNR: {fnr_vals.mean():.4f} ± {fnr_vals.std():.4f}")
            print(f"Mean Time: {time_vals.mean():.4f} s")
            print("="*50)
            
            np.save(os.path.join(OUTPUT_DIR, "kitti_me.npy"), me_vals)
            np.save(os.path.join(OUTPUT_DIR, "kitti_fnr.npy"), fnr_vals)