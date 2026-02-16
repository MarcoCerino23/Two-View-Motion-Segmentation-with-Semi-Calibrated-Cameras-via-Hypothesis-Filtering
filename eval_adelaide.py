# -*- coding: utf-8 -*-
"""
Main evaluation script for the Adelaide RMF dataset.
Physically Consistent T-Linkage (Full Continuous Version).
"""

import os, glob
import numpy as np
import cv2
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

from semicalibrated_tlinkage import sampson_errors_numba, run_tlinkage_numba_heap
from geometry import refine_F_nonlinear, eight_point_unweighted, choose_weights, kde_1d, find_peaks_1d, poselib
from metrics import hungarian_me, purity, false_negative_rate, frob_dist, save_cluster_plot

np.set_printoptions(precision=3, suppress=True)

# ============================================================================
# GENERAL PARAMETERS
# ============================================================================
DATASET_ROOT = "dataset2/AdelaideRMF_Ready"  
OUTPUT_DIR   = "output_improved_adelaide_final"

MSS_SIZE            = 8
NUM_MSS_SAMPLES     = 5000  
RNG_SEED            = 12345

BASE_THR_PX         = 4.5
TAU                 = BASE_THR_PX**2
CUTOFF              = 5.0 * TAU

MAX_REL_F12_DIFF    = 0.15  
FOCAL_PEAK_REL_RANGE = 0.25 

MAX_ITERS_SOLVER    = 100
COMMON_PRIOR_W      = np.array([1e-4, 3.0, 1e-4, 3.0])

# ============================================================================
# TARGET SEQUENCES (Adelaide Fundamental Matrix Subset)
# ============================================================================
fundamental_sequences = [
    "cube.mat",
    "book.mat",
    "biscuit.mat",
    "game.mat",
    "biscuitbook.mat",
    "breadcube.mat",
    "breadtoy.mat",
    "cubechips.mat",
    "cubetoy.mat",
    "gamebiscuit.mat",
    "breadtoycar.mat",
    "carchipscube.mat",
    "toycubecar.mat",
    "breadcubechips.mat",
    "biscuitbookbox.mat",
    "cubebreadtoychips.mat",
    "breadcartoychips.mat",
    "dinobooks.mat",
    "boardgame.mat"
]

# Rimuove l'estensione '.mat' per matchare i nomi delle cartelle
target_scenes = [seq.replace(".mat", "") for seq in fundamental_sequences]

# ============================================================================
# PER-SCENE PROCESSING PIPELINE
# ============================================================================
def process_scene_focal_tlinkage_adelaide(scene_dir):
    rng=np.random.default_rng(RNG_SEED)
    scene_name=os.path.basename(scene_dir)
    print("\n"+"#"*90)
    print(f"SCENE: {scene_name}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scene_out=os.path.join(OUTPUT_DIR, scene_name)
    os.makedirs(scene_out, exist_ok=True)

    # 1. Load Matches
    matches_path = os.path.join(scene_dir, "matches.txt")
    if not os.path.exists(matches_path): return None
    try:
        data = np.loadtxt(matches_path)
        if data.shape[1] < 5: return None
        p1 = data[:, 0:2].astype(np.float64)
        p2 = data[:, 2:4].astype(np.float64)
        labels_gt = data[:, -1].astype(int)
        N = len(p1)
        if N < MSS_SIZE: return None
    except: return None

    # 2. Load Images
    all_imgs = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")) + 
                      glob.glob(os.path.join(scene_dir, "*.png")) +
                      glob.glob(os.path.join(scene_dir, "*.bmp")))
    
    img1 = img2 = None
    if len(all_imgs) >= 2:
        img1 = cv2.imread(all_imgs[0])
        img2 = cv2.imread(all_imgs[1])
    elif len(all_imgs) == 1:
        img1 = cv2.imread(all_imgs[0])
    
    if img1 is not None:
        H, W = img1.shape[:2]
    else:
        W = max(np.max(p1[:,0]), np.max(p2[:,0])) * 1.1
        H = max(np.max(p1[:,1]), np.max(p2[:,1])) * 1.1

    # 3. Geometric Setup
    cx, cy = (W-1)/2, (H-1)/2
    f_prior = float(max(H, W))
    p1c = p1 - [cx, cy]; p2c = p2 - [cx, cy]

    indices_pool = np.arange(N)
    t_pipeline_start = time.time()

    # --- MSS Sampling & Focal Check ---
    F_models=[]; fmed_list=[]; focal_ok=[]

    for _ in range(NUM_MSS_SAMPLES):
        idx=rng.choice(indices_pool, size=MSS_SIZE, replace=False)
        p1m,p2m=p1c[idx],p2c[idx]
        try:
            F0=eight_point_unweighted(p1m,p2m)
            if not np.isfinite(F0).all(): continue
        except: continue
        
        inl0=np.zeros(N,bool); inl0[idx]=True
        F_ref,inl_ref=refine_F_nonlinear(p1c,p2c,F0,inl0,thr_px=BASE_THR_PX)
        if inl_ref.sum()<8 or not np.isfinite(F_ref).all(): continue
        
        if poselib is None:
            F_models.append(F_ref); fmed_list.append(f_prior); focal_ok.append(True)
            continue
            
        try:
            bestw=choose_weights(
                F_ref, f_prior, f_prior,
                [np.array([1e-3,1,1e-3,1]), np.array([5e-4,1,5e-4,1]),
                 np.array([1e-4,1,1e-4,1]), np.array([5e-4,2,5e-4,2])],
                max_iters=MAX_ITERS_SOLVER
            )
            if bestw is None: continue
            f1,f2=int(bestw["f1"]),int(bestw["f2"])
        except: continue
        
        fmed=0.5*(f1+f2)
        if fmed<=0: continue
        rel=abs(f1-f2)/fmed
        focal_ok.append(rel<=MAX_REL_F12_DIFF)
        F_models.append(F_ref)
        fmed_list.append(fmed)

    F_models=np.array(F_models,dtype=object)
    fmed_arr=np.array(fmed_list,float)
    focal_ok=np.array(focal_ok,bool)
    M_total=len(F_models)

    # --- Focal Filtering (KDE) ---
    valid_peak=np.isfinite(fmed_arr)&focal_ok
    if valid_peak.sum()==0:
        F_use=F_models
        M=len(F_use)
    else:
        vals=fmed_arr[valid_peak]
        fmin,fmax=np.percentile(vals,[1,99])
        grid=np.linspace(fmin,fmax,512)
        kde,_=kde_1d(vals,grid)
        idx_pk,_=find_peaks_1d(kde,prominence=0.05,distance=10)
        if len(idx_pk)==0: f_peak=np.median(vals)
        else: f_peak=float(grid[idx_pk[np.argmax(kde[idx_pk])]])
        
        F_use=[]
        for Fi,fm in zip(F_models,fmed_arr):
            if np.isfinite(fm) and abs(fm - f_peak)/f_peak <= FOCAL_PEAK_REL_RANGE:
                F_use.append(Fi)
        if len(F_use)==0: F_use=F_models
        M=len(F_use)

    discarded = M_total - M
    discard_ratio = discarded / M_total if M_total > 0 else 0.0
    print(f"  Total models: {M_total} -> Used: {M} (Discarded: {discarded}, {discard_ratio*100:.1f}%)")
    
    if M == 0: return None

    # --- Continuous T-Linkage ---
    t_start = time.time()
    
    PS = np.zeros((N, M), dtype=np.float64)
    for i in range(M):
        errs = sampson_errors_numba(F_use[i].astype(np.float64), p1c, p2c)
        weights = np.exp(-errs / TAU)
        weights[errs > CUTOFF] = 0.0
        PS[:, i] = weights
    
    pred = run_tlinkage_numba_heap(PS)
    
    elapsed_time = time.time() - t_start
    print(f"  T-Linkage Heap (Cont): {elapsed_time:.4f} sec")
    total_time = time.time() - t_pipeline_start

    K_total=pred.max()+1
    valid_clusters=[c for c in range(K_total) if (pred==c).sum()>=MSS_SIZE]
    pred_eval=pred.copy()
    pred_eval[~np.isin(pred_eval,valid_clusters)] = -1

    # --- PLOTTING SEPARATE VIEWS ---
    if img1 is not None:
        save_cluster_plot(img1, p1, pred_eval, os.path.join(scene_out, "clusters_view1.png"))
    if img2 is not None:
        save_cluster_plot(img2, p2, pred_eval, os.path.join(scene_out, "clusters_view2.png"))

    # --- Evaluation ---
    F_est_list = []
    for c in valid_clusters:
        mask = (pred == c)
        if mask.sum() >= 8: F_est_list.append(eight_point_unweighted(p1c[mask], p2c[mask]))
    F_gt_list = []
    unique_gt = np.unique(labels_gt)
    for g in unique_gt:
        if g == 0: continue
        mask = (labels_gt == g)
        if mask.sum() >= 8: F_gt_list.append(eight_point_unweighted(p1c[mask], p2c[mask]))
    frob_mean = np.nan
    if len(F_est_list) > 0 and len(F_gt_list) > 0:
        cost_matrix = np.zeros((len(F_est_list), len(F_gt_list)))
        for i, Fe in enumerate(F_est_list):
            for j, Fg in enumerate(F_gt_list): cost_matrix[i, j] = frob_dist(Fe, Fg)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        frob_mean = cost_matrix[row_ind, col_ind].mean()

    ME=hungarian_me(pred_eval,labels_gt)
    ACC=1-ME; PUR=purity(pred_eval,labels_gt); FNR=false_negative_rate(pred_eval,labels_gt)
    ARI=adjusted_rand_score(labels_gt,pred_eval); NMI=normalized_mutual_info_score(labels_gt,pred_eval)

    print(f"  METRICS -> ME={ME:.3f}, ACC={ACC:.3f}, FNR={FNR:.3f}, PUR={PUR:.3f}, ARI={ARI:.3f}")
    if np.isfinite(frob_mean): print(f"  FROBENIUS = {frob_mean:.4f}")

    return dict(
        scene=scene_name, N=N, M=M, ME=ME, ACC=ACC, FNR=FNR, PUR=PUR, ARI=ARI, NMI=NMI, FROB=frob_mean,
        TIME=elapsed_time, TIME_TOTAL=total_time, DISCARDED=discarded, DISC_RATIO=discard_ratio
    )

# ============================================================================
# BATCH EXECUTION
# ============================================================================
if __name__ == "__main__":
    all_dirs = sorted([d for d in glob.glob(os.path.join(DATASET_ROOT,"*")) if os.path.isdir(d)])
    
    # Filtra le cartelle usando la lista target_scenes
    sel_dirs = [d for d in all_dirs if os.path.basename(d) in target_scenes]

    print(f"Found {len(sel_dirs)} out of {len(target_scenes)} requested scenes in {DATASET_ROOT}")

    all_results=[]
    for sd in sel_dirs:
        try:
            r=process_scene_focal_tlinkage_adelaide(sd)
            if r: all_results.append(r)
        except Exception as e:
            print(f"[ERR] {sd}: {e}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    if all_results:
        ME_arr   = np.array([r["ME"] for r in all_results])
        ACC_arr  = np.array([r["ACC"] for r in all_results])
        FNR_arr  = np.array([r["FNR"] for r in all_results])
        PUR_arr  = np.array([r["PUR"] for r in all_results])
        ARI_arr  = np.array([r["ARI"] for r in all_results])
        FROB_arr = np.array([r["FROB"] for r in all_results if np.isfinite(r["FROB"])])
        TIME_arr = np.array([r["TIME"] for r in all_results])
        DISC_arr = np.array([r["DISCARDED"] for r in all_results])

        print("\n"+"="*80)
        print("📊 GLOBAL METRICS IMPROVED (MEAN ± STD)")
        print(f"ME   = {ME_arr.mean():.4f} ± {ME_arr.std():.4f}")
        print(f"ACC  = {ACC_arr.mean():.4f} ± {ACC_arr.std():.4f}")
        print(f"FNR  = {FNR_arr.mean():.4f} ± {FNR_arr.std():.4f}")
        print(f"PUR  = {PUR_arr.mean():.4f} ± {PUR_arr.std():.4f}")
        print(f"ARI  = {ARI_arr.mean():.4f} ± {ARI_arr.std():.4f}")
        print(f"FROB = {FROB_arr.mean():.4f} ± {FROB_arr.std():.4f}")
        print("-" * 40)
        print(f"Average T-Linkage Time = {TIME_arr.mean():.4f} sec")
        print(f"Average Discarded Models = {DISC_arr.mean():.1f}")

        np.save("results_improved_adelaide_FNR.npy", FNR_arr) 
        np.save("results_improved_adelaide_ME.npy", ME_arr)
    else:
        print("No valid results. Check dataset paths and libraries.")