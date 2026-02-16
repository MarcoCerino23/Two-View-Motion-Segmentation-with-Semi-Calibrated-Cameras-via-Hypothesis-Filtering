# -*- coding: utf-8 -*-
"""
Main evaluation script for Physically Consistent T-Linkage.
"""

import os, glob
import numpy as np
import cv2
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

# Custom local imports
from semicalibrated_tlinkage import sampson_errors_numba, run_tlinkage_numba_heap
from geometry import refine_F_nonlinear, eight_point_unweighted, choose_weights, kde_1d, find_peaks_1d
from metrics import hungarian_me, purity, false_negative_rate, frob_dist, save_cluster_plot

np.set_printoptions(precision=3, suppress=True)

# ============================================================================
# GENERAL PARAMETERS
# ============================================================================
DATASET_ROOT_BASE = "dataset" 
SUB_CATEGORIES    = ["1", "2", "3", "4"]

TEST_START_IDX = 900
TEST_END_IDX   = 1000 

OUTPUT_DIR   = "output_hopef_test_focal_tlinkage"

MSS_SIZE            = 8
NUM_MSS_SAMPLES     = 3000
RNG_SEED            = 12345

BASE_THR_PX         = 2.5  
MAX_REL_F12_DIFF    = 0.15
FOCAL_PEAK_REL_RANGE = 0.25

MAX_ITERS_SOLVER    = 120
COMMON_PRIOR_W      = np.array([1e-4, 3.0, 1e-4, 3.0])

# ============================================================================
# PER-SCENE PROCESSING PIPELINE
# ============================================================================
def process_scene_focal_tlinkage(scene_dir):
    rng=np.random.default_rng(RNG_SEED)
    scene_name=os.path.basename(scene_dir)
    category = os.path.basename(os.path.dirname(scene_dir))

    print("\n"+"#"*90)
    print(f"SCENE: {scene_name} (Category: {category})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scene_out = os.path.join(OUTPUT_DIR, category, scene_name)
    os.makedirs(scene_out, exist_ok=True)

    img1=cv2.imread(os.path.join(scene_dir,"render0.png"))
    img2=cv2.imread(os.path.join(scene_dir,"render1.png"))
    if img1 is None or img2 is None: return None
    H,W=img1.shape[:2]
    
    cv2.imwrite(os.path.join(scene_out,"view0.png"), img1)
    cv2.imwrite(os.path.join(scene_out,"view1.png"), img2)

    cam_path=os.path.join(scene_dir,"camera_parameters.npz")
    K=None; fgt=None; cxK=cyK=None
    if os.path.exists(cam_path):
        cam=np.load(cam_path)
        K=cam["K"].astype(float)
        fgt=float(K[0,0])
        cxK,cyK=float(K[0,2]),float(K[1,2])
    
    ff=np.load(os.path.join(scene_dir,"features_and_ground_truth.npz"))
    p1=ff["points1"][:,:2].astype(float)
    p2=ff["points2"][:,:2].astype(float)
    labels_gt=ff["labels"]
    N=len(p1)
    if N<MSS_SIZE: return None

    cx=cxK if K is not None else (W-1)/2
    cy=cyK if K is not None else (H-1)/2
    f_prior=float(max(H,W))
    p1c=p1 - [cx,cy]
    p2c=p2 - [cx,cy]

    indices_pool = np.arange(N)

    t_pipeline_start = time.time()

    # --- MSS Sampling & Focal Check ---
    F_models=[]; fmed_list=[]; focal_ok=[]
    mss_count=0

    for _ in range(NUM_MSS_SAMPLES):
        idx=rng.choice(indices_pool, size=MSS_SIZE, replace=False)
        mss_count+=1
        p1m,p2m=p1c[idx],p2c[idx]
        try:
            F0=eight_point_unweighted(p1m,p2m)
            if not np.isfinite(F0).all(): continue
        except: continue
        inl0=np.zeros(N,bool); inl0[idx]=True
        F_ref,inl_ref=refine_F_nonlinear(p1c,p2c,F0,inl0,thr_px=BASE_THR_PX)
        if inl_ref.sum()<8 or not np.isfinite(F_ref).all(): continue
        
        bestw=choose_weights(
            F_ref,f_prior,f_prior,
            [np.array([1e-3,1,1e-3,1]), np.array([5e-4,1,5e-4,1]),
             np.array([1e-4,1,1e-4,1]), np.array([5e-4,2,5e-4,2]),
             np.array([1e-3,2,1e-3,2]), np.array([1e-4,3,1e-4,3])],
            max_iters=MAX_ITERS_SOLVER
        )
        if bestw is None:
            F_models.append(F_ref); fmed_list.append(f_prior); focal_ok.append(True)
            continue
            
        try:
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

    # --- Focal Filtering ---
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

    t_start = time.time()
    
    TAU = BASE_THR_PX**2
    CUTOFF = 5.0 * TAU
    
    PS = np.zeros((N, M), dtype=np.float64)
    for i in range(M):
        errs = sampson_errors_numba(F_use[i].astype(np.float64), p1c, p2c)
        weights = np.exp(-errs / TAU)
        weights[errs > CUTOFF] = 0.0
        PS[:, i] = weights
    
    pred = run_tlinkage_numba_heap(PS)
    
    t_end = time.time()
    elapsed_time = t_end - t_start
    print(f"  T-Linkage Time (Heap+Continuous): {elapsed_time:.4f} sec")

    t_pipeline_end = time.time()
    total_time = t_pipeline_end - t_pipeline_start
    print(f"  Total pipeline time: {total_time:.4f} sec")

    K_total=pred.max()+1
    valid_clusters=[c for c in range(K_total) if (pred==c).sum()>=MSS_SIZE]
    pred_eval=pred.copy()
    pred_eval[~np.isin(pred_eval,valid_clusters)] = -1

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

    save_cluster_plot(img1, p1, pred_eval, os.path.join(scene_out, "clusters_view1.png"))
    save_cluster_plot(img2, p2, pred_eval, os.path.join(scene_out, "clusters_view2.png"))

    ME=hungarian_me(pred_eval,labels_gt)
    ACC=1-ME; PUR=purity(pred_eval,labels_gt); FNR=false_negative_rate(pred_eval,labels_gt)
    ARI=adjusted_rand_score(labels_gt,pred_eval); NMI=normalized_mutual_info_score(labels_gt,pred_eval)

    print(f"  METRICS → ME={ME:.3f}, ACC={ACC:.3f}, FNR={FNR:.3f}, PUR={PUR:.3f}, ARI={ARI:.3f}, NMI={NMI:.3f}")
    print(f"  FROBENIUS = {frob_mean:.4f}")

    return dict(
        scene=scene_name, N=N, M_models_used=M, clusters_total=K_total,
        ME=ME, ACC=ACC, FNR=FNR, PUR=PUR, ARI=ARI, NMI=NMI, FROB=frob_mean,
        TIME=elapsed_time, TIME_TOTAL=total_time, DISCARDED=discarded, DISC_RATIO=discard_ratio
    )

# ============================================================================
# BATCH LOOP
# ============================================================================
if __name__ == "__main__":
    all_results=[]

    print(f"\nSTARTING EVALUATION ON HOPE-F TEST SET")
    print(f"Categories: {SUB_CATEGORIES}")
    print(f"Sequence Range: {TEST_START_IDX} - {TEST_END_IDX-1}")

    for cat in SUB_CATEGORIES:
        cat_dir = os.path.join(DATASET_ROOT_BASE, cat)
        
        if not os.path.exists(cat_dir):
            print(f"WARNING: Directory {cat_dir} not found, skipping.")
            continue
            
        all_scenes = sorted([
            d for d in glob.glob(os.path.join(cat_dir, "*")) 
            if os.path.isdir(d)
        ])
        
        if len(all_scenes) >= 1000:
            test_scenes = all_scenes[TEST_START_IDX:TEST_END_IDX]
            print(f"\n--- Processing Category {cat}: {len(test_scenes)} TEST sequences ---")
            
            for sd in test_scenes:
                try:
                    r=process_scene_focal_tlinkage(sd)
                    if r: all_results.append(r)
                except Exception as e:
                    print(f"[ERROR in scene] {sd}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"WARNING: Category {cat} has only {len(all_scenes)} sequences. Cannot extract range {TEST_START_IDX}-{TEST_END_IDX}.")


    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    if all_results:
        print("\n"+"="*100)
        print(f"FINAL SUMMARY ON {len(all_results)} TEST SCENES")

        print("\n"+"="*100)
        print("📊 GLOBAL METRICS HOPE-F TEST SET (MEAN ± STD)")

        ME_arr  = np.array([r["ME"]  for r in all_results])
        ACC_arr = np.array([r["ACC"] for r in all_results])
        FNR_arr = np.array([r["FNR"] for r in all_results])
        PUR_arr = np.array([r["PUR"] for r in all_results])
        ARI_arr = np.array([r["ARI"] for r in all_results])
        NMI_arr = np.array([r["NMI"] for r in all_results])
        FROB_arr= np.array([r["FROB"] for r in all_results if np.isfinite(r["FROB"])])
        TIME_arr= np.array([r["TIME"] for r in all_results])
        RATIO_arr= np.array([r["DISC_RATIO"] for r in all_results])
        DISC_arr = np.array([r["DISCARDED"] for r in all_results])
        TIME_TOTAL_arr = np.array([r["TIME_TOTAL"] for r in all_results])

        print(f"ME   = {ME_arr.mean():.4f} ± {ME_arr.std():.4f}")
        print(f"ACC  = {ACC_arr.mean():.4f} ± {ACC_arr.std():.4f}")
        print(f"FNR  = {FNR_arr.mean():.4f} ± {FNR_arr.std():.4f}")
        print(f"PUR  = {PUR_arr.mean():.4f} ± {PUR_arr.std():.4f}")
        print(f"ARI  = {ARI_arr.mean():.4f} ± {ARI_arr.std():.4f}")
        print(f"NMI  = {NMI_arr.mean():.4f} ± {NMI_arr.std():.4f}")
        print(f"FROB = {FROB_arr.mean():.4f} ± {FROB_arr.std():.4f}")
        print("-" * 40)
        print(f"Average T-Linkage Time = {TIME_arr.mean():.4f} sec")
        print(f"Average PIPELINE Time (Total) = {TIME_TOTAL_arr.mean():.4f} sec ± {TIME_TOTAL_arr.std():.4f}")
        print(f"Average Discarded Models = {DISC_arr.mean():.1f} (Drop Rate: {RATIO_arr.mean()*100:.1f}%)")

        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
        np.save(os.path.join(OUTPUT_DIR, "results_improved_FNR.npy"), FNR_arr) 
        np.save(os.path.join(OUTPUT_DIR, "results_improved_ME.npy"), ME_arr)
        print(f"Results saved in {OUTPUT_DIR}")
    else:
        print("No results calculated. Please check the dataset paths.")