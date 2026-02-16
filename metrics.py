# -*- coding: utf-8 -*-
"""
Evaluation metrics and plotting utilities.
This module implements the quantitative evaluation metrics described 
in Section 5.3 of the paper, including the Misclassification Error (ME) 
and the Frobenius Norm (FROB).
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def normalize_F(F):
    """
    Normalizes the Fundamental matrix to unit norm.
    Used for computing the geometric accuracy of the estimated models.
    """
    n = np.linalg.norm(F)
    return F / n if n > 1e-9 else F

def frob_dist(F1, F2):
    """
    Computes the Frobenius Norm (FROB) between two Fundamental matrices.
    Refers to the FROB metric mentioned in Section 5.3 of the paper to evaluate 
    the geometric accuracy of the estimated Fundamental matrices.
    """
    F1n = normalize_F(F1)
    F2n = normalize_F(F2)
    return min(np.linalg.norm(F1n - F2n, 'fro'), np.linalg.norm(F1n + F2n, 'fro'))

def save_cluster_plot(img, pts, labels, path):
    """
    Generates and saves the qualitative segmentation results.
    Refers to the visualizations presented in Section 5.6 (Qualitative Results)
    and Figures 1 and 4 of the paper.
    """
    if labels.max() < 0: return
    if hasattr(matplotlib, 'colormaps'): cmap = matplotlib.colormaps['tab20']
    else: cmap = plt.cm.get_cmap("tab20")
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    unique_labels = np.unique(labels)
    for cid in unique_labels:
        if cid < 0: continue
        mask = labels == cid
        color = cmap(cid % 20)
        plt.scatter(pts[mask, 0], pts[mask, 1], s=10, color=color, label=f"C{cid}")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    
def hungarian_me(pred, gt):
    """
    Computes the Misclassification Error (ME) as the fraction of points 
    assigned to an incorrect cluster.
    Refers strictly to Equation (9) in Section 5.3.
    The optimal label permutation is computed using the Hungarian algorithm 
    (Kuhn, 1955) as explicitly stated in the text.
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    valid_gt_mask = gt != 0
    if np.sum(valid_gt_mask) == 0: return 0.0
    gt_clean = gt[valid_gt_mask]
    pred_clean = pred[valid_gt_mask]
    if len(pred_clean) == 0: return 1.0
    pred_ids = np.unique(pred_clean)
    gt_ids = np.unique(gt_clean)
    if len(pred_ids) == 1 and pred_ids[0] == -1: return 1.0
    valid_p_ids = pred_ids[pred_ids != -1]
    if len(valid_p_ids) == 0: return 1.0
    
    # Build cost matrix for the Hungarian algorithm
    C = np.zeros((len(valid_p_ids), len(gt_ids)), dtype=int)
    pid_map = {pid: i for i, pid in enumerate(valid_p_ids)}
    gid_map = {gid: i for i, gid in enumerate(gt_ids)}
    for i, p in enumerate(pred_clean):
        if p != -1:
            g = gt_clean[i]
            C[pid_map[p], gid_map[g]] += 1
            
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(-C)
    correct = C[row_ind, col_ind].sum()
    
    # Return Misclassification Error (Equation 9)
    return 1.0 - (correct / len(gt_clean))

def purity(pred, gt):
    """
    Computes the Purity (PUR) of the segmentation.
    Refers to the PUR metric mentioned in Section 5.3 of the paper.
    """
    pred=np.asarray(pred); gt=np.asarray(gt)
    valid=pred>=0
    pred=pred[valid]; gt=gt[valid]
    if len(pred) == 0: return 0.0
    clusters=np.unique(pred)
    correct=0
    for c in clusters:
        mask=(pred==c)
        if mask.sum()==0: continue
        vals,cnts=np.unique(gt[mask],return_counts=True)
        correct+=cnts.max()
    return correct/len(pred)

def false_negative_rate(pred, gt):
    """
    Computes the False Negative Rate (FNR) of the segmentation.
    Refers to the FNR metric mentioned in Section 5.3 and analyzed 
    in the Statistical Significance test.
    """
    gt_valid_mask = (gt != 0)
    total_valid_gt = np.sum(gt_valid_mask)
    if total_valid_gt == 0: return 0.0 
    valid_p_mask = (pred >= 0)
    if np.sum(valid_p_mask) == 0: return 1.0
    p_curr = pred[valid_p_mask]
    g_curr = gt[valid_p_mask]
    pids = np.unique(p_curr)
    gids = np.unique(g_curr)
    
    C = np.zeros((len(pids), len(gids)), dtype=int)
    pid_map = {p: i for i, p in enumerate(pids)}
    gid_map = {g: i for i, g in enumerate(gids)}
    for i, p in enumerate(pids):
        mask_p = (p_curr == p)
        gt_in_cluster = g_curr[mask_p]
        u_gt, counts = np.unique(gt_in_cluster, return_counts=True)
        for g_val, count in zip(u_gt, counts):
            if g_val in gid_map: C[pid_map[p], gid_map[g_val]] = count
            
    row_ind, col_ind = linear_sum_assignment(-C)
    pred_to_gt = {}
    for r, c in zip(row_ind, col_ind): pred_to_gt[pids[r]] = gids[c]
    
    correct = 0
    pred_on_valid = pred[gt_valid_mask]
    gt_on_valid = gt[gt_valid_mask]
    for p, g in zip(pred_on_valid, gt_on_valid):
        if p >= 0 and p in pred_to_gt:
            if pred_to_gt[p] == g: correct += 1
            
    return 1.0 - (correct / total_valid_gt)