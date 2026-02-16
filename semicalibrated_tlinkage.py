# -*- coding: utf-8 -*-
"""
Core algorithmic logic for Continuous T-Linkage with Min-Heap optimization.
This module implements the Preference Representation and Clustering phase 
(Step 4 of Section 4.1 in the paper), including the continuous preference 
computation and the agglomerative clustering using Tanimoto distance.
"""

import numpy as np
from numba import njit, float64, int32, boolean

# ============================================================================
# NUMBA HEAP UTILS (Optimizations for Algorithm 1, Line 15)
# ============================================================================
@njit(cache=True)
def _heap_push(val, i, j, h_val, h_i, h_j, size):
    """
    Inserts a distance element into the min-heap.
    This is a computational optimization to efficiently find the minimum 
    Tanimoto distance during the agglomerative clustering (Section 4.1, Step 4).
    """
    idx = size
    h_val[idx] = val
    h_i[idx] = i
    h_j[idx] = j
    
    while idx > 0:
        parent = (idx - 1) // 2
        swap = False
        if h_val[idx] < h_val[parent]:
            swap = True
        elif h_val[idx] == h_val[parent]:
            if h_i[idx] < h_i[parent]:
                swap = True
            elif h_i[idx] == h_i[parent]:
                if h_j[idx] < h_j[parent]:
                    swap = True
        
        if swap:
            v_tmp, i_tmp, j_tmp = h_val[parent], h_i[parent], h_j[parent]
            h_val[parent], h_i[parent], h_j[parent] = h_val[idx], h_i[idx], h_j[idx]
            h_val[idx], h_i[idx], h_j[idx] = v_tmp, i_tmp, j_tmp
            idx = parent
        else:
            break
    return size + 1

@njit(cache=True)
def _heap_pop(h_val, h_i, h_j, size):
    """
    Extracts the minimum distance element from the min-heap.
    Used to rapidly find the closest pair of clusters to merge.
    """
    if size <= 0:
        return 0.0, -1, -1, 0
    
    res_v, res_i, res_j = h_val[0], h_i[0], h_j[0]
    
    last = size - 1
    h_val[0], h_i[0], h_j[0] = h_val[last], h_i[last], h_j[last]
    new_size = last
    
    idx = 0
    while True:
        left = 2 * idx + 1
        right = 2 * idx + 2
        smallest = idx
        
        if left < new_size:
            is_smaller = False
            if h_val[left] < h_val[smallest]: is_smaller = True
            elif h_val[left] == h_val[smallest]:
                if h_i[left] < h_i[smallest]: is_smaller = True
                elif h_i[left] == h_i[smallest]:
                    if h_j[left] < h_j[smallest]: is_smaller = True
            if is_smaller:
                smallest = left
                
        if right < new_size:
            is_smaller = False
            if h_val[right] < h_val[smallest]: is_smaller = True
            elif h_val[right] == h_val[smallest]:
                if h_i[right] < h_i[smallest]: is_smaller = True
                elif h_i[right] == h_i[smallest]:
                    if h_j[right] < h_j[smallest]: is_smaller = True
            if is_smaller:
                smallest = right
        
        if smallest != idx:
            v_tmp, i_tmp, j_tmp = h_val[idx], h_i[idx], h_j[idx]
            h_val[idx], h_i[idx], h_j[idx] = h_val[smallest], h_i[smallest], h_j[smallest]
            h_val[smallest], h_i[smallest], h_j[smallest] = v_tmp, i_tmp, j_tmp
            idx = smallest
        else:
            break
            
    return res_v, res_i, res_j, new_size

# ============================================================================
# NUMBA KERNELS (Geometric & Continuous Tanimoto)
# ============================================================================
@njit(cache=True, fastmath=True)
def sampson_errors_numba(F, p1, p2):
    """
    Calculates the squared Sampson distance for the entire dataset against a hypothesis F.
    Refers to the geometric residual r_{ij} used to build the preference matrix 
    in Section 4.1, Step 4, Equation (7).
    """
    N = p1.shape[0]
    errors = np.empty(N, dtype=np.float64)
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
        den = Fx1_0**2 + Fx1_1**2 + Ftx2_0**2 + Ftx2_1**2
        if den <= 1e-15: den = 1e-15
        errors[i] = (num**2) / den
    return errors

@njit(cache=True, fastmath=True)
def tanimoto_distance_jit(vec_a, vec_b):
    """
    Computes the Tanimoto distance between two continuous preference vectors.
    Refers strictly to Equation (8) in Section 4.1, Step 4:
    d_T(p_i, p_k) = 1 - (<p_i, p_k> / (||p_i||^2 + ||p_k||^2 - <p_i, p_k>)).
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    n = len(vec_a)
    
    for k in range(n):
        val_a = vec_a[k]
        val_b = vec_b[k]
        if val_a > 1e-9 or val_b > 1e-9:
            dot += val_a * val_b
            norm_a += val_a * val_a
            norm_b += val_b * val_b
            
    den = norm_a + norm_b - dot
    if den <= 1e-15:
        return 1.0 
    return 1.0 - (dot / den)

@njit(cache=True)
def run_tlinkage_numba_heap(PS_float):
    """
    Agglomerative T-Linkage Clustering logic using Min-Heap optimization.
    Refers to Algorithm 1, Line 15: L <- T-LinkageClustering(P).
    Implements the termination condition described in Section 4.1, Step 4, 
    halting the agglomeration when the Tanimoto distance d_T equals 1 
    (i.e., preference sets become mutually exclusive).
    """
    N, M = PS_float.shape
    active = np.ones(N, dtype=np.bool_)
    labels = np.arange(N, dtype=np.int32)
    clusters_PS = PS_float.copy()
    
    D = np.full((N, N), np.inf, dtype=np.float64)

    MAX_HEAP_SIZE = N * N * 2 
    h_val = np.empty(MAX_HEAP_SIZE, dtype=np.float64)
    h_i = np.empty(MAX_HEAP_SIZE, dtype=np.int32)
    h_j = np.empty(MAX_HEAP_SIZE, dtype=np.int32)
    h_size = 0

    for i in range(N):
        for j in range(i + 1, N):
            d = tanimoto_distance_jit(clusters_PS[i], clusters_PS[j])
            D[i, j] = d
            D[j, i] = d 
            # Agglomerate only if they share preferences (d_T < 1)
            if d < 1.0:
                h_size = _heap_push(d, i, j, h_val, h_i, h_j, h_size)

    n_active = N

    while n_active > 1:
        best_d = 1.0
        best_i = -1
        best_j = -1
        
        while h_size > 0:
            d, u, v, h_size = _heap_pop(h_val, h_i, h_j, h_size)
            # Termination condition: mutually exclusive preference sets
            if d >= 1.0 - 1e-15:
                h_size = 0 
                break

            if active[u] and active[v] and abs(d - D[u, v]) < 1e-12:
                best_d = d
                best_i = u
                best_j = v
                break
        
        if best_i == -1: 
            break
            
        # Update the preference of the merged cluster
        for k in range(M):
            val_i = clusters_PS[best_i, k]
            val_j = clusters_PS[best_j, k]
            if val_j < val_i:
                clusters_PS[best_i, k] = val_j
        
        for p in range(N):
            if labels[p] == best_j: labels[p] = best_i
        
        active[best_j] = False
        n_active -= 1
        
        for k in range(N):
            if active[k] and k != best_i:
                new_d = tanimoto_distance_jit(clusters_PS[best_i], clusters_PS[k])
                if best_i < k:
                    D[best_i, k] = new_d
                    if new_d < 1.0:
                         h_size = _heap_push(new_d, best_i, k, h_val, h_i, h_j, h_size)
                else:
                    D[k, best_i] = new_d
                    if new_d < 1.0:
                         h_size = _heap_push(new_d, k, best_i, h_val, h_i, h_j, h_size)

    final_labels = np.full(N, -1, dtype=np.int32)
    unique_active = np.unique(labels)
    map_ids = np.full(N, -1, dtype=np.int32)
    current_cid = 0
    for old_id in unique_active:
        map_ids[old_id] = current_cid
        current_cid += 1
    for i in range(N):
        final_labels[i] = map_ids[labels[i]]
    return final_labels