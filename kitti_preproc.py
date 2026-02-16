# -*- coding: utf-8 -*-
"""
kitti_to_npz.py
Converts the KITTI Flow 2015 dataset into the .npz format compatible with T-Linkage.
Uses RELATIVE paths: it must be executed in the same folder containing the 'training' directory.
"""

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Look for the 'training' folder right next to it
KITTI_ROOT = os.path.join(SCRIPT_DIR, "training")

# Create the output folder right next to it
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset_processed")

NUM_FEATURES = 2000

def process_kitti_sequence(seq_id, root_dir, out_dir):
    img1_path = os.path.join(root_dir, "image_2", f"{seq_id}_10.png")
    img2_path = os.path.join(root_dir, "image_2", f"{seq_id}_11.png")
    gt_map_path = os.path.join(root_dir, "obj_map", f"{seq_id}_10.png")
    if not os.path.exists(img1_path) or not os.path.exists(gt_map_path):
        return
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    gt_map = cv2.imread(gt_map_path, cv2.IMREAD_GRAYSCALE) 

    if img1 is None or img2 is None or gt_map is None:
        return


    sift = cv2.SIFT_create(nfeatures=NUM_FEATURES)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if kp1 is None or kp2 is None or des1 is None or des2 is None:
        return
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    pts1 = []
    pts2 = []
    labels = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            p1 = kp1[m.queryIdx].pt
            p2 = kp2[m.trainIdx].pt
            
            x, y = int(p1[0]), int(p1[1])
            
            if 0 <= y < gt_map.shape[0] and 0 <= x < gt_map.shape[1]:
                label_val = gt_map[y, x]

                if label_val == 0:
                    l = 1 
                else:
                    l = 2 
                
                pts1.append(p1)
                pts2.append(p2)
                labels.append(l)

    if len(pts1) < 8: 
        return

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    labels = np.array(labels)

    K = np.array([[721.5, 0, 609.5], [0, 721.5, 172.8], [0, 0, 1]])

    scene_dir = os.path.join(out_dir, f"kitti_{seq_id}")
    os.makedirs(scene_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(scene_dir, "render0.png"), img1)
    cv2.imwrite(os.path.join(scene_dir, "render1.png"), img2)
    
    np.savez(os.path.join(scene_dir, "features_and_ground_truth.npz"),
             points1=pts1, points2=pts2, labels=labels)
    
    np.savez(os.path.join(scene_dir, "camera_parameters.npz"), K=K)

if __name__ == "__main__":
    print(f"Script Directory: {SCRIPT_DIR}")
    print(f"Input Directory:  {KITTI_ROOT}")
    print(f"Output Directory: {OUTPUT_DIR}")

    if not os.path.exists(KITTI_ROOT):
        print("\n[ERROR] Cannot find the 'training' folder.")
        print("Make sure to execute the script INSIDE the folder containing 'training'.")
        exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting conversion (200 scenes)...")
    for i in tqdm(range(200)):
        seq_str = f"{i:06d}"
        process_kitti_sequence(seq_str, KITTI_ROOT, OUTPUT_DIR)

    print("\nConversion completed!")
    print(f"Ready data is located at: {OUTPUT_DIR}")