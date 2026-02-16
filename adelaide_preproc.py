import os
import scipy.io
import numpy as np
import cv2
import shutil

SOURCE_PATH = "adelaidermf"
OUTPUT_ROOT = "dataset2/AdelaideRMF_Ready"

def rebuild_adelaide_complete():
    if not os.path.exists(SOURCE_PATH):
        print(f"ERROR: Folder '{SOURCE_PATH}' not found.")
        return

    if os.path.exists(OUTPUT_ROOT):
        print(f"🧹 Cleaning existing output folder: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    files = [f for f in os.listdir(SOURCE_PATH) if f.endswith(".mat")]
    print(f"Starting COMPLETE extraction (Img + Data) for {len(files)} scenes...")
    print("-" * 60)

    count_ok = 0
    count_skip = 0

    for f in files:
        mat_path = os.path.join(SOURCE_PATH, f)
        scene_name = f.replace(".mat", "")
        output_dir = os.path.join(OUTPUT_ROOT, scene_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            mat = scipy.io.loadmat(mat_path)
            
            has_images = False
            if 'img1' in mat and 'img2' in mat:
                img1 = mat['img1']
                img2 = mat['img2']
                
                if img1.ndim >= 2 and img2.ndim >= 2:
                    if img1.ndim == 3:
                        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
                    if img2.ndim == 3:
                        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                        
                    cv2.imwrite(os.path.join(output_dir, "img1.png"), img1)
                    cv2.imwrite(os.path.join(output_dir, "img2.png"), img2)
                    has_images = True
            
            if not has_images:
                print(f"{scene_name}: Images not found in .mat (proceeding with data only)")

            raw = None
            if 'data' in mat: raw = mat['data']
            elif 'x' in mat: raw = mat['x']
            
            if raw is None:
                print(f"{scene_name}: Data matrix missing. Skip.")
                count_skip += 1
                continue

            if raw.shape[0] < raw.shape[1]:
                raw = raw.T
            
            N, D = raw.shape
            
            p1, p2, gt = None, None, None
            
            if D >= 7:
                idx_label = 8 if D >= 9 else 6
                p1 = raw[:, 0:2]
                p2 = raw[:, 3:5]
                gt = raw[:, idx_label]

            elif D == 6:
                p1 = raw[:, 0:2]
                p2 = raw[:, 3:5]
                
                possible_keys = ['gt', 'label', 'labels', 'group', 'clustering', 'S', 'membership']
                for key in possible_keys:
                    if key in mat:
                        gt_temp = mat[key]
                        if gt_temp.size == N:
                            gt = gt_temp.reshape(-1)
                            break
            
            if gt is None:
                print(f"{scene_name}: Unable to find Labels. Skip.")
                count_skip += 1
                continue

            gt = gt.reshape(-1, 1)
            clean_data = np.hstack([p1, p2, gt])
            
            valid_mask = np.isfinite(clean_data).all(axis=1)
            clean_data = clean_data[valid_mask]

            if len(clean_data) < 8:
                print(f"⚠️  {scene_name}: Too few valid points ({len(clean_data)}). Skip.")
                count_skip += 1
                continue

            np.savetxt(os.path.join(output_dir, "matches.txt"), clean_data, fmt="%.6f %.6f %.6f %.6f %d")
            
            count_ok += 1

        except Exception as e:
            print(f"Critical error on {scene_name}: {e}")
            count_skip += 1

    print("-" * 60)
    print(f"RECONSTRUCTION COMPLETED.\n Ready scenes: {count_ok}\n⚠️ Failed scenes: {count_skip}")
    print(f"Output folder: {OUTPUT_ROOT}")

rebuild_adelaide_complete()