# Two-View-Motion-Segmentation-with-Semi-Calibrated-Cameras-via-Hypothesis-Filtering

This repository contains the official supplementary code for the paper: 
**"Two-View-Motion-Segmentation-with-Semi-Calibrated-Cameras-via-Hypothesis-Filtering"**.

The code is provided to allow reviewers to fully reproduce the quantitative and qualitative results presented in the paper. It includes the continuous Tanimoto formulation, the deterministic Min-Heap optimization (via Numba), and the focal length consensus filtering (KDE).

---

## Repository Structure

To ensure maximum readability and code reuse, the repository is organized into a modular architecture:

### Core Modules (Algorithm & Math)
All core methods are thoroughly commented with direct cross-references to the Equations and Algorithm 1 in the main paper.
* `semicalibrated_tlinkage.py`: Implements the Continuous Preference representation and the Min-Heap agglomerative clustering (Section 4.1, Step 4).
* `geometry.py`: Contains the Iteratively Reweighted Least Squares (IRLS) refinement, focal length estimation, and KDE consensus (Section 4.1, Steps 1-3).
* `metrics.py`: Implements the evaluation metrics (Misclassification Error, Purity, FNR, Frobenius Norm) and the plotting utilities (Section 5.3).

### Preprocessing Scripts (Data Preparation)
* `adelaide_preproc.py`: Extracts images and point matches from the raw AdelaideRMF `.mat` files, filters out invalid data, and formats them into the required directory structure.
* `kitti_preproc.py`: Extracts SIFT features, performs matching, and maps the segmentation ground truth from the raw KITTI 2015 images into `.npz` files compatible with our pipeline.

### Evaluation Scripts (Orchestrators)
* `eval_hopef.py`: Runs the evaluation pipeline on the HOPE-F test set.
* `eval_adelaide.py`: Runs the evaluation pipeline on the AdelaideRMF dataset.
* `eval_kitti.py`: Runs the evaluation pipeline on the KITTI (Autonomous Driving) dataset.

---

## Environment Setup & Dependencies

We highly recommend using a fresh virtual environment (e.g., Conda) with Python 3.9+ to run the code.

### 1. Standard Python Packages
Install the standard dependencies via the provided requirements file:
`pip install -r requirements.txt`

### 2. PoseLib Installation (Crucial Step)
Our pipeline uses `PoseLib` for robust focal length estimation under the semi-calibrated assumption. Because it relies on C++ bindings, **it must be compiled from source without build isolation**.
Please run the following commands sequentially in your terminal:
`git clone https://github.com/vlarsson/PoseLib.git`
`cd PoseLib`
`pip install --no-build-isolation .`
`cd ..`

*Note: If `PoseLib` is not installed, the code will not crash but will automatically bypass the focal filtering step (falling back to standard unconstrained T-Linkage) and print a warning.*

---

## Datasets & Downloads (Note for Reviewers)

To comply with the strict file size limits for supplementary materials, we have included only a small **"Toy Dataset"** (a few sample scenes) in the zipped repository to verify code execution. 

To evaluate the full datasets, please download them from their respective sources and process them using our provided scripts:

* **HOPE-F**: 
  1. Download at [https://github.com/fkluger/hope-f].
  2. Extract directly to the `dataset/` folder. No preprocessing required.

* **AdelaideRMF**: 
  1. Download the complete dataset (`.mat` files) at [https://www.ai4space.group/research/adelaidermf].
  2. Run `python adelaide_preproc.py` to automatically build the `dataset2/AdelaideRMF_Ready/` directory required by the evaluation script.

* **KITTI**: 
  1. Download the KITTI Flow 2015 dataset at [https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015].
  2. Run `python kitti_preproc.py` to extract features and generate the `dataset_kitty/dataset_processed/` directory.

---

## How to Run the Code

Once the datasets are ready (or if you are using the provided Toy Dataset), simply run the evaluation script corresponding to the desired dataset from the root directory:

`python eval_hopef.py`
`python eval_adelaide.py`
`python eval_kitti.py`

### Expected Output
1. **Terminal Logs**: The console will display real-time progress, including the number of sampled models, discarded hypotheses (via KDE filter), T-Linkage execution time, and scene-specific metrics (ME, ACC, FNR, etc.).
2. **Global Summary**: At the end of the batch, the script will output the Mean ± Std of all metrics across the dataset.
3. **Output Folders**: A new directory (e.g., `output_hopef_test_focal_tlinkage/`) will be generated. Inside, you will find a subfolder for each scene containing the qualitative visual results (`clusters_view1.png` and `clusters_view2.png`), matching the figures in the paper.

---

## Reproducibility Note
To ensure full reproducibility of the hypothesis generation phase, the random seed (`RNG_SEED`) is strictly fixed in all evaluation scripts. Executing the code will yield the exact same Misclassification Error values reported in the paper's tables.
