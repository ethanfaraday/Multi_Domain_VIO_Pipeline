# Investigating a Generalised Visual–Inertial Odometry Framework for Multi-Domain Robotic Localisation

This repository contains the implementation and evaluation code used in the thesis **"Investigating a Generalised Visual–Inertial Odometry Framework for Multi-Domain Robotic Localisation" (2025)** by *Ethan Faraday*.  
The project explores how a modular stereo–IMU pipeline performs across **terrestrial, aerial, and underwater** environments, analysing the effect of texture loss, lighting degradation, and feature quality on odometric accuracy.

---

## 🧩 Overview

The framework implements a **stereo visual–inertial odometry (VIO)** system with modular front-end and back-end components.
The ORB backend feature detector was not used for analysis, but was created and is provided in this repository. The datasets are too large to be uploaded to this repository, but a link to a Google Drive containing the compressed folders is provided.
### Pipeline Structure

1. **Synchronisation & Calibration**  
   - Temporal and spatial alignment between stereo cameras and IMU  
   - Intrinsic and extrinsic calibration via OpenCV and motion-capture alignment  

2. **Image Pre-processing**  
   - CLAHE contrast enhancement  
   - Domain transformations (aerial, underwater, photometric degradation)

3. **Feature Extraction & Matching**  
   - SIFT and ORB detectors/descriptors  
   - Epipolar-constrained stereo matching and outlier rejection (RANSAC)

4. **Triangulation & IMU Fusion**  
   - 3D point reconstruction from stereo disparities  
   - Inertial preintegration using GTSAM-based factor graph  
   - Incremental optimisation via iSAM2  

5. **Evaluation**  
   - Absolute Trajectory Error (ATE), Relative Pose Error (RPE), Translational Drift Error (TDE), Endpoint Error (EPE), RMSE/MAE metrics  
   - 2D trajectory plots and per-frame Euclidean error visualisations  

---

## 📁 Project Structure
```text
VisualOdometry/
├── datasets/ # Link to Husky, Aerial, Underwater, ShipHullVinyl datasets
├── src/
│ ├── stereo_vo.py # Front-end visual odometry
│ ├── vio.py # VIO with SIFT, IMU preintegration and factor-graph fusion
│ ├── aerial.py # aerial transformations
│ ├── underwater.py # underwater transformations
│ ├── ORB_backend_implementation.py # ORB Feature Detector Implementation
└── README.md
```
---

## ⚙️ Requirements

- **Python 3.10+**
- **OpenCV** ≥ 4.5  
- **NumPy**, **SciPy**, **Matplotlib**, **Pandas**
- **GTSAM** (Python bindings)
- **argparse**, **tqdm**, **glob**

🚀 Running the Pipeline
1. Run Stereo VIO with IMU Fusion
```bash
python src/vio.py --left data/left --right data/right --output results/
```

##🧠 Key Findings

Texture and feature quality are the dominant factors governing VIO accuracy.

IMU fusion improves short-term stability but cannot remove slow drift without global constraints.

Calibration integrity is crucial for meaningful domain comparisons.

The framework provides a validated baseline for cross-domain odometry research.

## 📘 Thesis Abstract
Visual–inertial odometry (VIO) provides robust motion estimation by fusing visual and inertial data, yet its performance degrades under domain shifts such as low texture or light attenuation.
This work investigates a modular, domain-agnostic framework to evaluate VIO robustness across terrestrial, aerial, and underwater domains.
Through controlled degradation of texture, lighting, and contrast, the study quantifies trajectory accuracy using standard metrics (ATE, RPE, EPE).
Results show that visual texture density dominates overall accuracy, while IMU fusion improves short-term consistency.
The pipeline establishes a reproducible baseline for multi-domain odometry research and highlights key directions for adaptive front-end tuning and cross-sensor calibration.
