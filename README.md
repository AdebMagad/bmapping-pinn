
# PINN-UKF-SLAM

Physics-informed magnetic-field localization using a Physics-Informed Neural Network (PINN) map and an Unscented Kalman Filter (UKF) estimator.

This repository accompanies the manuscript on magnetic-field-based localization in GPS-denied environments. It contains the trained PINN model, result files, saved workspaces, figures, and supporting materials used to reproduce the main results reported in the paper.

------------------------------------------------------------------

## 📁 Repository Overview

```text
PINN-UKF-SLAM/
├── figs/                    # Figures used in the manuscript
├── saved_models/            # Trained PINN model(s)
├── Workspace.mat            # Saved workspace from simulation runs
├── pinn_ukf_results.mat     # Main result data
├── rmse_UKF.csv             # Position RMSE data
├── rmse_UKF_q.csv           # Orientation RMSE data
├── rmse_ekf_gpr.mat         # Baseline GPR-EKF results
├── rmse_ukf_gpr.mat         # Baseline GPR-UKF results
├── timespent.mat            # Runtime information
├── requirements.txt         # Python dependencies
└── README.md
```

------------------------------------------------------------------

## 🚀 What This Repository Provides

* A trained PINN magnetic-field model for approximating the ambient magnetic field map
* A UKF-based localization framework that fuses magnetic measurements and odometry
* Saved numerical results for the main simulation study
* Baseline comparison results against GPR-based methods
* Figures and data files supporting the manuscript

-------------------------------------------------------------------

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Main dependencies:

* Python 3.x
* TensorFlow 2.8.2
* NumPy
* SciPy
* Matplotlib

---

## ▶️ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AdebMagad/PINN-UKF-SLAM.git
cd PINN-UKF-SLAM
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify trained model

Ensure the trained PINN model exists in:

```bash
saved_models/
```

## 📊 Reproducing the Main Results

``````bash
Run pinn_ukf_extended.py to reproduce the proposed PINN-UKF framework baseline results
Run map_update.py to reproduce results including the online map update with continual learning
Run PINN_Training.py to train the PINN model offline
Run magnetic_field_plots.py to reproduce the magnetic field prediction results.
Run pinn_ukf_real_data and utilize the /datasets folder to reproduce the experimental results.

```

## 📖 Citation

```bibtex
@article{magad_pinn_ukf_slam,
  title   = {Improving the Localization of the Magnetic Field-based SLAM Through a Novel PINN-UKF Framework},
  author  = {Adeb A. Magad and Muhammad F. Emzir},
  journal = {IEEE Access},
  year    = {2026}
}
```

---

## 📬 Contact

**Adeb A. Magad**
📧 [adeb.magad@gmail.com](mailto:adeb.magad@gmail.com)


