

# PINN-UKF-SLAM

This repository accompanies the manuscript on magnetic-field-based localization in GPS-denied environments. It contains the trained PINN model, result files, saved workspaces, figures, and supporting materials used to reproduce the main results reported in the paper.

---

## 📁 Repository Overview

```text
PINN-UKF-SLAM/
├── Scripts/                    # Main Python scripts for simulation 
│   ├── pinn_ukf_extended.py    # Baseline PINN-UKF simulation 
│   ├── map_update.py           # PINN-UKF with online map update 
│   ├── PINN_Training.py        # Offline training of the PINN model
│   ├── magnetic_field_plots.py # Magnetic field prediction and 
│   └── pinn_ukf_real_data.py   # Real-world experimental validation 
├── Results/                    # Saved outputs and evaluation 
│   ├── Workspace.mat           # Saved workspace from simulation 
│   ├── pinn_ukf_results.mat    # Main simulation results
│   ├── rmse_UKF.csv            # Position RMSE data
│   ├── rmse_UKF_q.csv          # Orientation RMSE data
│   ├── rmse_ekf_gpr.mat        # Baseline GPR-EKF results
│   ├── rmse_ukf_gpr.mat        # Baseline GPR-UKF results
│   └── timespent.mat           # Runtime statistics
├── figs/                       # Figures used in the manuscript
├── saved_models/               # Trained PINN model(s)
├── datasets/                   # Real-world datasets for 
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 What This Repository Provides

* A trained PINN magnetic-field model for approximating the ambient magnetic field map
* A UKF-based localization framework that fuses magnetic measurements and odometry
* A continual learning mechanism for online map update
* Saved numerical results for the main simulation study
* Baseline comparison results against GPR-based methods
* Real-world dataset evaluation scripts

---

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

---



## 📊 Reproducing the Main Results

### A. PINN Training (Offline Map Construction)

```bash
python PINN_Training.py
```

---

### B. Magnetic Field Prediction Visualization

```bash
python magnetic_field_plots.py
```

---

### C. PINN-UKF Localization (Baseline)

```bash
python pinn_ukf_extended.py
```

---

### D. PINN-UKF with Online Map Update

```bash
python map_update.py
```

---

### E. Real-World Experimental Validation

```bash
python pinn_ukf_real_data.py
```

Ensure the dataset is located in:

```bash
/datasets/
```

---

## 📂 Loading Saved Results

### Python

```python
from scipy.io import loadmat

data = loadmat("Workspace.mat")
print(data.keys())
```

### MATLAB

```matlab
data = load('Workspace.mat');
whos('-file', 'Workspace.mat')
```

---


## 📖 Citation

```bibtex
@ARTICLE{11510238,
  author={Magad, Adeb A. and Emzir, Muhammad F.},
  journal={IEEE Access}, 
  title={Improving the Localization of the Magnetic Field-based SLAM Through a Novel PINN-UKF Framework}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Filtering;Filters;Kalman filters;Circuits and systems;Nonlinear filters;Location awareness;Mobile communication;Indoor environment;SIMO;Network architecture;PINNs;UKF;Magnetic Field;Filtering;Continual Learning;SLAM},
  doi={10.1109/ACCESS.2026.3690969}}

```

---

## 📬 Contact

**Adeb A. Magad**

📧 [adeb.magad@gmail.com](mailto:adeb.magad@gmail.com)


