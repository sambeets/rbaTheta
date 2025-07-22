# ⚡ rbaTheta+: Adaptive Ramping Behaviour Analysis with Event Filtering and SQLite Indexing

**rbaTheta+** is an enhanced version of the original RBATheta algorithm for detecting ramping events in wind power signals. It introduces adaptive thresholding (traditional and MCMC-based), robust data handling through SQLite, and a post-filtering mechanism to better distinguish stationary from significant events using a sliding window + log-based transformation.

---

## 🔍 Problem Statement

Sudden or gradual variations in time-series data (such as wind energy output) can be categorized into **significant events** (sharp changes) and **stationary events** (persistent but stable). Most real-world decisions require only these events — the rest is noise.

---

## 🧠 Key Enhancements

- ✅ **Adaptive thresholding**: Traditional statistical & MCMC-based parameter selection  
- ✅ **Sliding Window + Log Function**: Improves detection of stationary events  
- ✅ **Post-filtering**: Reduces false positives after primary detection  
- ✅ **SQLite integration**: Efficient data indexing and handling  
- ✅ **Notebook-based evaluation**: F1-score, recall, and visualization of true/false positives

---

## 📁 Directory Overview

| Folder/File | Description |
|-------------|-------------|
| `core/` | Contains model logic (`event_extraction.py` with sliding window, log scaling, etc.) |
| `simulations/` | Includes scripts and test results (e.g., `fast_test.py`) |
| `plots/` | Output event visualizations, including MCMC and traditional comparisons |
| `input_data/` | Original wind time-series and QGIS spatial data |
| `RBA_theta_vis.ipynb` | Main visualization notebook (SQLite + post-filtering) |
| `Visualization_SlidingWindow.ipynb` | Evaluation of stationary event detection logic |

---

## ⚙️ How to Run

### 1. Run the Main Model

```bash
python main.py
```

It extracts events into an Excel file and logs runtime in terminal.

2. Visualize and Evaluate
Run the notebook: RBAtheta_vis_SQLite-Indexing.ipynb 

This will:

Load raw and event data from SQLite

Apply post-filtering

Compute approx recall, F1, TP/FP ratios

Plot detected events

---

🧪 Environment Setup
To replicate the full environment (non-spatial, cross-platform):

```bash
conda env create --name rba_env -f requirements.txt
conda activate rba_env
#Dependencies include simpy, pandas, matplotlib, sqlite3, and numpy.
```

---

## 📝 Citation

**Please cite the below publication as the source. The source code has a MIT liscence meaning users are free to modify and use only with a explicit written permission or citation of the publication.**

> [Mishra S, Ören E, Bordin C, Wen F, Palu I. Features extraction of wind ramp events from a virtual wind park. Energy Reports. 2020 Nov 1;6:237-49.](https://doi.org/10.1016/j.egyr.2020.08.047)

@article{mishra2020features,
  title={Features extraction of wind ramp events from a virtual wind park},
  author={Mishra, Sambeet and {\"O}ren, Esin and Bordin, Chiara and Wen, Fushuan and Palu, Ivo},
  journal={Energy Reports},
  volume={6},
  pages={237--249},
  year={2020},
  publisher={Elsevier}
}