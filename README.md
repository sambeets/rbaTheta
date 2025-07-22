# <h1> rbaTheta: Ramping Behaviour Analysis

**Definition**: A sudden change in time-varying data is termed as an event. For example, a day (24 h) can be summed up with a few
events rather than 24 individual points. This algorithm identifies such events and classifies them into **stationary** or **significant** events.
An abrupt change is understood as a **significant** event, while a persistent flat interval is classified as **stationary**. The rest of the data carry little
importance in decision-making contexts.

![rbaTheta](/plots/publication_figures/RBAevents_MCMC_0_150.png?raw=true)

---

### 🗂 Directory Organization

| Directory / File                       | Description                                                                 |
| ------------------------------------- | --------------------------------------------------------------------------- |
| `core/`                               | Contains the full event detection pipeline and supporting scripts.          |
| ├── `event_extraction.py`            | Core event segmentation logic (updated)                                     |
| ├── `helpers.py`                     | Utility functions for signal processing and support                         |
| ├── `model.py`                       | Enhanced model interface with parameter control and dynamic tuning          |
| ├── `sensitivity.py`                 | Experiments on threshold robustness and sensitivity analysis                |
| ├── `database.py`                    | SQLite-based event logging and signal trace management                      |
| ├── `classic_model.py` | Original RBAθ implementation from the cited literature             |
| ├── `cusum_method.py`       | CUSUM-based ramp detection              |
| ├── `swrt_method.py`        | SWRT method                        |                               |
| `input_data/`                         | Contains test Excel files  shapefiles                     |
| `plots/`                              | Figures and scripts to visualize output events                              |                             |
| `simulations/`                        | Outputs will be saved in "all_tests_together" after simulation          |
| `main.py`                             | Unified execution script — upgraded with full control and visualization     |
| `metric_comparison.py`               | 📊 Primary script to compare performance metrics across all methods         |

---

### ▶ How to Run

To run an experiment:

`python main.py`

It calls the enhanced model and helper functions from the core/ directory and executes them using multiprocessing.

For metric comparison of all methods after the simulation:

`python metric_comparison.py`

Upon completion, the console will display execution time and generate plots and comparison summaries.

### Environment Setup

Export cross-platform environment (from Windows):

```conda env export --no-builds | findstr -v "prefix" > rba_non-spatial_environment.yml```

Create a new conda environment:
```
conda env create --name m_rba -f rba_non-spatial_environment.yml
conda activate m_rba
```

### Citation and License

Please cite the below publication if you use this repository. The code is released under the MIT License, meaning users are free to use and modify it with explicit citation or written permission.

[Mishra S, Ören E, Bordin C, Wen F, Palu I. Features extraction of wind ramp events from a virtual wind park. *Energy Reports*. 2020 Nov 1;6:237–49.](https://doi.org/10.1016/j.egyr.2020.08.047)

```
@article{mishra2020features,
  title={Features extraction of wind ramp events from a virtual wind park},
  author={Mishra, Sambeet and {\"O}ren, Esin and Bordin, Chiara and Wen, Fushuan and Palu, Ivo},
  journal={Energy Reports},
  volume={6},
  pages={237--249},
  year={2020},
  publisher={Elsevier}
}
```
