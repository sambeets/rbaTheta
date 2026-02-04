# <h1> PredRBAθ: Adaptive RBAθ Event Prediction Method for Wind Turbine Monitoring Using Agentic and Advanced Signal Processing Techniques

**Definition**: A sudden change in time-varying data is termed as an event. For example, a day (24 h) can be summed up with a few
events rather than 24 individual points. This algorithm identifies such events and classifies them into **stationary** or **significant** events.
An abrupt change is understood as a **significant** event, while a persistent flat interval is classified as **stationary**. The rest of the data carry little
importance in decision-making contexts.

![PredRBATheta](/plots/zoomed_events.png?raw=true)

---

PredRBAθ extends the classical RBA-theta algorithm with:

- Enhanced Event Detection (Existing)
- Event Prediction(Extended Development)

### 🗂 Directory Organization
```
PredRBATheta/
├── core/                       # Enhanced RBA-Theta core modules
├── workflows/                  # Event prediction workflows
├── input_data/                 # Datasets
├── simulations/                # Event detection outputs
│   └── all_tests_together/     # Consolidated outputs after simulation
├── main_agentic.py             # MAIN orchestrator – unified execution script
├── bootstrap_experience.py     # Creates synthetic prior-experience database
│                                # used by main_agentic.py
├── event_detector.py           # Event detection workflow (6 methods)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

### ▶ How to Run

1. Environment Setup
Option A: Using Conda (Recommended)

```
conda env create --name pred_rba -f rba_non-spatial_environment.yml
conda activate pred_rba
```
Option B: Using pip

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run "bootstrap_experience.py"
```
python bootstrap_experience.py
```
for first time run, choose option 1 after running bootstrap.

2. Run the Main Orchestrator

```
python main_agentic.py
```

Interactive prompts will guide you through:

Mode Selection:

1 - Event Detection (Analyze historical events)
2 - Event Prediction (Forecast future events)

Dataset Selection:

Enter path to your Excel file (e.g., input_data/Baltic_Eagle.xlsx)

The current input_data folder is empty, so need to include your own data in there. Also, you need to run full end-to-end training on a dataset first, and then you can use transfer learning-zero shot on some other data saved in the same input_data folder. It was neeeded to adjust the space to upload in the Github. When you run main_agentic.py, it will inspect the dataset characteristics and suggest you to run any one of the workflows. I suggest you run all workflows once. The agentic system learns over time so the more you run different workflows on different datasets, the more the system becomes intelligent.

Configuration (varies by mode)

### Version History

This is version 0.3.0, i.e, the PredRBAθ framework. Previous implementations can be found in release 0.2.0 and 0.1.0 contains the original RBAθ framework.

### 🤝 Contribution
This is a Master's thesis research project. For collaboration or questions:

Author: Purbak Sengupta
Supervisor: Prof. Sonal Shreya
Co-Supervisor: Prof. Sambeet Mishra
Institution: Aarhus University
### Citation and License

Please cite the below publications if you use this repository. The code is released under the MIT License, meaning users are free to use and modify it with explicit citation or written permission.

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

[P. Sengupta and S. Mishra, "Enhanced RBAθ method for uncertainty quantification in time varying dataset," 2025 12th International Conference on Electrical and Electronics Engineering (ICEEE), Istanbul, Turkiye, 2025, pp. 314-320.](https://doi.org/10.1109/ICEEE67194.2025.11261961)

```
@inproceedings{sengupta2025enhanced,
  author={Sengupta, Purbak and Mishra, Sambeet},
  booktitle={2025 12th International Conference on Electrical and Electronics Engineering (ICEEE)}, 
  title={Enhanced RBA$\theta$ method for uncertainty quantification in time varying dataset}, 
  year={2025},
  pages={314-320},
  doi={10.1109/ICEEE67194.2025.11261961}
}
```

⭐ If you find this work useful, please consider starring the repository!

Last Updated: January 29, 2026
