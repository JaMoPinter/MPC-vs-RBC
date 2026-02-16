# Averaging Favors MPC: How Typical Evaluation Setups Overstate MPC Performance for Residential Battery Scheduling


[![](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/Contact-janik.pinter%40kit.edu-orange?label=Contact)](janik.pinter@kit.edu)

This repository contains the Python implementation for the paper:
> [Averaging Favors MPC: How Typical Evaluation Setups Overstate MPC Performance for Residential Battery Scheduling](https://arxiv.org/abs/2510.25373) <br>
> Authors: Janik Pinter, Maximilian Beichter, Ralf Mikut, Frederik Zahn, and Veit Hagenmeyer <br>
> NOTE: The paper is currently under revision! An updated link to the final version will follow when available!

## Repository Structure
```
.
├── 01_data/
│   ├── price_data/create_price_data.ipynb      # Create price timeseries (import/export) in Time-Of-Use manner
│   ├── load_data.ipynb                         # Execute to download the data. Create net-load timeseries for 15 buildings.
│   ├── raw_data/                               # Contains untouched original zipped data created via load_data.ipynb
│   └── prosumption_data/                       # Contains net-load data created via load_data.ipynb
│
|
├── 02_forecast/ 
│   ├── src/create_forecast_deterministic.py    # Creates deterministic forecasts using Komolgorov-Arnold-Networks
│   ├── configs_hyperparameter_space_local.py   # Selected hyperparameters for creating the forecasts
│   ├── store_det_forecasts.ipynb               # Stores deterministic forecasts in the required format
│   └── mount/                                  # Storage that holds the forecasts
|
│
└── 03_optimization/ 
    ├── configs/                                # Specify parameters to run simulations
    ├── evaluation/                             # Contains evaluation and results of simulations
    ├── input_manager/                          # Contains architectural code that manages prices, forecasts, ground-truth
    ├── optimization/models/                    # Contains investigated battery control models
    ├── optimization/base.py                    # Base class for all battery models
    ├── optimization/registry.py                # Register battery control models. Add new battery models here
    ├── optimization/simulation_engine.py       # Executes optimization and applies decision to ground-truth iteratively
    ├── runs/                                   # Storage that contains optimization results. Here, one exemplary result is included.
    ├── utils.py                                # Helper functions
    └── main.py                                 # Execute this file to start simulation with in configs specified specs

```

## Installation
1. Install virtualenv
   ```
   pip install virtualenv
   ```
2. Create a Virtual Environment
   ```
   virtualenv venv
   ```
3. Activate the Virtual Environment
   ```
   source venv/Scripts/activate
   ```
4. Install Packages specified in requirements.txt
   ```
   python -m pip install -r requirements.txt
   ```
Furthermore, ensure that IPOPT is properly installed. For more information, see
[IPOPT](https://github.com/coin-or/Ipopt).

## Execution
To start an optimization process, execute main.py. Note that the desired optimization configurations can be specified in 03_optimization/configs/configs_paper.json.
   ```
   python 03_optimization/main.py
   ```



## Reproducibility
To keep this repository manageable, only the forecasts of a single building (SFH3) are uploaded. Nevertheless, regarding the results, all 15 buildings and performed simulations are included such that all plots and tables listed in the paper can be reproduced without having the need to rerun optimization processes.

### Reproduce Forecasts
Note that forecasts are only necessary if realistic results should be reproduced. If only results of Rule-Based and MPC with ideal forecasts are of interest, this step can be skipped. To reproduce the forecasts, the following steps need to be done:
1. Install corresponding forecasting requirements
   ```
   python -m pip install -r requirements-forecasting.txt
   ```
2. Execute create_quantile_forecasts.ipynb with the following specifications (GPU necessary):
    - The forecast were generated seeded using a system with the following specs, os, python version:
      - **Processor**: Intel 13th Gen Core i9-13900
      - **Memory**: 64 GB RAM
      - **Graphics**: NVIDIA GeForce RTX 3090 (Driver Version: 555.42.02 / CUDA Version: 12.5)
      - **OS**: Ubuntu 22.04.4 LTS
      - **PYTHON**: 3.12.5


### Reproduce Optimization Results
To reproduce the results included in this repository and listed in the paper, execute the optimization process with the configs listed in configs_paper.json. Here, adapt the forecast_creation_time to match the creation time of your forecasts. Adjust forecast_update_frequency, mpc_update_frequency, and ground-truth_frequency accordingly.

<br>



## Funding
This project is funded by the Helmholtz Association under the "Energy System Design" program and the German Research Foundation as part of the Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation".

## License
This code is licensed under the [MIT License](LICENSE).
