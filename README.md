# Averaging favors MPC: How typical evaluation setups overstate MPC performance for residential battery scheduling


[![](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/Contact-janik.pinter%40kit.edu-orange?label=Contact)](janik.pinter@kit.edu)

This repository contains the Python implementation for the paper:
> Averaging favors MPC: How typical evaluation setups overstate MPC performance for residential battery scheduling <br>
> Authors: Janik Pinter, Maximilian Beichter, Ralf Mikut, Frederik Zahn, and Veit Hagenmeyer <br>
> NOTE: The paper is currently under revision! An updated link to the final version will follow when available!

## Repository Structure
```
.
├── 01_data/
│   ├── price_data/create_price_data.ipynb      # Create price timeseries (import/export) in Time-Of-Use manner
│   ├── load_data.ipynb/                        # Execute to download the data. Create net-load timeseries for 15 buildings.
│   ├── raw_data/                               # Contains zipped original (untouched) data (once load_data.ipynb is executed)
│   └── prosumption_data/                       # Contains net-load data created via load_data.ipynb
│
|
├── 02_forecast/ 
│   ├── XXX.py                                  # XX
│   ├── store_det_forecasts.ipynb               # Stores deterministic forecasts
│   └── mount/                                  # Storage that holds the forecasts
|
│
└── 03_optimization/ 
    ├── configs/                                # Specify paramters to run simulations
    ├── evaluation/                             # Contains evaluation and results of simulations
    ├── input_manager/                          # Contains architectural code that manages prices, forecasts, ground-truth
    ├── optimization/models/                    # Contains investigated battery control models
    ├── optimization/base.py                    # Base class for all battery models
    ├── optimization/registry.py                # Register battery control models
    ├── optimization/simulation_engine.py       # Executes optimization and applies decision to ground-truth iteratively
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
   python -m pip install -r requirements-optimization.txt
   ```
Furthermore, ensure that IPOPT is properly installed. For more information, see
[IPOPT](https://github.com/coin-or/Ipopt)

## Execution
To start an optimization process, select the desired configurations in 03_optimization/configs/test_config.json, and execute main.py
   ```
   python 03_optimization/main.py
   ```

## Reproducibility
### Reproduce Optimization Results
In order to reproduce the results shown in the paper, execute the optimization process with the corresponding parameter file for Case 1, Case 2, or Case 3 specified in main.py. The necessary forecasts are included in the repository.

<br>

### Reproduce Forecasts
In order to reproduce the forecasts, the following steps need to be done:
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


## Funding
This project is funded by the Helmholtz Association under the "Energy System Design" program and the German Research Foundation as part of the Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation"

## License
This code is licensed under the [MIT License](LICENSE).

## First Version
The content of this paper was first published on [Arxiv](https://arxiv.org/abs/2411.12480). If you found this repository over the Arxiv version, we kindly refer you to the revised version available soon.
