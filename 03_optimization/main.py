import json
import os
import mlflow
import pandas as pd
from datetime import datetime, timedelta

from input_manager.fc_loader import ForecastLoader
from input_manager.fc_manager import ForecastManager

from input_manager.gt_manager import GroundTruthManager
from input_manager.price_manager import PriceManager

from optimization.registry import OptimizerRegistry

from utils import map_costs_to_timestamps

from optimization.simulation_engine import SimulationEngine


def main(config_path: str):
    """
    Pipeline entry point for running the optimization.
    
    Args:
        config_path (str): Path to the configuration file.
    """

    # Load config file
    with open(config_path, 'r') as f:
        config = json.load(f)


    # Create run folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # run_folder = f'runs/{timestamp}'
    # os.makedirs(run_folder, exist_ok=True)

    # Save a copy of the config for reproducibility
    # with open(os.path.join(run_folder, "config.json"), "w") as f:
    #     json.dump(config, f, indent=2)


    # Load ground truth data
    gt_manager = GroundTruthManager(config)

    # Load price data
    price_manager = PriceManager(filename=config['optimization']['prices'])


    # Load building forecasts
    fc_loader = ForecastLoader(config)
    fc_loader.validate_config()







    # TODO: Add MLflow tracking
    # TODO: Add evaluation parts. Here it is important, that it executes directly after each building run for mlflow tracking. But also
    #       that it can be run separately to evaluate the results of a previous run and visualize the results again!


    results = []  
    for opt_name in config['optimization']['models']:
        print(f"\n=== Running optimization model: {opt_name} ===\n")
        
        OptClass = OptimizerRegistry.get_optimizer(opt_name)


        for mpc_freq in config['optimization']['mpc_update_freq']:
            print(f"\n=== Running optimization with MPC frequency: {mpc_freq} minutes ===\n")




            for b in config['optimization']['buildings']:
                print(f"\n=== Running simulation for building: {b} ===")

                prices = price_manager.get_prices(mpc_freq)
                gt_full = gt_manager.get_gt(b)
                gt_delta = gt_manager.gt_freq  # Use the GT frequency from the manager


                optimizer = OptClass(battery_cfg=config['battery'], mpc_freq=mpc_freq, param_assumption=config['forecasts']['parametric_assumption'], prices=prices)
                fc_manager = ForecastManager(building=b, mpc_freq=mpc_freq, loader=fc_loader)


                sim = SimulationEngine(
                    optimizer    = optimizer,
                    fc_manager   = fc_manager,
                    gt_full      = gt_full,
                    gt_delta     = gt_delta,
                    building     = b,
                    mpc_freq     = mpc_freq
                )

                df_run = sim.run(start=fc_loader.start_time, end=fc_loader.end_time)
                results.append(df_run)

                os.makedirs('03_optimization/runs', exist_ok=True)
                out_path = f"03_optimization/runs/logs_{timestamp}_{opt_name}_op-freq-{mpc_freq}.csv"
                df_run.to_csv(out_path, index=False)




  

if __name__ == "__main__":
    import sys
    main(sys.argv[1])


