import json
import os
import mlflow
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import hashlib


from input_manager.fc_loader import ForecastLoader
from input_manager.fc_manager import ForecastManager
from input_manager.gt_manager import GroundTruthManager
from input_manager.price_manager import PriceManager

from optimization.registry import OptimizerRegistry

from optimization.simulation_engine import SimulationEngine

from evaluation.evaluator import Evaluator
from evaluation.multi_run_evaluator import MultiRunEvaluator

def _sha1_file(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main(config_path: str):
    """
    Pipeline entry point for running the optimization.
    
    Args:
        config_path (str): Path to the configuration file.
    """

    # Load config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

    # Load Loaders and Managers
    gt_manager = GroundTruthManager(config)
    price_manager = PriceManager(filename=config['optimization']['prices'])
    fc_loader = ForecastLoader(config)
    fc_loader.validate_config()

    mlflow.set_experiment(config.get("mlflow_experiment_name"))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(config, tmp, indent=2)
        tmp.flush()
        config_artifact_path = tmp.name

    parent_run_name = f"batch_{timestamp}"
    config_sha = _sha1_file(config_path)
    with mlflow.start_run(run_name=parent_run_name, tags={
        "run_group": timestamp,
        "config_path": str(Path(config_path).resolve()),
        "config_sha": config_sha,
    }) as parent_run:

        # also log as params for easy filtering
        mlflow.log_params({
            "config_sha": config_sha,
            "config_basename": Path(config_path).name,
            "price_source": str(config["optimization"]["prices"]),
        })
        mlflow.log_artifact(config_artifact_path, artifact_path="config")


        runs_dir = Path("03_optimization/runs")
        runs_dir.mkdir(parents=True, exist_ok=True)

        produced_csvs: list[Path] = []

        # Child runs (optimizer x freq x building)
        for opt_name in config['optimization']['models']:
            print(f"\n=== Running optimization model: {opt_name} ===\n")

            OptClass = OptimizerRegistry.get_optimizer(opt_name)

            for mpc_freq in config['optimization']['mpc_update_freq']:
                print(f"\n=== Running optimization with MPC frequency: {mpc_freq} minutes ===\n")

                for b in config['optimization']['buildings']:
                    print(f"\n=== Running simulation for building: {b} ===")

                    # ---- build components for this scenario
                    prices = price_manager.get_prices(mpc_freq)
                    gt_full = gt_manager.get_gt(b)
                    gt_delta = gt_manager.gt_freq  # Use the GT frequency from the manager


                    optimizer = OptClass(battery_cfg=config['battery'], mpc_freq=mpc_freq, gt_freq=gt_delta, param_assumption=config['forecasts']['parametric_assumption'], prices=prices)
                    fc_manager = ForecastManager(building=b, mpc_freq=mpc_freq, loader=fc_loader)


                    # ---- nested Child Run
                    child_name = f"{opt_name}_freq-{mpc_freq}_building-{b}"
                    with mlflow.start_run(run_name=child_name, nested=True,tags={
                            "run_group": timestamp,
                            "optimizer": opt_name,
                            "building": b,
                            "mpc_freq": str(mpc_freq),
                            "config_sha": config_sha,
                        }) as child_run:
                            
                            mlflow.log_params({
                                "optimizer": opt_name,
                                "building": b,
                                "mpc_freq_min": mpc_freq,
                                "param_assumption": config["forecasts"]["parametric_assumption"],
                                "battery_capacity_min_kWh": config["battery"]["capacity_min"],
                                "battery_capacity_max_kWh": config["battery"]["capacity_max"],
                                "battery_eta_charge": config["battery"]["charge_efficiency"],
                                "battery_eta_discharge": config["battery"]["discharge_efficiency"],
                                "horizon_start": str(fc_loader.start_time),
                                "horizon_end": str(fc_loader.end_time),
                            })


                            # ---- simulation
                            sim = SimulationEngine(
                                optimizer    = optimizer,
                                fc_manager   = fc_manager,
                                gt_full      = gt_full,
                                gt_delta     = gt_delta,
                                building     = b,
                                mpc_freq     = mpc_freq
                            )

                            df_run = sim.run(start=fc_loader.start_time, end=fc_loader.end_time)

                            # Attach prices (forwardfill to ensure that 01:14 gets prices from 01:00)
                            df_run['import_price'] = prices['import_price'].reindex(df_run.index, method='ffill')
                            df_run['export_price'] = prices['export_price'].reindex(df_run.index, method='ffill')


                            # ---- evaluate & log metrics
                            ev = Evaluator(df_run, prices)
                            costs_summary = ev.get_costs()
                            print(f"Costs summary for building {b} with model {opt_name} and MPC frequency {mpc_freq}: {costs_summary}")
                            mlflow.log_metrics({**costs_summary})


                            # ---- save results 
                            rid = mlflow.active_run().info.run_id[:8]
                            out_path = runs_dir / f"logs_{timestamp}_op-{opt_name}_freq-{mpc_freq}_building-{b}_run-{rid}.csv"
                            df_run.to_csv(out_path, index=True)


                            produced_csvs.append(out_path)
                            mlflow.set_tag("run_csv", out_path.name)
                            mlflow.log_artifact(str(out_path), artifact_path="runs")


        # ---- Batch-level summary 
        all_csv = Path("03_optimization/runs").glob(f"logs_{timestamp}_*.csv")
        mre = MultiRunEvaluator(run_paths=all_csv)

        print("\n\n==== Pivot table ====\n")
        print(mre.pivot().to_string())

        print("\n\n==== Net-cost leaderboard ====\n")
        print(mre.leaderboard().to_string(index=False))

        batch_csvs = list(runs_dir.glob(f"logs_{timestamp}_*.csv"))
        if batch_csvs:
            mre = MultiRunEvaluator(run_paths=batch_csvs)
            summary_path = runs_dir / f"summary_{timestamp}.csv"
            mre.df.to_csv(summary_path, index=False)
            mlflow.log_artifact(str(summary_path), artifact_path="summary")

    
    # cleanup temp file
    try:
        os.remove(config_artifact_path)
    except OSError:
        pass




if __name__ == "__main__":
    #import sys
    #main(sys.argv[1])

    # debugging mode
    path = "03_optimization/configs/test_config.json"
    main(path)

