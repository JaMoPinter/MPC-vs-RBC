#
# File to generate the forecasts 
#

# This script is designed to create quantile forecasts using the NeuralForecaster library

#
# Imports
#


import pandas as pd
from datetime import datetime
import os
import numpy as np
import glob


from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.auto import AutoPatchTST
from neuralforecast.auto import AutoNHITS
from neuralforecast.auto import AutoTSMixerx
from neuralforecast.auto import AutoTiDE
from neuralforecast.auto import AutoKAN

from neuralforecast.models.nhits import NHITS
from neuralforecast.models.patchtst import PatchTST
from neuralforecast.models.tsmixerx import TSMixerx
from neuralforecast.models.tide import TiDE
from neuralforecast.models.kan import KAN

from neuralforecast import NeuralForecast
from ray import tune
from ray.tune.search.optuna import OptunaSearch


#
# Constants
#
random_seed=123456789


BASE_PATH_DATA = 'GermanBuildingData/01_data/prosumption_data'
BASE_RESULTS_PATH = 'GermanBuildingData/02_forecast/tmp_forecast/results'

RESOLUTION = 15  # in min change this to '15', or '60' as needed
HORIZON = int(24*60 / RESOLUTION)  # 24 hours in the future at the given resolution
STEPS_YEAR = int(365 * 24 * 60/ RESOLUTION)  # 1 year in the future at the given resolution<

# LIST OF BUILDINGS
BUILDINGS = [
 'SFH3',
 'SFH4',
 'SFH9',
 'SFH10',
 'SFH12',
 'SFH14',
 'SFH16',
 'SFH18',
 'SFH19',
 'SFH22',
 'SFH27',
 'SFH28',
 'SFH29',
 'SFH30',
 'SFH32',
 'SFH36'
 ]

#
# Methods
#

def data_readin(resolution, building):
    """
    Reads in the data for a specific building and resolution.
    """
    file_path_resolution = os.path.join(BASE_PATH_DATA, f"{resolution}min")
    file_path_building = os.path.join(file_path_resolution, f"*_{building}_*")
    files = glob.glob(file_path_building)
    #
    #
    #
    if not files:
        raise FileNotFoundError(f"No data files found for building {building} at resolution {resolution} min")
    # check not multiple files
    if len(files) > 1:
        raise ValueError(f"Multiple files found for building {building} at resolution {resolution}: {files}")
    


    file_path = files[0]  # Assuming there's only one file per building and resolution
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    data.index = pd.to_datetime(data.index)
    return data



#
# main method
#
if __name__ == "__main__":
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for building in BUILDINGS:

        #
        # Data readin routine
        #

        prosumption_building = data_readin(RESOLUTION, building)
        print(f"Data for building {building} loaded successfully.")

        #
        # adjust the ranges of the data for the specific dataset
        #

        start_date = prosumption_building.index[0]
        end_date = prosumption_building.index[-1]

        # First year of data is used for training
        
        train_start = start_date
        train_end = start_date + pd.DateOffset(years=1)  # 1 year of training data
        train_steps = int((train_end - train_start).total_seconds() / (RESOLUTION * 60))

        # Validation data is the next year after training
        validation_start = train_end + pd.DateOffset(minutes=RESOLUTION)
        validation_end = validation_start + pd.DateOffset(years=1)  # 1 year of validation data 
        val_steps = int((validation_end - validation_start).total_seconds() / (RESOLUTION * 60)) 

        # rest of the data is used for testing
        test_start = validation_end + pd.DateOffset(minutes=RESOLUTION)
        test_end = end_date
        test_steps = int((test_end - test_start).total_seconds() / (RESOLUTION * 60))

        print(f"Training data from {train_start} to {train_end}")
        print(f"Validation data from {validation_start} to {validation_end}")
        print(f"Test data from {test_start} to {test_end}")

        # get the data into form for training and neuralforecaster 

        # Create DataFrame with proper format for neuralforecast
        df = pd.DataFrame({
            'ds': prosumption_building.index,  # Timestamps
            'y': prosumption_building['P_TOT'],  # Scaled target variable
            'unique_id': building  # Unique identifier for different time series
        })
    

        print(f"DataFrame for building {building} created with shape: {df.shape}")

        #
        # Check for missing values
        #

        if df.isnull().values.any():
            print(f"Warning: Missing values found in the data for building {building}. Filling missing values with forward fill method.")
            df.fillna(method='ffill', inplace=True)

        #
        # generate the quantile forecasts
        #

        # Split into train and validation
        df_train = df[df['ds'] <= validation_end]
        len_train = len(df_train)
        print(f"Training data shape: {df_train.shape}")

        # reset index 
        df_train.reset_index(drop=True, inplace=True)

        # Define TFT model with adjusted settings
        quantiles = [np.round(i,2) for i in np.arange(0.01, 1, 0.01)]


        #
        # random configuration for the hyperparameter search
        #

        BATCH_SIZE = 1  # Set a fixed batch size for all models as it only sets the amount of time series in a batch, not the number of time steps

        #
        # AutoPatchTST configuration
        #

        config_patch_tst = AutoPatchTST.default_config.copy()

        config_patch_tst["input_size"] = tune.choice(
                     [HORIZON * x for x in [3, 7]]
                )
        config_patch_tst["learning_rate"] = tune.choice([0.0001])
        config_patch_tst["hidden_size"] = tune.choice([32, 128, 256])
        config_patch_tst["windows_batch_size"] = tune.choice([256, 512, 1024])
        config_patch_tst["n_heads"] = tune.choice([4, 16])
        config_patch_tst["max_steps"] = tune.choice([800])
        config_patch_tst["revin"] = tune.choice([True, False])
        config_patch_tst["batch_size"] = tune.choice([BATCH_SIZE])
        config_patch_tst["step_size"] = tune.choice([1, HORIZON])
        config_patch_tst["random_seed"] = random_seed
        config_patch_tst["val_check_steps"] = tune.choice([10])
        config_patch_tst["early_stop_patience_steps"] = tune.choice([5])
        del config_patch_tst["input_size_multiplier"]

        # AutoNHITS configuration
        config_nhits = AutoNHITS.default_config.copy()
        config_nhits.update({
            "input_size": tune.choice([HORIZON * 3, HORIZON * 7]),
            "max_steps": tune.choice([800]),
            "batch_size": tune.choice([BATCH_SIZE]),
            "windows_batch_size": tune.choice([64, 128]),
            "val_check_steps": tune.choice([10]),
            "early_stop_patience_steps": tune.choice([5]),
            "n_pool_kernel_size": tune.choice([[2, 2, 1], [4, 2, 1]]),
            "n_freq_downsample": tune.choice([[8, 4, 1], [24, 6, 1]]),
            "learning_rate": tune.choice([5e-4, 1e-3]),
            "scaler_type": tune.choice([None, "standard"]),
            "random_seed": random_seed,
        })
        del config_nhits["input_size_multiplier"]


        # AutoTSMixerx
        config_tsmixerx = AutoTSMixerx.default_config.copy()
        config_tsmixerx["input_size"] = tune.choice([HORIZON * 3, HORIZON * 7])
        config_tsmixerx["max_steps"] = tune.choice([800])
        config_tsmixerx["windows_batch_size"] = tune.choice([256, 512, 1024])
        config_tsmixerx["batch_size"] = tune.choice([BATCH_SIZE])
        config_tsmixerx["val_check_steps"] = tune.choice([10])
        config_tsmixerx["early_stop_patience_steps"] = tune.choice([5])
        config_tsmixerx["random_seed"] = random_seed
        del config_tsmixerx["input_size_multiplier"]


        # AutoTiDE
        config_tide = AutoTiDE.default_config.copy()
        config_tide["input_size"] = tune.choice([HORIZON * 3, HORIZON * 7])
        config_tide["max_steps"] = tune.choice([800])
        config_tide["windows_batch_size"] = tune.choice([256, 512, 1024])
        config_tide["batch_size"] = tune.choice([BATCH_SIZE])
        config_tide["val_check_steps"] = tune.choice([10])
        config_tide["early_stop_patience_steps"] = tune.choice([5])
        config_tide["random_seed"] = random_seed
        del config_tide["input_size_multiplier"]

        # AutoKAN
        config_kan = AutoKAN.default_config.copy()
        config_kan["input_size"] = tune.choice([HORIZON * 3, HORIZON * 7])
        config_kan["max_steps"] = tune.choice([800])
        config_kan["windows_batch_size"] = tune.choice([256, 512, 1024])
        config_nhits["windows_batch_size"] = tune.choice([1])
        config_kan["batch_size"] = tune.choice([BATCH_SIZE])
        config_kan["early_stop_patience_steps"] = tune.choice([5])
        config_kan["random_seed"] = random_seed
        del config_kan["input_size_multiplier"]



        # Define models 

        num_samples = 10
        search_algorithm = OptunaSearch(seed=random_seed, metric="loss", mode="min")
       
        models = [
        AutoPatchTST(h=HORIZON,loss=MQLoss(quantiles=quantiles),valid_loss=MQLoss(quantiles=quantiles), num_samples=num_samples, config=config_patch_tst, search_alg=search_algorithm),
        AutoNHITS(h=HORIZON, loss=MQLoss(quantiles=quantiles), num_samples=num_samples, config=config_nhits, valid_loss=MQLoss(quantiles=quantiles), search_alg=search_algorithm),
        AutoTSMixerx(h=HORIZON, loss=MQLoss(quantiles=quantiles),n_series=1, num_samples=num_samples, config=config_tsmixerx, valid_loss=MQLoss(quantiles=quantiles), search_alg=search_algorithm),
        AutoTiDE(h=HORIZON, loss=MQLoss(quantiles=quantiles), num_samples=num_samples, config=config_tide, valid_loss=MQLoss(quantiles=quantiles), search_alg=search_algorithm),
        AutoKAN(h=HORIZON, loss=MQLoss(quantiles=quantiles), num_samples=num_samples, config=config_kan, valid_loss=MQLoss(quantiles=quantiles), search_alg=search_algorithm)
        ]

        # Initialize NeuralForecast with all models
        nf = NeuralForecast(models=models, freq=f'{RESOLUTION}min')
        # Train models on training data
        
        #
        # TODO May include leap year handling in the future
        #

        nf.fit(df_train, val_size=STEPS_YEAR)

        # --- 1. Extract best configs from hyperparameter search results ---
        best_configs = []

        for model in nf.models:
            model_name = model.model.__class__.__name__

            # Get and save all results
            results_df = model.results.get_dataframe()
            results_df = results_df.sort_values("loss", ascending=True)

            output_dir = f"{BASE_RESULTS_PATH}/{building}/{time_stamp}"
            os.makedirs(output_dir, exist_ok=True)

            results_path = f"{output_dir}/results_hyper_{time_stamp}_{building}_{model_name}.csv"
            results_df.to_csv(results_path)

            print(f"Hyperparameter results saved to {results_path}")

            # Extract best config
            best_config = model.results.get_best_result().config
            best_configs.append((model.model.__class__, best_config))

        
        # Loop over a list of (model_class, best_config) tuples
        for model_class, best_config in best_configs:  # best_configs = [(PatchTST, config_patch), (NHITS, config_nhits), ...]
            input_size = best_config["input_size"]
            print(f"Best config for {model_class.__name__}: {best_config}")

            # Initialize model
            model = model_class(**best_config)
            model_name = model.__class__.__name__
            print(f"\nTraining {model_name} for building {building} with input size {input_size}")

            nf = NeuralForecast(models=[model], freq=f'{RESOLUTION}min')
            #nf.fit(df_train, val_size=STEPS_YEAR)

            # Prepare test dataframe with context
            #test_start_with_context = test_start - pd.DateOffset(minutes=int(input_size * RESOLUTION))
            #test_df = df[df['ds'] >= test_start_with_context]

            # Perform cross-validation forecast
            cv_df = nf.cross_validation(df, step_size=1, val_size=val_steps, test_size=test_steps)

            # Adjustments for Janik
            cv_df['ds'] = pd.to_datetime(cv_df['ds'])
            cv_df.rename(columns={
                'unique_id': 'building',
                'ds': 'timestamp',
                'y': 'P_TOT',
                'cutoff': 'time_fc_created'
            }, inplace=True)
            cv_df['time_fc_created'] = cv_df['time_fc_created'] + pd.DateOffset(minutes=RESOLUTION)

            # Rename quantile columns
            columns = list(cv_df.columns)
            quantile_names = [f'quantile_{np.round(i, 2)}' for i in quantiles]
            columns[3:3 + len(quantile_names)] = quantile_names
            cv_df.columns = columns

            # Create results folder
            output_dir = f"{BASE_RESULTS_PATH}/{building}/{time_stamp}"
            os.makedirs(output_dir, exist_ok=True)

            # Save forecast file
            output_file = f"{output_dir}/file_fc_{model_name}_{building}_{time_stamp}_freq{RESOLUTION}.csv"
            cv_df.to_csv(output_file, index=False)

            print(f"Cross-validation results for {model_name} saved to {output_file}")
            print(f"Forecasting completed for {model_name} with input size {input_size} for building {building}.")