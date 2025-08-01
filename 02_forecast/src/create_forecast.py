#
# File to generate the forecasts 
#

# This script is designed to create quantile forecasts using the NeuralForecaster library

#
# Imports
#


import pandas as pd
import os
import numpy as np
import glob


from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.auto import AutoPatchTST
from neuralforecast.auto import AutoNHITS
from neuralforecast.auto import AutoTSMixerx
from neuralforecast.auto import AutoTiDE
from neuralforecast.auto import AutoKAN

from neuralforecast import NeuralForecast
from ray.tune.search.optuna import OptunaSearch

# Importing the custom configurations
from configs_hyperparameter_space_local import (
    RESOLUTION,
    HORIZON,
    RANDOM_SEED,
    BASE_PATH_DATA,
    BASE_RESULTS_PATH,
    NUM_SAMPLES,
    TIME_STAMP,
    config_patch_tst,
    config_nhits,
    config_tsmixerx,
    config_tide,
    config_kan
)




# LIST OF BUILDINGS
BUILDINGS = [
 'SFH3',
 #'SFH4',
 #'SFH9',
 #'SFH10',
 #'SFH12',
 #'SFH14',
 #'SFH16',
 #'SFH18',
 #'SFH19',
 #'SFH22',
 #'SFH27',
 #'SFH28',
 #'SFH29',
 #'SFH30',
 #'SFH32',
 #'SFH36'
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
    
    for building in BUILDINGS:

        output_dir = f"{BASE_RESULTS_PATH}/{building}/{TIME_STAMP}"
        os.makedirs(output_dir, exist_ok=True)
        # tuner directory
        tuner_dir = f"{BASE_RESULTS_PATH}/{building}/{TIME_STAMP}/tuner"


        #
        # 0. DATA PREPARATION
        #

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

        print(f"Training data from {train_start} to {train_end} with {train_steps} steps")
        print(f"Validation data from {validation_start} to {validation_end} with {val_steps} steps")
        print(f"Test data from {test_start} to {test_end} with {test_steps} steps")

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
        # 1. Initialize NeuralForecast with models and hyperparameter search and train
        #

   

        # Initialize models with hyperparameter search

        search_algorithm = OptunaSearch(seed=RANDOM_SEED, metric="loss", mode="min")

        # Create a linearly spaced tensor between 0 and 1
        #import torch
        #x = torch.linspace(0, 1, steps=96)
        #        # Quadratic decrease from 1 to 0.5: y = 1 - 0.5 * x^2
        #horizon_weight = 1.0 - 0.75 * x**2
        
        #models = [
        #AutoPatchTST(h=HORIZON,loss=MQLoss(quantiles=quantiles,horizon_weight=x),valid_loss=MQLoss(quantiles=quantiles,horizon_weight=x), num_samples=NUM_SAMPLES, config=config_patch_tst, search_alg=search_algorithm),
        #AutoNHITS(h=HORIZON, loss=MQLoss(quantiles=quantiles,horizon_weight=x), num_samples=NUM_SAMPLES, config=config_nhits, valid_loss=MQLoss(quantiles=quantiles,horizon_weight=x), search_alg=search_algorithm),
        #AutoTSMixerx(h=HORIZON, loss=MQLoss(quantiles=quantiles,horizon_weight=x),n_series=1, num_samples=NUM_SAMPLES, config=config_tsmixerx, valid_loss=MQLoss(quantiles=quantiles,horizon_weight=x), search_alg=search_algorithm),
        #AutoTiDE(h=HORIZON, loss=MQLoss(quantiles=quantiles,horizon_weight=x), num_samples=NUM_SAMPLES, config=config_tide, valid_loss=MQLoss(quantiles=quantiles,horizon_weight=x), search_alg=search_algorithm),
        #AutoKAN(h=HORIZON, loss=MQLoss(quantiles=quantiles,horizon_weight=x), num_samples=NUM_SAMPLES, config=config_kan, valid_loss=MQLoss(quantiles=quantiles,horizon_weight=x), search_alg=search_algorithm)
        #]

        models = [
        AutoPatchTST(h=HORIZON,loss=MQLoss(quantiles=quantiles),valid_loss=MQLoss(quantiles=quantiles), num_samples=NUM_SAMPLES, config=config_patch_tst, search_alg=search_algorithm),
        AutoNHITS(h=HORIZON, loss=MQLoss(quantiles=quantiles), num_samples=NUM_SAMPLES, config=config_nhits, valid_loss=MQLoss(quantiles=quantiles), search_alg=search_algorithm),
        AutoTSMixerx(h=HORIZON, loss=MQLoss(quantiles=quantiles),n_series=1, num_samples=NUM_SAMPLES, config=config_tsmixerx, valid_loss=MQLoss(quantiles=quantiles), search_alg=search_algorithm),
        AutoTiDE(h=HORIZON, loss=MQLoss(quantiles=quantiles), num_samples=NUM_SAMPLES, config=config_tide, valid_loss=MQLoss(quantiles=quantiles), search_alg=search_algorithm),
        AutoKAN(h=HORIZON, loss=MQLoss(quantiles=quantiles), num_samples=NUM_SAMPLES, config=config_kan, valid_loss=MQLoss(quantiles=quantiles), search_alg=search_algorithm)
        ]

        # Initialize NeuralForecast with all models
        nf = NeuralForecast(models=models, freq=f'{RESOLUTION}min')

        
        #
        # TODO May include leap year handling in the future
        #

        df_results = nf.cross_validation(df, step_size=1, val_size=val_steps, test_size=test_steps, n_windows=None)


        #
        # --- 2. Extract best configs from hyperparameter search results and the predictions ---
        # 
        
        best_configs = []

        for model in nf.models:
            model_name = model.__class__.__name__

            # Get and save all results
            results_df = model.results.get_dataframe()
            results_df = results_df.sort_values("loss", ascending=True)

            results_path = f"{output_dir}/results_hyper_{TIME_STAMP}_{building}_{model_name}.csv"
            results_df.to_csv(results_path)

            print(f"Hyperparameter results saved to {results_path}")

            # Extract best config
            best_config = model.results.get_best_result().config
            best_configs.append((model.model.__class__, best_config))


        

        for model in nf.models:
            model_name = model.__class__.__name__
            # Select only the columns for this model (e.g. AutoKAN_0.5, AutoKAN_0.9, ...)
            model_cols = [col for col in df_results.columns if col.startswith(model_name)]

            # Optionally keep an index or time column (e.g. df_results.index or df_results["ds"])
            # Example: if you have a "ds" column
            base_cols = ['cutoff','ds', 'y']

            model_df = df_results[base_cols + model_cols].copy()
            # Rename columns (Janik format)
            model_df.rename(columns={
                'ds': 'timestamp',
                'y': 'P_TOT',
                'cutoff': 'time_fc_created'
            }, inplace=True)

            model_df['timestamp'] = pd.to_datetime(model_df['timestamp'])
            model_df['time_fc_created'] += pd.DateOffset(minutes=RESOLUTION)

            # Rename quantile columns
            columns = list(model_df.columns)
            quantile_names = [f'quantile_{np.round(q, 2)}' for q in quantiles]
            columns[3:3 + len(quantile_names)] = quantile_names
            model_df.columns = columns


            output_file = f"{output_dir}/file_fc_{model_name}_{building}_{TIME_STAMP}_freq{RESOLUTION}_with_qudratic_weights.csv"
            model_df.to_csv(output_file, index=False)

            print(f"Saved results for {model_name} to {output_file}")
