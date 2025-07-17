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
from neuralforecast.models.patchtst import PatchTST
from neuralforecast import NeuralForecast
from ray import tune

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
        validation_start = train_end + pd.DateOffset(minutes=RESOLUTION)
        validation_end = validation_start + pd.DateOffset(years=1)  # 1 year of validation data 
        
        # rest of the data is used for testing
        test_start = validation_end + pd.DateOffset(minutes=RESOLUTION)
        test_end = end_date

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

        # reset index
        df_train.reset_index(drop=True, inplace=True)

        # Define TFT model with adjusted settings
        quantiles = [np.round(i,2) for i in np.arange(0.01, 1, 0.01)]


        #
        # random configuration for the hyperparameter search
        #

        config = AutoPatchTST.default_config.copy()
        print(f"Using configuration: {config}")

        config["max_steps"]= tune.choice([5,10,20])# 10, 20, 40, 80])
        config["input_size"] = tune.choice(
                     [HORIZON * x for x in [3, 7, 10]]
                )
        config["step_size"] = tune.choice([1, HORIZON])
        config["random_seed"] = random_seed

        del config["input_size_multiplier"]

        # Define models in this case only one model Auto Patch TST
        models = [
        AutoPatchTST(h=HORIZON,loss=MQLoss(quantiles=quantiles), num_samples=3, config=config)
        ]

        # Initialize NeuralForecast with all models
        nf = NeuralForecast(models=models, freq=f'{RESOLUTION}min')
        # Train models on training data
        
        #
        # TODO May include leap year handling in the future
        #

        nf.fit(df_train, val_size=STEPS_YEAR)
        results = nf.models[0].results.get_dataframe()
        results = results.sort_values('loss', ascending=True)
        # start the prediction
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # create a folder for the results
        os.makedirs(f"{BASE_RESULTS_PATH}/{building}/{time_stamp}", exist_ok=True)
        results.to_csv(f"{BASE_RESULTS_PATH}/{building}/{time_stamp}/results_hyper_{time_stamp}.csv")

        # get the best configuration
        best_config = nf.models[0].results.get_best_result().config
        print(f"Best configuration for building {building}: {best_config}")
    
        input_size = best_config["input_size"]
        # initialize the model with the best configuration
        models = [
            PatchTST(**best_config)
        ]

        nf = NeuralForecast(models=models, freq=f'{RESOLUTION}min')
        # Fit the model on the training data
        nf.fit(df_train, val_size=STEPS_YEAR)


        # get input size for the prediction to forecast
        input_size = best_config["input_size"]
        # Create forecast
        
        model_name = nf.models[0].__class__.__name__


        #
        #
        #       # Get the forecast data for the test set
        #
        #


        #start the prediction for each timestamp in the test data
        print(f"Starting predictions for building {building} at resolution {RESOLUTION} min with horizon {HORIZON}.")

        test_start_with_context = test_start - pd.DateOffset(minutes=int(input_size * RESOLUTION))
        # test df
        test_df = df[df['ds'] >= test_start_with_context]

        cv_df = nf.cross_validation(test_df, step_size=1, n_windows=len(test_df[test_start:test_end-pd.DateOffset(minutes=RESOLUTION)*HORIZON]))

        # make adjustments so that Janik is happy
        cv_df['ds'] = pd.to_datetime(cv_df['ds'])
        # rename the colum unique_id to building
        cv_df.rename(columns={'unique_id': 'building'}, inplace=True)
        # rename the ds column to timestamp
        cv_df.rename(columns={'ds': 'timestamp'}, inplace=True)
        # rename the y column to P_TOT
        cv_df.rename(columns={'y': 'P_TOT'}, inplace=True)
        # increment the cutoff column by 1 resolution
        cv_df['cutoff'] = cv_df['cutoff'] + pd.DateOffset(minutes=RESOLUTION)
        # and rename the column to cutoff to time_fc_created
        cv_df.rename(columns={'cutoff': 'time_fc_created'}, inplace=True)

        # rename the columns to match quantiles
        # rename the 5th collumn to the 104th column with the quantiles 0.01 to 0.99
        #
        columns = list(cv_df.columns)
        quantile_names = [f'quantile_{np.round(i, 2)}' for i in np.arange(0.01, 1, 0.01)]
        columns[3:102]= quantile_names
        cv_df.columns = columns



        # save it to a csv file fc_{model_name}_{building_name}_{timestamp}_{06-00}_freq{resolution}.csv
        cv_df.head(96*28).to_csv(f"{BASE_RESULTS_PATH}/{building}/{time_stamp}/file_fc_{model_name}_{building}_{time_stamp}_freq{RESOLUTION}.csv", index=False)
        # log the results
        print(f"Cross-validation results for building {building} saved to {BASE_RESULTS_PATH}/{building}/{time_stamp}/cv_results_{time_stamp}.csv")