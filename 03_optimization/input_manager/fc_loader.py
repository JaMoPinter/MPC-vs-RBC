import sys
import pandas as pd
from datetime import timedelta
import os

from utils import load_chunks

# TODO: Create a base class for the input_managers, so that all the managers have access to the full config easily.

class ForecastLoader:
    """
    Class to load forecasts and check if all necessary forecasts are available for a specific experiment.
    """

    def __init__(self, config: dict):
        self.config = config
        self.buildings = config['optimization']['buildings']
        self.fc_model = config['forecasts']['model']
        self.fc_creation_time = pd.Timestamp(config['forecasts']['fc_creation_time'])
        self.parametric_assumption = config['forecasts']['parametric_assumption']

        self.start_time = pd.Timestamp(config['optimization']['start_time'])
        self.end_time = pd.Timestamp(config['optimization']['end_time'])
        self.minutes = (self.end_time - self.start_time).total_seconds() / 60

        self.op_models = config['optimization']['models']
        self.only_ideal_model = self.op_models == ['ideal'] # Check if only ideal model is used

        self.fc_freq = config['forecasts']['fc_update_freq']
        self.mpc_horizon = config['optimization']['mpc_horizon']
        self.mpc_update_freq = config['optimization']['mpc_update_freq']

        self.time_last_fc_iteration = self.end_time - timedelta(minutes=config['forecasts']['fc_update_freq'])
        self.time_last_op_iteration = self.end_time - timedelta(minutes=min(config['optimization']['mpc_update_freq'])) # TODO: Min does not make sense anymore. We need for each different MPC frequency a new forecast!

        self.forecasts_to_load = self._get_forecast_starting_points()
        print('Forecast timestamps to load:', self.forecasts_to_load)

    
    def _forecast_path(self, building: str, mpc_freq: int) -> str:
        path = f"02_forecast/mount/storage_param_fc/{building}/{self.fc_creation_time.strftime('%Y-%m-%d_%H-%M-%S')}/file_fc_parametric_{self.fc_model}_{building}_{self.fc_creation_time.strftime('%Y-%m-%d_%H-%M-%S')}_freq{mpc_freq}.csv"
        return path


    def _get_forecast_starting_points(self):
        """ Get all the timestamps of the forecasts that need to be loaded. """
        times = []
        t = self.start_time
        while t < self.end_time:
            times.append(t)
            t += timedelta(minutes=self.fc_freq)
        return times



    def load(self, building: str, mpc_freq: int) -> dict:
        """
        Loads all relevant forecasts for a specific building and MPC frequency.
        """

        if self.only_ideal_model:
            print("\nSkipped loading forecasts since only GT is used as Forecasts")
            
            dummy_fc = {}
            for t0 in self.forecasts_to_load:
                t1 = t0 + pd.Timedelta(hours=self.mpc_horizon)
                idx = pd.date_range(start=t0, end=t1, freq=f"{mpc_freq}min", inclusive='left')
                df = pd.DataFrame({"dummy_col": 0}, index=idx)
                df.index.name = 'timestamp'
                dummy_fc[t0] = df
            return dummy_fc


        forecasts = {}
        path = self._forecast_path(building, mpc_freq)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Forecast file not found. Tried to open: {path}")
        
        df = load_chunks(path, self.forecasts_to_load[0], self.forecasts_to_load[-1], filter_col='time_fc_created' , parse_dates=['time_fc_created', 'timestamp'])

        if self.parametric_assumption == 'sum2gaussian':
            # scale the forecast values of mu1, mu2, std1, and std2 to kW
            df['mu1'] = df['mu1'] / 1000.0
            df['mu2'] = df['mu2'] / 1000.0
            df['std1'] = df['std1'] / 1000.0
            df['std2'] = df['std2'] / 1000.0
        else:
            raise ValueError(f"Rescaling for {self.parametric_assumption} is not yet implemented.")

        for t in self.forecasts_to_load:
            # the first index of the DataFrame is the time_fc_created. Make the df to a dictionary with the time_fc_created as key and the DataFrame as value
            forecasts[t] = df.loc[t]

        print(f"Loaded {len(forecasts)} forecast-Dataframes for building {building} with MPC frequency {mpc_freq}.")

        return forecasts



    def validate_config(self):
        """
        Check if all necessary forecasts are available before running the experiment.

        In detail, this checks if:
            - Each forecasting file for the specified building/MPC frequency exists.
            - Each forecasting file has the correct frequency.
            - Each forecast covers a time range of at least self.mpc_horizon + self.fc_freq.
        """

        if self.only_ideal_model:
            print("Skipped forecast validation since only GT is used as Forecasts")
            return None

        for b in self.buildings:
            for mpc_freq in self.mpc_update_freq:
            
                fcs = self.load(b, mpc_freq)

                for t, df in fcs.items():

                    # Check if df covers a time range of at least self.mpc_horizon
                    if (df.index[-1] - df.index[0]) + timedelta(minutes=self.fc_freq) < (timedelta(hours=self.mpc_horizon)):
                        raise ValueError(f'Forecast for building {b} at timestamp {t} does not cover the required time range of {self.mpc_horizon} OP-Horizon hours.')

                    # Check if the frequency of the df is at least self.mpc_update_freq
                    if len(df.index) < 2:
                        raise ValueError(f'Forecast for building {b} at timestamp {t} does not have enough data points to determine frequency.')                   
                    actual_freq = (df.index[1] - df.index[0])
                    if actual_freq != pd.Timedelta(minutes=mpc_freq):
                        raise ValueError(f'Forecast for building {b} at timestamp {t} does not have the necessary frequency of {self.mpc_update_freq} minutes. It has an actual frequency of {actual_freq}.')


        print("All forecasts are valid and ready for the experiment.")
