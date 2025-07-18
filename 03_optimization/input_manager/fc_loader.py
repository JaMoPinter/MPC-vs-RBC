import pandas as pd
from datetime import timedelta


class ForecastLoader:
    """
    Class to load forecasts and check if all necessary forecasts are available for a specific experiment.
    """

    def __init__(self, config: dict):
        self.config = config
        self.buildings = config['optimization']['buildings']
        self.fc_name = config['forecasts']['name']

        self.start_time = pd.Timestamp(config['optimization']['start_time'])
        self.end_time = pd.Timestamp(config['optimization']['end_time'])
        self.minutes = (self.end_time - self.start_time).total_seconds() / 60

        self.fc_freq = config['forecasts']['fc_update_freq']
        self.mpc_horizon = config['optimization']['mpc_horizon']
        self.mpc_update_freq = config['optimization']['mpc_update_freq']

        self.time_last_fc_iteration = self.end_time - timedelta(minutes=config['forecasts']['fc_update_freq'])
        self.time_last_op_iteration = self.end_time - timedelta(minutes=min(config['optimization']['mpc_update_freq'])) # TODO: Min does not make sense anymore. We need for each different MPC frequency a new forecast!

        self.forecasts_to_load = self._get_forecast_starting_points()


        # TODO: Do we want to load everything at once or might this lead to memory issues?
        # TODO: In any case, it makes sense to cut the forecasts to the necessary time range, so that we do not load unnecessary data.
        # TODO: Rename forecasts to a better name. They should contain the date hour and minute of the start time of the forecast and the frequency!


    
    def _forecast_path(self, building: str, issuance: pd.Timestamp, mpc_freq: int) -> str:
        #filename = f'data/{building}/forecasts/{building}_{self.fc_name}_{issuance.day:02d}-{issuance.month:02d}-{issuance.year}_hour{issuance.hour}.csv'
        filename = f'data/{building}/forecasts/fc_{self.fc_name}_{building}_{issuance.day:02d}-{issuance.month:02d}-{issuance.year}_{issuance.hour:02d}-{issuance.minute:02d}_freq{mpc_freq}.csv'
        return filename





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

        forecasts = {}

        for t in self.forecasts_to_load:
            path = self._forecast_path(building, t, mpc_freq)
            df = pd.read_csv(path, index_col=0, parse_dates=True)

            # Filter the DataFrame to decrease memory usage but still cover roughly the necessary time range
            df = df.loc[t:self.time_last_op_iteration + timedelta(hours=self.mpc_horizon+1)]

            forecasts[t] = df
        print(f"Loaded {len(forecasts)} forecasts for building {building} with MPC frequency {mpc_freq}.")
        print(f"Forecasts for building {building} with MPC frequency {mpc_freq}:\n{forecasts}")
        return forecasts





    def validate_config(self):
        """
        Check if all necessary forecasts are available before running the experiment.

        In detail, this checks if:
            - Each forecasting file for the specified building/MPC frequency exists.
            - Each forecasting file has the correct frequency.
            - Each forecast covers a time range of at least self.mpc_horizon + self.fc_freq.
        """

        for b in self.buildings:
            for mpc_freq in self.mpc_update_freq:
                

                try:
                    fcs = self.load(b, mpc_freq)
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Forecasts for building {b} with MPC frequency {mpc_freq} could not be loaded. Ensure that the necessary files exist.") from e
                
                for t, df in fcs.items():

                    

                    # Check if df covers a time range of at least self.mpc_horizon
                    if (df.index[-1] - df.index[0]) < (timedelta(hours=self.mpc_horizon)):
                        raise ValueError(f'Forecast for building {b} at timestamp {t} does not cover the required time range of {self.mpc_horizon} OP-Horizon hours + {self.fc_freq} Forecasting-Frequency minutes.')

                    # Check if the frequency of the df is at least self.mpc_update_freq
                    if len(df.index) < 2:
                        raise ValueError(f'Forecast for building {b} at timestamp {t} does not have enough data points to determine frequency.')                   
                    actual_freq = (df.index[1] - df.index[0])
                    if actual_freq != pd.Timedelta(minutes=mpc_freq):
                        raise ValueError(f'Forecast for building {b} at timestamp {t} does not have the necessary frequency of {self.mpc_update_freq} minutes. It has an actual frequency of {actual_freq}.')


        print("All forecasts are valid and ready for the experiment.")


    # def _load_all_forecasts(self):
    #     """ Load all forecasts for the specified buildings. """

    #     #self.forecasts_to_load = self._get_forecast_starting_points()
    #     #print('FC timestamps to load:', self.forecasts_to_load)


    #     for b in self.buildings:
    #         self.fc[b] = {}

    #         for t in self.forecasts_to_load:
    #             # Load the necessary forecasts
    #             try:
    #                 path = self._forecast_path(b, t) # pd.Timestamp(self.start_time + timedelta(hours=h)
    #                 #print('timestamp to load:', t)
    #                 df = pd.read_csv(path, index_col=0, parse_dates=True)
    #             except FileNotFoundError:
    #                 #print(f'Forecast file not found for building {b} at hour {h}. Ensure that necessary files exist.')
    #                 raise FileNotFoundError(f'File {path} not found.')

    #             # # Filter the DataFrame => The last forecasts need to go longer than the end time since the MPC should always consider 24 hours
    #             # df_filtered = df.loc[self.start_time:self.time_last_op_iteration]


    #             self.fc[b][t] = df


    # def validate_config(self):
    #     """ Validate if all necessary forecasting information is present in order to run the experiment. """

    #     for b in self.buildings:
    #         for h in self.fc[b]:
    #             df = self.fc[b][h]

    #             # check if df covers a time range of at least self.mpc_horizon + self.fc_freq
    #             if (df.index[-1] - df.index[0]) < (timedelta(hours=self.mpc_horizon) + timedelta(minutes=self.fc_freq)):
    #                 print(df.index[-1] - df.index[0])
    #                 raise ValueError(f'Forecast for building {b} at hour {h} does not cover the required time range of {self.mpc_horizon} OP-Horizon hours + {self.fc_freq} Forecasting-Frequency minutes.')

    #             # check if the frequency of the df is at least self.mpc_update_freq
    #             if len(df.index) < 2:
    #                 raise ValueError(f'Forecast for building {b} at hour {h} does not have enough data points to determine frequency.')
                
    #             if h != self.forecasts_to_load[-1]:
    #                 actual_freq = (df.index[1] - df.index[0])
    #                 if actual_freq > pd.Timedelta(minutes=min(self.mpc_update_freq)):
    #                     raise ValueError(f'Forecast for building {b} at hour {h} does not have the necessary frequency of at least {self.mpc_update_freq} minutes. It needs to be at least {actual_freq}.')
    #             #else:
    #             #    if df.index[-1] < (self.end_time + timedelta(hours=self.mpc_horizon)):
    #             #        raise ValueError(f'FINAL Forecast for building {b} at hour {h} does not cover the required time range of {self.mpc_horizon} OP-Horizon hours + the difference between the last time of the last forecast and the end time of the problem.')




