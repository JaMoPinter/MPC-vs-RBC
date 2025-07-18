import pandas as pd

from .fc_window import ForecastWindow
from .fc_loader import ForecastLoader


class ForecastManager:
    """ """

    def __init__(self, *,
                 building: str,
                 mpc_freq: int,
                 loader: ForecastLoader):
        """ """

        self.building = building
        self.mpc_freq = mpc_freq

        #self.loader = loader

        # This now is only all the instances we need to load! Just one Building and one MPC frequency
        self.fcs = loader.load(self.building, self.mpc_freq)


        self.mpc_horizon = loader.mpc_horizon
        self.forecast_update_times = loader.forecasts_to_load
        #self.fc = loader.fc

        self._current_window = None
        

    def get_forecast(self, t_now: pd.Timestamp) -> pd.DataFrame:
        """ 
        Return a forecast-dataframe slice from t_now with the length of the MPC horizon. If the previous forecast is not 
        valid anymore, load the next forecast and return the slice from there.
        """

        # If no window exists or its expired -> create a new one
        if (self._current_window is None or t_now >= self._current_window.valid_until):
            
            # Load a new (more recent) forecast
            df, valid_until = self._select_forecast(t_now)
            print(f"Load new forecast at {t_now}")

            # Initialize a new ForecastWindow
            self._current_window = ForecastWindow(
                df=df,
                valid_until=valid_until,
                mpc_horizon=self.mpc_horizon
            )

            
        # Return the slice of the current timestamp
        return self._current_window.slice(t_now)
    

    def _select_forecast(self, t_now: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
        """ Select the most recent forecast dataframe (all entries) based on the current timestamp.
        Returns the dataframe and the Timestamp until which the forecast is valid."""
        
        latest_fc_time = None

        # Get the timestamp of the latest forecast
        for i, fc_time in enumerate(self.forecast_update_times):
            if fc_time > t_now:
                latest_fc_time = self.forecast_update_times[i - 1]
                valid_until = fc_time
                break
        
        if latest_fc_time is None:
            # Get the last forecast time from the loader
            latest_fc_time = self.forecast_update_times[-1]
            print("Using last forecast!")
            valid_until = t_now  # Check if this does not cause any issues. Should be valid for the remainder of the optimization
        
        
        df = self.fcs[latest_fc_time]
        return df, valid_until