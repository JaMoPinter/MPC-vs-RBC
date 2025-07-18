import pandas as pd
from datetime import timedelta
from typing import Optional


class ForecastWindow:
    """
    Class to obtain a single forecast window of a specific building that is valid for a single MPC iteration.
    """

    def __init__(self, df: pd.DataFrame, mpc_horizon: int, valid_until: pd.Timestamp = None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the forecasted data.
            mpc_horizon (int): Length of the MPC horizon in minutes.
            valid_until (pd.Timestamp): Timestamp until which the forecast is valid. If None, the forecast is valid indefinitely.
        """

        self.df = df.copy()
        self.mpc_horizon = mpc_horizon
        self.valid_until = valid_until


    def slice(self, t_now: pd.Timestamp) -> Optional[pd.DataFrame]:
        """ Slice the dataframe to return the forecast starting from t_now with the length of the MPC horizon. Returns None if the forecast is expired. """

        # If t_now is beyond the validity window, indicate expiration
        if t_now >= self.valid_until:
            return None
        
        # Otherwise, return the slice of the DataFrame starting from t_now with the length of the MPC horizon
        print(f"Forecast slice: {self.df.loc[t_now:t_now + timedelta(hours=self.mpc_horizon)].iloc[:-1]}")
        return self.df.loc[t_now:t_now + timedelta(hours=self.mpc_horizon)].iloc[:-1] # .iloc[:-1] => Exclude the last row to avoid including the next timestamp which is not part of the current MPC iteration

