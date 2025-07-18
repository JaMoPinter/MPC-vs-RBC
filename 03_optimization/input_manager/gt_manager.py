import os
import pandas as pd


class GroundTruthManager:
    """
    Loads fine-resolution ground truth time series for multiple buildings, and provides resampled versions on demand.    
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        buildings : list of str
            List of building identifiers.
        mpc_freqs : list of int
            List of MPC update intervals in minutes.
        """
        self.buildings = config['optimization']['buildings']
        self.mpc_freqs = config['optimization']['mpc_update_freq']
        self.gt_freq = config['optimization']['gt_freq']  # Ground truth frequency in minutes

        self.start_time = pd.Timestamp(config['optimization']['start_time'])
        self.end_time = pd.Timestamp(config['optimization']['end_time'])

        self.mpc_horizon = config['optimization']['mpc_horizon']

        self._dfs = {}
        self._cache = {}  # cache[(building, freq)] -> resampled df






        self._load_gt()  
        self._validate_freq()


    def _load_gt(self):
        # TODO: Resample GT to fit the gt_freq
        # TODO: Cut the GT data such that not too much unnecessary stuff exists
        # TODO: Have some logic check to ensure that the gt freq is lower equal to the mpc_freq
        # TODO: Check that the mpc_freq is a integer multiple of gt_freq
        

        # Load all gt data for each building
        # Filter the data to the relevant time range. Relevant time range goes from start time to end_time + mpc_horizon

        # Load each building's fine-resolution DataFrame
        for b in self.buildings:
            path = f'data/{b}/ground_truth/{b}_ground_truth.csv'

            if not os.path.exists(path):
                raise FileNotFoundError(f"GT file not found for building '{b}': {path}")
            
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # Ensure a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"GT file for '{b}' must have a DatetimeIndex")
            
            self._dfs[b] = df.sort_index()

        # Filter each DataFrame to the relevant time range
        for b, df in self._dfs.items():
            # Ensure the DataFrame covers the full range from start_time to end_time + mpc_horizon
            # Exclude the last hour since it is not needed
            df_filtered = df.loc[self.start_time:self.end_time + pd.Timedelta(hours=self.mpc_horizon - 1)]

            
            self._dfs[b] = df_filtered
        
        


    def _validate_freq(self):
        """
        Ensure mpc_freqs (in minutes) are all multiples of the base frequency.
        """

        # Infer base frequency in minutes from the first building (assumes all use same base frequency)
        freqs = (self._dfs[self.buildings[0]].index.to_series().diff().dropna().unique())
        if len(freqs) != 1:
            raise ValueError(f"GT for '{self.buildings[0]}' has irregular timestamps. Maybe the ground truth does not cover the full optimization horizon.")
        self.base_freq_min = int(freqs[0].seconds // 60)

        for mpc_freq in self.mpc_freqs:
            if mpc_freq % self.base_freq_min != 0:
                raise ValueError(
                    f"mpc_freq={mpc_freq}min is not a multiple of base GT freq={self.base_freq_min}min"
                )
            
    def get_gt(self, building, gt_freq):
        """
        Return the GT for `building` resampled to `gt_freq` minutes.

        Parameters
        ----------
        building : str
            Building identifier.
        gt_freq : int
            gt update interval in minutes.
        """
        if building not in self._dfs:
            raise KeyError(f"Unknown building '{building}'")

        key = (building, gt_freq)

        if key not in self._cache:
            df_fine = self._dfs[building]
            # e.g. '30min', '60min'
            rule = f"{gt_freq}min"  # TODO: Make this via its own parameter from the config

            # use mean and linear interpolation for missing values
            df_rs = (df_fine
                    .resample(rule)    
                    .mean()
                    #.interpolate(method="time")  # Possibly needed for uneven resampling (e.g. 15min to 35min)
                    )
            self._cache[key] = df_rs

        return self._cache[key]
        
        





