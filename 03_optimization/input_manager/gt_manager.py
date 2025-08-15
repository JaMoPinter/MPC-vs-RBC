import os
import pandas as pd
import sys

from utils import load_chunks


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
            num_pv_modules, orientation = self.map_building_to_pv_num_orientation(b)
            path = f'01_data/prosumption_data/1min/prosumption_{b}_num_pv_modules_{num_pv_modules}_pv_{orientation}_hp_1.0.csv'


            df = load_chunks(path, self.start_time, self.end_time + pd.Timedelta(hours=self.mpc_horizon - 1), filter_col='index', parse_dates=['index'], usecols=['index', 'P_TOT'])
            df['P_TOT'] = df['P_TOT'] / 1000.0  # Convert from W to kW

            # Ensure a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"GT file for '{b}' must have a DatetimeIndex")
            
            self._dfs[b] = df.sort_index()

        

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
            
    def get_gt(self, building):
        """
        Return the GT for `building` resampled to `gt_freq` minutes.

        Parameters
        ----------
        building : str
            Building identifier.
        """
        if building not in self._dfs:
            raise KeyError(f"Unknown building '{building}'")

        key = (building, self.gt_freq)

        if key not in self._cache:
            df_fine = self._dfs[building]
            # e.g. '30min', '60min'
            rule = f"{self.gt_freq}min"  # TODO: Make this via its own parameter from the config

            # use mean and linear interpolation for missing values
            df_rs = (df_fine
                    .resample(rule)    # TODO: Check this resampling logic. Be aware of possible timestamp shifts!
                    .mean()
                    #.interpolate(method="time")  # Possibly needed for uneven resampling (e.g. 15min to 35min)
                    )
            self._cache[key] = df_rs

        return self._cache[key]
        
        

    def map_building_to_pv_num_orientation(self, b): # TODO: Use from Utils
        """
        Maps a building string to the number of PV modules and orientation used. Right now, this is a hardcoded mapping.
        Once the GT data is changed (eg. change pv orientation or scaling), this function needs to be adapted accordingly!

        Returns
        -------
        tuple
            (num_pv_modules, orientation) with orientation representing the pv orientation, i.e. 'SOUTH', 'EAST', 'WEST'.
        """

        mapper = {
            'SFH3': (26, 'SOUTH'),
            'SFH4': (30, 'SOUTH'),
            'SFH9': (36, 'SOUTH'),
            'SFH10': (25, 'SOUTH'),
            'SFH12': (21, 'SOUTH'),
            'SFH14': (28, 'SOUTH'),
            'SFH16': (25, 'EAST'),
            'SFH18': (16, 'EAST'),
            'SFH19': (38, 'EAST'),
            'SFH22': (21, 'EAST'),
            'SFH27': (20, 'WEST'),
            'SFH28': (27, 'WEST'),
            'SFH29': (19, 'WEST'),
            'SFH30': (18, 'WEST'),
            'SFH32': (30, 'WEST'),
            # 'SFH36': (XX, 'XXX'),

        }

        if b not in mapper:
            raise ValueError(f"Building '{b}' not found in GT mapping. Available buildings: {list(mapper.keys())}")
        
        num_pv_modules, orientation = mapper[b]
        return num_pv_modules, orientation




