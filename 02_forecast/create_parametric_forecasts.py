
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from typing import List
from pathlib import Path

import numpy as np
import os
import sys
import re
import random


class ParametricForecasts:
    def __init__(self):
        """
        Initialize the ParametricForecasts class. This class is designed to handle the loading, processing, 
        and storage of parametric forecasts.
        """
        self.quantile_forecasts = None
        self.param_forecasts = None
        self.csv_path = None
        self.implemented_distributions = ['sum2gaussian']

        self.seed = 42
        np.random.seed(self.seed)
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)


    def load_quantile_forecasts(self, csv_path, timerange=None, chunksize=200_000, include_ptot=False, downcast=True):

        self.csv_path = csv_path

        # Only read specific columns from disk
        def _usecols(c: str) -> bool:
            if c in ("timestamp", "time_fc_created"):
                return True
            if c.startswith("quantile_"):
                return True
            if include_ptot and c == "P_TOT":
                return True
            return False
        
        parse_dates = ['timestamp', 'time_fc_created']

        iterator = pd.read_csv(
            csv_path,
            usecols=_usecols,
            parse_dates=parse_dates,
            memory_map=True,
            chunksize=chunksize,
            low_memory=True
        )

        if timerange is not None:
            start = pd.to_datetime(timerange[0])
            end   = pd.to_datetime(timerange[1])
        else:
            start, end = None, None

        frames: List[pd.DataFrame] = []

        for chunk in iterator:
            # optional timerange filtering
            if timerange is not None:
                mask = (chunk['time_fc_created'] >= start) & (chunk['time_fc_created'] <= end)
                if not mask.any():
                    continue
                chunk = chunk[mask]

            # strip 'quantile_' prefix
            new_cols = {
                c: (c.replace("quantile_", "") if c.startswith("quantile_") else c)
                for c in chunk.columns
            }
            chunk.rename(columns=new_cols, inplace=True)

            keep_cols = [c for c in chunk.columns if _is_float_like(c)]
            if include_ptot and "P_TOT" in chunk.columns:
                keep_cols.append("P_TOT")

            # set multi-index
            chunk.set_index(['time_fc_created', 'timestamp'], inplace=True)

            # select quantiles (sorted numerically)
            qcols = sorted([c for c in keep_cols if c != "P_TOT"], key=float)
            chunk = chunk[qcols + (["P_TOT"] if include_ptot and "P_TOT" in keep_cols else [])]

            # downcast to float32 if desired
            if downcast:
                chunk[qcols] = chunk[qcols].astype("float32", copy=False)
                if include_ptot and "P_TOT" in chunk:
                    chunk["P_TOT"] = chunk["P_TOT"].astype("float32", copy=False)
            frames.append(chunk[keep_cols])

        if not frames:
            raise ValueError("No data found in the specified timerange or file.")
        
        df = pd.concat(frames, axis=0)
        df.index.set_names(['time_fc_created', 'timestamp'], inplace=True)

        # keep only quantiles in self.quantile_forecasts; drop P_TOT here to save RAM
        qcols = [c for c in df.columns if _is_float_like(c)]
        self.quantile_forecasts = df[qcols]

        

    def load_parametric_forecasts(self, csv_path, name='sum2gaussian'):
        """ Load parametric forecasts from the specified path. """

        df = pd.read_csv(
            csv_path,
            parse_dates=['timestamp', 'time_fc_created'],
            index_col='timestamp'
        )

        # Use multi-index for the DataFrame
        df.set_index(['time_fc_created', df.index], inplace=True)

        return df


    def compute_expected_values(self, df, name):

        # TODO: Recode that name is a class .self variable

        # For now just code the expected value for the sum2gaussian distribution
        if name == 'sum2gaussian':
            for t, row in df.iterrows():
                mu1 = row['mu1']
                mu2 = row['mu2']
                w1 = row['w1']
                w2 = row['w2']
                # Compute the expected value as a weighted sum
                expected_value = w1 * mu1 + w2 * mu2
                df.at[t, 'expected_value'] = expected_value

        return df



    def fit_distribution(self, name):
        """ Fit the specified distribution to the quantile forecasts. """
        if self.quantile_forecasts is None:
            raise ValueError("Quantile forecasts not loaded. Call load_quantile_forecasts() first.")
        
        if name not in self.implemented_distributions:
            raise ValueError(f"Distribution '{name}' is not implemented. Available distributions: {self.implemented_distributions}")
        
        if name == 'sum2gaussian':
            self.fit_sum2gaussian()

    
    def fit_sum2gaussian(self):
        ''' Fit a sum of two Gaussian distributions to the quantile forecasts. '''

        idx = self.quantile_forecasts.index
        param_cols = ['w1', 'mu1', 'std1', 'w2', 'mu2', 'std2']
        self.param_forecasts = pd.DataFrame(index=idx, columns=param_cols)
        np.random.seed(self.seed)

        total_rows = len(self.quantile_forecasts)
        processed = 0
        last_percent = -1  # to avoid printing too often

        for created_time, group in self.quantile_forecasts.groupby(level='time_fc_created'):

            df_quantiles = group.drop(columns=['P_TOT'], errors='ignore')
            quantile_probabilites = df_quantiles.columns.astype(float)

            for t, quants in df_quantiles.iterrows():
                # Step 1: Create an interpolator to map continuous probabilities to continuous values
                inv_cdf = interp1d(quantile_probabilites, quants, kind='linear', fill_value='extrapolate')

                # Step 2: Generate synthethic samples from the inverse CDF via interpolation
                synthetic_probs = np.random.uniform(0.0, 1.0, 800)
                synthetic_values = inv_cdf(synthetic_probs)

                # Step 3: Fit Gaussian Mixture Model (GMM) to the synthetic samples
                gmm = GaussianMixture(n_components=2, random_state=self.seed, covariance_type='full')
                gmm.fit(synthetic_values.reshape(-1, 1))

                # Step 4: Store the GMM parameters in a DataFrame
                self.param_forecasts.loc[t, :] = [
                    gmm.weights_[0], gmm.means_[0, 0], np.sqrt(gmm.covariances_[0, 0, 0]),
                    gmm.weights_[1], gmm.means_[1, 0], np.sqrt(gmm.covariances_[1, 0, 0])
                ]

                # Progress update (every 1%)
                processed += 1
                percent = int(100 * processed / total_rows)
                if percent != last_percent and percent % 1 == 0:
                    print(f"\rProgress: {percent}% ({processed}/{total_rows})", end="")
                    sys.stdout.flush()
                    last_percent = percent



    def store_parametric_forecasts(self, creation_time=False):
        """ Store the parametric forecasts to a CSV file. If the file already exists, a counter is appended to the filename.
        
        Args:
            creation_time (bool): If True, add the time of execution to the filename.
        """
        if self.param_forecasts is None:
            raise ValueError("Parametric forecasts not generated. Call fit_distribution() first.")

        # Create the filename and directory based on the original CSV path
        filename = os.path.basename(self.csv_path).replace('file_fc', 'file_fc_parametric')
        if creation_time:
            # add current timestamp to the filename
            current_time = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')
            filename = filename.replace('.csv', f'_CreationTime{current_time}.csv')
        directory = os.path.dirname(self.csv_path).replace('storage_quantile_fc', 'storage_param_fc')
        #directory = '/home/ws/fh6281/GermanBuildingDate/02_forecast'

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Prepare the filepath
        filepath = os.path.join(directory, filename)
        base, ext = os.path.splitext(filename)
        counter = 1

        # Find the '_SFH' pattern in the filename (e.g., 'file_fc_SFH3_2024-07-25.csv')
        pattern = r'(_SFH\d+)'
        match = re.search(pattern, base)
        sfh_part = match.group(1)

        while os.path.exists(filepath):
            # Insert the counter before _SFH part
            new_base = base.replace(sfh_part, f"{counter:02d}{sfh_part}")
            filepath = os.path.join(directory, f"{new_base}{ext}")
            counter += 1

        # Save to CSV
        self.param_forecasts.to_csv(filepath, index=True)
        print(f"Parametric forecasts saved to {filepath}")


    def sort_quantiles(self):
        """ Sort the quantile columns in ascending order for each forecast (each row). """
        print("Sorting quantiles in ascending order…")

        quant_cols = sorted([c for c in self.quantile_forecasts.columns if _is_float_like(c)], key=float)
        Q = self.quantile_forecasts[quant_cols].to_numpy(copy=False)

        Q_sorted = np.sort(Q, axis=1)
        self.quantile_forecasts.loc[:, quant_cols] = Q_sorted
        return self.quantile_forecasts




    def smooth_quantiles_over_time(self, df, window_size=3):
        """
        For each forecast creation time, smooth quantiles over time and re-sort quantiles at each time.
        Assumes df has MultiIndex ('time_fc_created', 'timestamp') and columns 0.01 to 0.99 (floats or strings).
        """
        print("Smoothing each quantile over time with a rolling mean…")
        quantile_cols = [col for col in df.columns if (isinstance(col, float) or col.replace('.', '', 1).isdigit())]

        smoothed_pieces = []
        # Group by 'time_fc_created'
        for fc_time, group in df.groupby(level='time_fc_created'):
            # Remove first level of index for easier handling
            group = group.droplevel('time_fc_created')
            # Apply rolling mean over time to each quantile column
            smoothed = group[quantile_cols].rolling(window=window_size, min_periods=1).mean()
            # Sort quantiles at each time step (row)
            smoothed = smoothed.apply(np.sort, axis=1, result_type='broadcast')
            # Restore multiindex
            smoothed.index = pd.MultiIndex.from_product([[fc_time], smoothed.index], names=['time_fc_created', 'timestamp'])
            smoothed_pieces.append(smoothed)

        smoothed_df = pd.concat(smoothed_pieces).sort_index()
        return smoothed_df

    

    def smooth_quantiles_at_each_time(self, df, window_size=3):
        """ Smooth at each time over all 99 quantiles using a rolling mean. """
        print("Smoothing all quantiles at each time with a rolling mean…")

        fc_smoothed = []
        for idx, row in df.iterrows():
            # row is a single timestamp’s array of quantile values
            smoothed_values = row.rolling(window=window_size, min_periods=1, center=True).mean()
            fc_smoothed.append(smoothed_values.values)
        return pd.DataFrame(fc_smoothed, index=df.index, columns=df.columns)

                
def _is_float_like(x) -> bool:
    try:
        float(x); return True
    except Exception:
        return False
    
        

if __name__ == "__main__":
    time_start = pd.Timestamp.now()
    print("Time start:", time_start)
    # Example usage
    # path_list = [
    #     '/srv/fh6281/GermanBuildingDate/02_forecast/mount/storage_quantile_fc/SFH3/2025-08-13_10-08-01/file_fc_AutoKAN_SFH3_2025-08-13_10-08-01_freq60.csv',
    #     '/srv/fh6281/GermanBuildingDate/02_forecast/mount/storage_quantile_fc/SFH4/2025-08-13_10-08-01/file_fc_AutoKAN_SFH4_2025-08-13_10-08-01_freq60.csv',
    #     '/srv/fh6281/GermanBuildingDate/02_forecast/mount/storage_quantile_fc/SFH9/2025-08-13_14-34-02/file_fc_AutoKAN_SFH9_2025-08-13_14-34-02_freq60.csv'
    # ]

    # get a list of paths of each fc with the same time
    time_of_fc_creation = '2025-08-18_14-43-38'
    path_list = list(Path('02_forecast/mount/storage_quantile_fc/').glob(f'**/file_fc*_{time_of_fc_creation}_freq*.csv'))
    # Sort the path list according to SFH
    path_list.sort(key=lambda p: int(re.search(r"SFH(\d+)", p.name).group(1)))

    print("Found paths:")
    for p in path_list:
        print(p)

    for path in path_list:
        pf = ParametricForecasts()
        print("Start Loading...", path)
        pf.load_quantile_forecasts(path, timerange=None)
        pf.sort_quantiles()
        pf.fit_distribution('sum2gaussian')
        pf.store_parametric_forecasts()

        t_now = pd.Timestamp.now()
        print("Time elapsed:", t_now - time_start, "    t_now:", t_now)

    t_end = pd.Timestamp.now()
    print("Time end:", t_end)
    print("Total time elapsed:", t_end - time_start)
