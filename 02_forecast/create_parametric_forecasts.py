
import pandas as pd
from scipy.stats import norm, gaussian_kde
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from typing import Optional, Tuple, Iterable, List

import numpy as np
import os
import sys
import re


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



    # def load_quantile_forecasts2(self, csv_path, timerange=None):
    #     """ Load quantile forecasts from the specified path and group them according to their creation timestamp. 
        
        
    #     Args:
    #         csv_path (str): Path to the CSV file containing quantile forecasts.
    #         timerange (list, optional): A list with two elements specifying the start and end time for filtering the forecasts.
    #     """

    #     self.csv_path = csv_path

    #     # 1) Load CSV, parse timestamps
    #     df = pd.read_csv(
    #         csv_path,
    #         parse_dates=['timestamp','time_fc_created'],
    #         index_col='timestamp'          # if you want 'timestamp' as the DataFrame index
    #     )

    #     # Change the order of the columns to have P_TOT before quantiles
    #     cols = df.columns.tolist()
    
    #     cols = ['P_TOT'] + [col for col in cols if col not in  ['P_TOT']]
    #     df = df[cols]

    #     # remove the quantile_ prefix from the quantile columns
    #     df.columns = df.columns.str.replace('quantile_', '', regex=False)

    #     # 2) Filter by timerange if provided. Keep all rows where 'time_fc_created' is within the specified range.
    #     if timerange is not None:
    #         start_time, end_time = pd.to_datetime(timerange[0]), pd.to_datetime(timerange[1])
    #         df = df[(df['time_fc_created'] >= start_time) & (df['time_fc_created'] <= end_time)]

        
        
    #     # 2) Group by the forecast‐creation time
    #     groups = {
    #         created_time: group.copy()
    #         for created_time, group in df.groupby('time_fc_created')
    #     }
        
    #     self.quantile_forecasts = {}

    #     for created_time, subdf in groups.items():
    #         subdf = subdf.drop(columns=['time_fc_created'])
    #         self.quantile_forecasts[created_time] = subdf


    #     self.quantile_forecasts = pd.concat(self.quantile_forecasts, axis=0, names=['time_fc_created', 'timestamp'])
    #     # drop the 'building' and 'P_TOT' columns from the quantile forecasts
    #     self.quantile_forecasts = self.quantile_forecasts.drop(columns=['P_TOT'])
        

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
        np.random.seed(42)

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
                gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='full')
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

    # def fit_sum2gaussian3(self,
    #                     n_samples: int = 800,
    #                     reg_covar: float = 1e-6,
    #                     random_state: int = 42,
    #                     print_every_percent: int = 5):
    #     """
    #     Fit a 2-Gaussian mixture to each row of quantile forecasts
    #     using inverse-CDF sampling (your original approach), but faster.

    #     Stores columns: ['w1','mu1','std1','w2','mu2','std2']
    #     """
    #     rng = np.random.default_rng(random_state)

    #     # --- 0) Prepare data once ------------------------------------------------
    #     # keep only quantile columns (float-like names)
    #     qcols = sorted(
    #         [c for c in self.quantile_forecasts.columns if _is_float_like(c)],
    #         key=float
    #     )
    #     Qdf = self.quantile_forecasts[qcols]   # already dropped P_TOT earlier
    #     idx = Qdf.index
    #     Q = Qdf.to_numpy(dtype=np.float32, copy=False)     # shape (N, K)
    #     N, K = Q.shape
    #     alpha = np.array([float(c) for c in qcols], dtype=np.float32)  # (K,)

    #     # --- 1) Precompute uniforms once ----------------------------------------
    #     U = rng.random(n_samples).astype(np.float32)       # (n_samples,)

    #     # --- 2) Preallocate params and a single GMM instance --------------------
    #     params = np.empty((N, 6), dtype=np.float32)
    #     gmm = GaussianMixture(
    #         n_components=2,
    #         covariance_type="diag",   # 1D -> diag is enough & slightly faster
    #         reg_covar=reg_covar,
    #         random_state=random_state,
    #         max_iter=200,             # tune if needed
    #         tol=1e-3,                 # tune if needed
    #         warm_start=True           # reuse last solution as init
    #     )

    #     # --- 3) Main loop (no groupby, no iterrows) -----------------------------
    #     processed = 0
    #     next_print = print_every_percent

    #     for i in range(N):
    #         # inverse-CDF sample via fast np.interp (no interp1d object)
    #         x = np.interp(U, alpha, Q[i]).astype(np.float32, copy=False).reshape(-1, 1)

    #         # fit 2-GMM
    #         gmm.fit(x)

    #         # extract params; sort components by mean (deterministic ordering)
    #         w  = gmm.weights_.astype(np.float32, copy=False)
    #         mu = gmm.means_.ravel().astype(np.float32, copy=False)
    #         cov = gmm.covariances_
    #         if cov.ndim == 1:
    #             var = cov.astype(np.float32, copy=False)
    #         elif cov.ndim == 2:
    #             var = cov[:, 0].astype(np.float32, copy=False)
    #         else:
    #             var = cov[:, 0, 0].astype(np.float32, copy=False)
    #         std = np.sqrt(np.maximum(var, 1e-12, dtype=np.float32))

    #         order = np.argsort(mu)
    #         params[i, 0:3] = (w[order[0]], mu[order[0]], std[order[0]])
    #         params[i, 3:6] = (w[order[1]], mu[order[1]], std[order[1]])

    #         # light progress output
    #         processed += 1
    #         pct = (100 * processed) // N
    #         if pct >= next_print:
    #             print(f"\rProgress: {pct}% ({processed}/{N})", end="")
    #             next_print += print_every_percent

    #     print()  # newline after progress

    #     # --- 4) Build the DataFrame once ----------------------------------------
    #     cols = ['w1', 'mu1', 'std1', 'w2', 'mu2', 'std2']
    #     self.param_forecasts = pd.DataFrame(params, index=idx, columns=cols)



    # def fit_sum2gaussian2(
    #     self,
    #     n_rep: int = 2000,            # total replicated points per row
    #     reg_covar: float = 1e-6,      # stabilizes EM on flat quantiles
    #     random_state: int = 42,
    #     progress_every: int = 1       # print every N%
    # ):
    #     """
    #     Fit a 2-Gaussian mixture to each row of quantile forecasts using
    #     *replicated bin midpoints* (no sample_weight).

    #     Stores columns: ['w1','mu1','std1','w2','mu2','std2','expected_value']
    #     """
    #     rng = np.random.default_rng(random_state)

    #     # 1) pick quantile columns and make arrays
    #     qcols = sorted([c for c in self.quantile_forecasts.columns if _is_float_like(c)], key=float)
    #     Q = self.quantile_forecasts[qcols].to_numpy(dtype=float)   # shape: (N, K)
    #     N, K = Q.shape
    #     alpha = np.array([float(c) for c in qcols], dtype=float)   # (K,)

    #     # 2) bin masses and midpoints (include flat tails)
    #     alpha_ext = np.concatenate(([0.0], alpha, [1.0]))          # (K+2,)
    #     mass = np.diff(alpha_ext)                                  # (K+1,), sums to 1
    #     Q_ext = np.concatenate([Q[:, [0]], Q, Q[:, [-1]]], axis=1) # (N, K+2)
    #     midpoints = 0.5 * (Q_ext[:, :-1] + Q_ext[:, 1:])           # (N, K+1)

    #     params = np.empty((N, 6), dtype=float)

    #     def _extract_params(gmm: GaussianMixture):
    #         w = gmm.weights_.copy()
    #         mu = gmm.means_.ravel().copy()
    #         cov = gmm.covariances_
    #         if cov.ndim == 1:
    #             var = cov
    #         elif cov.ndim == 2:
    #             var = cov[:, 0]
    #         else:
    #             var = cov[:, 0, 0]
    #         std = np.sqrt(np.maximum(var, 1e-12))
    #         order = np.argsort(mu)  # sort by mean
    #         return w[order], mu[order], std[order]

    #     processed, last_percent = 0, -1

    #     for i in range(N):
    #         # 3) integer replication counts from masses
    #         raw = mass * n_rep
    #         base = np.floor(raw).astype(int)
    #         frac = raw - base
    #         diff = n_rep - base.sum()

    #         if diff > 0:  # distribute leftover to largest fractions
    #             order = np.argsort(-frac)
    #             base[order[:diff]] += 1
    #         elif diff < 0:  # remove from smallest fractions but keep >=1 if possible
    #             order = np.argsort(frac)
    #             j = 0
    #             to_remove = -diff
    #             while to_remove > 0 and j < base.size:
    #                 idx = order[j]
    #                 if base[idx] > 1:
    #                     base[idx] -= 1
    #                     to_remove -= 1
    #                 else:
    #                     j += 1

    #         # ensure at least 1 replica per bin with nonzero mass
    #         base[(mass > 0) & (base == 0)] = 1

    #         X_rep = np.repeat(midpoints[i], base).reshape(-1, 1)

    #         # 4) fit 2-GMM (diag covariance is enough for 1D)
    #         gmm = GaussianMixture(
    #             n_components=2,
    #             covariance_type="diag",
    #             reg_covar=reg_covar,
    #             random_state=random_state
    #         )
    #         gmm.fit(X_rep)

    #         w, mu, std = _extract_params(gmm)
    #         params[i, :] = [w[0], mu[0], std[0], w[1], mu[1], std[1]]

    #         processed += 1
    #         pct = int(100 * processed / N)
    #         if pct != last_percent and pct % max(1, progress_every) == 0:
    #             print(f"\rProgress: {pct}% ({processed}/{N})", end="")
    #             last_percent = pct

    #     cols = ["w1", "mu1", "std1", "w2", "mu2", "std2"]
    #     self.param_forecasts = pd.DataFrame(params, index=self.quantile_forecasts.index, columns=cols)










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
        directory = '/home/ws/fh6281/GermanBuildingDate/02_forecast'

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

    # def sort_quantiles2(self):
    #     """ Sort the quantile columns in ascending order for each forecast (each row). """
    #     print("Sorting quantiles in ascending order…")

    #     # 1) Identify the quantile columns by whatever naming convention you have:
    #     #    Here you used strings like "0.01", "0.02", … so we test for a leading '0.'
    #     quant_cols = [c for c in self.quantile_forecasts.columns if c.startswith("0.")]

    #     # 2) Extract all the quantile values as a 2D numpy array (shape: n_rows × n_quantiles)
    #     arr = self.quantile_forecasts[quant_cols].values

    #     # 3) Sort each row (axis=1) in-place
    #     sorted_arr = np.sort(arr, axis=1)

    #     # 4) Assign the sorted values back into the same columns
    #     #    .loc[:, quant_cols] works regardless of the MultiIndex on the rows
    #     self.quantile_forecasts.loc[:, quant_cols] = sorted_arr
    #     return self.quantile_forecasts



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

        #df_quantiles = self.quantile_forecasts.drop(columns=['building', 'P_TOT'])
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
    # Example usage
    path = 'path_to_your_quantile_forecasts.csv'
    # pf = ParametricForecasts(path)
    # pf.load_quantile_forecasts()
    # pf.sort_quantiles()
    
    # # Now you can call fit_distribution() to fit the distribution
    # # pf.fit_distribution()
    
    # # You can also access the sorted quantile forecasts
    # print(pf.quantile_forecasts)