
import pandas as pd
from scipy.stats import norm, gaussian_kde
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
import numpy as np
import os
import sys


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


    def load_quantile_forecasts(self, csv_path, timerange=None):
        """ Load quantile forecasts from the specified path and group them according to their creation timestamp. 
        
        
        Args:
            csv_path (str): Path to the CSV file containing quantile forecasts.
            timerange (list, optional): A list with two elements specifying the start and end time for filtering the forecasts.
        """

        self.csv_path = csv_path

        # 1) Load CSV, parse timestamps
        df = pd.read_csv(
            csv_path,
            parse_dates=['timestamp','time_fc_created'],
            index_col='timestamp'          # if you want 'timestamp' as the DataFrame index
        )

        # Change the order of the columns to have P_TOT before quantiles
        cols = df.columns.tolist()
    
        cols = ['P_TOT'] + [col for col in cols if col not in  ['P_TOT']]
        df = df[cols]

        # remove the quantile_ prefix from the quantile columns
        df.columns = df.columns.str.replace('quantile_', '', regex=False)

        # 2) Filter by timerange if provided. Keep all rows where 'time_fc_created' is within the specified range.
        if timerange is not None:
            start_time, end_time = pd.to_datetime(timerange[0]), pd.to_datetime(timerange[1])
            df = df[(df['time_fc_created'] >= start_time) & (df['time_fc_created'] <= end_time)]

        
        
        # 2) Group by the forecast‐creation time
        groups = {
            created_time: group.copy()
            for created_time, group in df.groupby('time_fc_created')
        }
        
        self.quantile_forecasts = {}

        for created_time, subdf in groups.items():
            subdf = subdf.drop(columns=['time_fc_created'])
            self.quantile_forecasts[created_time] = subdf


        self.quantile_forecasts = pd.concat(self.quantile_forecasts, axis=0, names=['time_fc_created', 'timestamp'])
        # drop the 'building' and 'P_TOT' columns from the quantile forecasts
        self.quantile_forecasts = self.quantile_forecasts.drop(columns=['P_TOT'])
        

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

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Prepare the filepath
        filepath = os.path.join(directory, filename)
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filepath):
            filepath = os.path.join(directory, f"{base}_{counter}{ext}")
            counter += 1

        # Save to CSV
        self.param_forecasts.to_csv(filepath, index=True)
        print(f"Parametric forecasts saved to {filepath}")



      

    def sort_quantiles(self):
        """ Sort the quantile columns in ascending order for each forecast (each row). """
        print("Sorting quantiles in ascending order…")

        # 1) Identify the quantile columns by whatever naming convention you have:
        #    Here you used strings like "0.01", "0.02", … so we test for a leading '0.'
        quant_cols = [c for c in self.quantile_forecasts.columns if c.startswith("0.")]

        # 2) Extract all the quantile values as a 2D numpy array (shape: n_rows × n_quantiles)
        arr = self.quantile_forecasts[quant_cols].values

        # 3) Sort each row (axis=1) in-place
        sorted_arr = np.sort(arr, axis=1)

        # 4) Assign the sorted values back into the same columns
        #    .loc[:, quant_cols] works regardless of the MultiIndex on the rows
        self.quantile_forecasts.loc[:, quant_cols] = sorted_arr
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

        #df_quantiles = self.quantile_forecasts.drop(columns=['building', 'P_TOT'])
        fc_smoothed = []
        for idx, row in df.iterrows():
            # row is a single timestamp’s array of quantile values
            smoothed_values = row.rolling(window=window_size, min_periods=1, center=True).mean()
            fc_smoothed.append(smoothed_values.values)
        return pd.DataFrame(fc_smoothed, index=df.index, columns=df.columns)

                

    
        

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