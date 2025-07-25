# File to fit parametric distribution to quantile forecasts

# Create a class to handle the creation of parametric forecasts
# The class needs the path to the file where the forecasts are stored and the name of the distribution to fit as inputs
# The class should contain a method to load the quantile_forecasts, a method to fit the distribution (including sorting the quantiles) and a method to store the results in a different file
# For now only normal distribution and the sum of two normal distributions is implemented as a parametric distribution

import pandas as pd
from scipy.stats import norm, gaussian_kde
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
import numpy as np
import os


class ParametricForecasts:
    def __init__(self):
        """
        Initialize the ParametricForecasts class.

        Args:
            path (str): Path to the file containing quantile forecasts.
            distribution (str): Type of distribution to fit ('normal' or 'sum2gaussian').
        """
        self.quantile_forecasts = None
        self.param_forecasts = None
        self.csv_path = None
        self.implemented_distributions = ['sum2gaussian']


    def load_quantile_forecasts(self, csv_path, timerange=None):
        """Load quantile forecasts from the specified path and group them according to their creation timestamp."""

        self.csv_path = csv_path

        # 1) Load CSV, parse timestamps
        df = pd.read_csv(
            csv_path,
            parse_dates=['timestamp','time_fc_created'],
            index_col='timestamp'          # if you want 'timestamp' as the DataFrame index
        )

        # Change the order of the columns to have P_TOT before quantiles
        cols = df.columns.tolist()
    
        cols = ['building', 'P_TOT'] + [col for col in cols if col not in  ['building', 'P_TOT']]
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


        # Drop the 'building' and 'P_TOT' columns from the quantile forecasts


        self.quantile_forecasts = pd.concat(self.quantile_forecasts, axis=0, names=['time_fc_created', 'timestamp'])
        # drop the 'building' and 'P_TOT' columns from the quantile forecasts
        self.quantile_forecasts = self.quantile_forecasts.drop(columns=['building', 'P_TOT'])
        

    def load_parametric_forecasts(self, csv_path, name='sum2gaussian'):
        """Load parametric forecasts from the specified. """

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
        """Fit the specified distribution to the quantile forecasts."""
        if self.quantile_forecasts is None:
            raise ValueError("Quantile forecasts not loaded. Call load_quantile_forecasts() first.")
        
        if name not in self.implemented_distributions:
            raise ValueError(f"Distribution '{name}' is not implemented. Available distributions: {self.implemented_distributions}")
        
        if name == 'sum2gaussian':
            self.fit_sum2gaussian()

    
    def fit_sum2gaussian(self):

        self.param_forecasts = {}

        for created_time, subdf in self.quantile_forecasts.items():
            
            # get a subdf excluding the 'building' and 'P_TOT' columns
            #df_quantiles = subdf.drop(columns=['building', 'P_TOT'])
            df_quantiles = subdf.copy()
            quantile_probabilites = df_quantiles.columns.astype(float)


            self.param_forecasts[created_time] = pd.DataFrame(index=df_quantiles.index, columns=['w1', 'mu1', 'std1', 'w2', 'mu2', 'std2'])   

            for t, quants in df_quantiles.iterrows():
                # Step 1: Create an interpolator to map continuous probabilities to continuous values
                inv_cdf = interp1d(quantile_probabilites, quants, kind='linear', fill_value='extrapolate')

                # Step 2: Generate synthethic samples from the inverse CDF via interpolation
                np.random.seed(42)
                synthetic_probs = np.random.uniform(0.0, 1.0, 10000)
                synthetic_values = inv_cdf(synthetic_probs)

                # Step 3: Fit Gaussian Mixture Model (GMM) to the synthetic samples
                gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='full')
                gmm.fit(synthetic_values.reshape(-1, 1))

                # Step 4: Store the GMM parameters in a DataFrame
                self.param_forecasts[created_time].loc[t, 'w1'] = gmm.weights_[0]
                self.param_forecasts[created_time].loc[t, 'mu1'] = gmm.means_[0, 0]
                self.param_forecasts[created_time].loc[t, 'std1'] = np.sqrt(gmm.covariances_[0, 0, 0]) # Transform covariance to standard deviation
                self.param_forecasts[created_time].loc[t, 'w2'] = gmm.weights_[1]
                self.param_forecasts[created_time].loc[t, 'mu2'] = gmm.means_[1, 0]
                self.param_forecasts[created_time].loc[t, 'std2'] = np.sqrt(gmm.covariances_[1, 0, 0]) # Transform covariance to standard deviation

        # Convert the dictionary of DataFrames to a single DataFrame
        self.param_forecasts = pd.concat(self.param_forecasts, axis=0, names=['time_fc_created', 'timestamp'])


    def store_parametric_forecasts(self):
        """Store the parametric forecasts to a CSV file."""
        if self.param_forecasts is None:
            raise ValueError("Parametric forecasts not generated. Call fit_distribution() first.")
        
        # Convert the dictionary of DataFrames to a single DataFrame
        if not self.param_forecasts:
            raise ValueError("No parametric forecasts to save.")
        #combined_df = pd.concat(self.param_forecasts, axis=0)
        #combined_df.index = combined_df.index.set_names(['time_fc_created', 'timestamp'])

        # add curent timestamp to the filename
        current_time = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')
        # Create the filename and directory based on the original CSV path
        filename = os.path.basename(self.csv_path).replace('file_fc', 'file_fc_parametric')
        filename = filename.replace('.csv', f'_CreationTime{current_time}.csv')
        directory = os.path.dirname(self.csv_path).replace('storage_quantile_fc', 'storage_param_fc')
        
        # Save to CSV
        self.param_forecasts.to_csv(directory + '/' + filename, index=True)
        print(f"Parametric forecasts saved to {directory}")


      

    def sort_quantiles(self):
        """Sort the quantile columns in ascending order for each forecast (each row),
        even though the DataFrame has a MultiIndex on its rows."""
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
        """Smooth the quantile forecasts using a rolling mean."""
        print("Smoothing each quantile over time with a rolling mean…")

        #df_quantiles = self.quantile_forecasts.drop(columns=['building', 'P_TOT'])
        
        # Apply rolling mean to each quantile column
        smoothed_df = df.rolling(window=window_size, min_periods=1).mean()
        
        # Update the original DataFrame with smoothed values
        self.quantile_forecasts.update(smoothed_df)
        
        return self.quantile_forecasts
    

    def smooth_quantiles_at_each_time(self, df, window_size=3):
        print("Smoothing all quantiles at each time with a rolling mean…")

        #df_quantiles = self.quantile_forecasts.drop(columns=['building', 'P_TOT'])
        fc_smoothed = []
        for idx, row in df.iterrows():
            # row is a single timestamp’s array of quantile values
            smoothed_values = row.rolling(window=window_size, min_periods=1, center=True).mean()
            fc_smoothed.append(smoothed_values.values)
        self.quantile_forecasts = df.copy()
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