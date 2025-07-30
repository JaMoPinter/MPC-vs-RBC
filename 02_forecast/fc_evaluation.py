from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import properscoring as ps
import numpy as np
import pandas as pd
import os
import sys


@dataclass
class _FCModel:
    name: str
    path: Path
    cache_file: Optional[Path] = None

class CrpsEvaluator:
    """ Evaluates CRPS metric for a single ground truth file. """

    def __init__(self, gt_path: Path, cache_dir: Path):
        self.gt = self._load_gt(gt_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._models: list[_FCModel] = []


    def add_model(self, *, name: str, path: Path):
        """ Adds a model to the pool of models to evaluate. """
        self._models.append(_FCModel(name=name, path=path))


    def evaluate(self, force=False):
        """ Loads/Computes all CRPS scores for all added models. If something is calculated, it is saved to a CSV 
            file in the cache directory. """
        for m in self._models:
            m.cache_file = self.cache_dir / f'evaluation_{m.name}.csv' # TODO: Add digest or timestamp to name?

            if m.cache_file.exists() and not force:
                print(f'Loading cached results for {m.name} from {m.cache_file}')
                continue

            crps_series = self._crps_evaluate(m.path)
            self._write_crps_to_csv(crps_series, m.cache_file)



    def _crps_evaluate(self, path: Path) -> pd.DataFrame:
        """ Calculates CRPS scores for quantile forecasts.

        Calculates for each 99 forecasted quantiles an inverse CDF function that maps probabilities to quantile values.
        Then, samples probabilities uniformly and calculates the crps score based on the samples, the ground truth and 
        the properscoring ps.crps_ensemble function. 

        Args:
            path (Path): Path to the CSV file containing quantile forecasts.
        """

        df = pd.read_csv(path)
        df.drop(columns=['building', 'P_TOT'], errors='ignore', inplace=True)  # Drop if exists
        df.set_index(['time_fc_created', 'timestamp'], inplace=True)

        # rename the columns
        df.columns = df.columns.str.replace(r"quantile_(\d+(?:\.\d+)?)", r"\1", regex=True)
        df.columns = df.columns.astype(float)

        y = np.sort(df.values, axis=1)
        df = pd.DataFrame(y, index=df.index, columns=df.columns)

        # check if values are sorted
        if not all(df.apply(lambda x: np.all(np.diff(x) >= 0), axis=1)):
            raise ValueError("Quantile values in some rows are not sorted.")


        def inverse_cdf_func(p, row):
            probs = row.index.values.astype(float)
            quantile_values = row.values
            # Interpolate to find x for given p
            return np.interp(p, probs, quantile_values)

        # Create a DataFrame to map probabilities to quantile values
        inverse_cdf_df = pd.DataFrame(index=df.index, columns=['inverse_cdf'])
        for i in range(len(df)):
            row = df.iloc[i]
            inverse_cdf_df.at[df.index[i], 'inverse_cdf'] = lambda p, row=row: inverse_cdf_func(p, row)

        # use crps_ensemble to calculate the CRPS for the quantile forecast
        crps_values = []
        sample_size = 1000  # Number of samples to draw
        uniform_samples = np.random.uniform(0, 1, sample_size)

        # get total number of rows for progress tracking

        total_rows = len(inverse_cdf_df)
        processed = 0
        last_percent = -1  # to avoid printing too often

        for index, row in inverse_cdf_df.iterrows():

            #print("Index: ", index)

            gt_value = float(self.gt.loc[index[1]])

            inverse_samples = row['inverse_cdf'](uniform_samples)
            crps_val = ps.crps_ensemble(gt_value, inverse_samples)  # TODO: Do I trust this method?
            crps_values.append(float(crps_val))

            # Progress update (every 1%)
            processed += 1
            percent = int(100 * processed / total_rows)
            if percent != last_percent and percent % 1 == 0:
                print(f"\rProgress: {percent}% ({processed}/{total_rows})", end="")
                sys.stdout.flush()
                last_percent = percent
        crps_values = pd.Series(crps_values, index=df.index, name='crps')
        return crps_values


    def _write_crps_to_csv(self, ser, path):
        ''' Writes the CRPS results to a CSV file. If the file already exists, it appends a counter to the filename. '''
        filepath = path
        base, ext = os.path.splitext(filepath)

        counter = 1
        while os.path.exists(filepath):
            filepath = base + f"_{counter}{ext}"
            counter += 1
        
        print(f'Saving CRPS results to {filepath}')

        out = ser.to_frame().reset_index()
        out.to_csv(filepath, index=False)

        

    def print_leaderboard(self, top_k: Optional[int] = None):
        ''' Print mean CRPS for each added model. '''
        rows = []
        for m in self._models:
            if not m.cache_file or not m.cache_file.exists():
                raise RuntimeError(f"Model {m.name} has not been evaluated.")
            crps = pd.read_csv(m.cache_file)["crps"]
            rows.append({"model": m.name, "mean_crps": crps.mean()})

        board = (
            pd.DataFrame(rows)
            .sort_values("mean_crps", ignore_index=True)
        )

        # Add the number of days used for the mean as third column
        board["num time_fc_created"] = board["model"].apply(
            lambda x: pd.read_csv(self.cache_dir / f'evaluation_{x}.csv')['time_fc_created'].nunique()
        )

        board["num total crps_vals"] = board["model"].apply(
            lambda x: pd.read_csv(self.cache_dir / f'evaluation_{x}.csv')['crps'].count()
        )


        if top_k:
            board = board.head(top_k)

        print(
            "\nCRPS leaderboard \n"
            + board.to_string(index=False, float_format="%.5f")
        )



    @staticmethod
    def _load_gt(path: Path) -> pd.Series:
        """
        Load a CSV and keep index and P_TOT column as Series.
        """        

        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        # only keep P_TOT column
        df = df[["P_TOT"]]
        ser = df.iloc[:, 0].sort_index()
        ser.name = "gt"
        return ser
