import pandas as pd

class PriceManager:
    """
    Loads and manages the prices for the optimization process. Also provides resampled versions if needed.
    """

    def __init__(self, filename: str, full_path: bool = False):
        self.filename = filename
        self.full_path = full_path
        self.prices = self._load_prices()

    def _load_prices(self):
        """
        Load the prices from the CSV file.
        """
        if self.full_path:
            path = self.filename
        else:
            path = f'01_data/price_data/{self.filename}'
        df = pd.read_csv(path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    

    def get_prices(self, mpc_freq: int) -> pd.DataFrame:
        """
        Get the prices resampled to the MPC frequency.
        """
        self.prices = self.prices.resample(f'{mpc_freq}min').mean()  # TODO: Check if this works properly! Check that the resampling keeps the timestamps aligned.
        self.prices = self.prices.ffill()

        return self.prices