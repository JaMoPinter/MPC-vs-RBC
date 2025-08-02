import pandas as pd

class PriceManager:
    """
    Loads and manages the prices for the optimization process. Also provides resampled versions if needed.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.prices = self._load_prices()

    def _load_prices(self):
        """
        Load the prices from the CSV file.
        """
        path = f'01_data/price_data/{self.filename}'
        df = pd.read_csv(path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    # def _resample_prices(self, mpc_freq: int) -> pd.DataFrame:
    #     """
    #     Resample the prices to the MPC frequency.
    #     """
    #     return self.prices.resample(f'{mpc_freq}min').mean()
    

    def get_prices(self, mpc_freq: int) -> pd.DataFrame:
        """
        Get the prices resampled to the MPC frequency.
        """
        return self.prices.resample(f'{mpc_freq}min').mean()  # TODO: Check if this works properly! Check that the resampling keeps the timestamps aligned.