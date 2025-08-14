from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go



class Evaluator:
    """ 
    Evaluates a single optimization run (one building, one optimization model, one MPC frequency, ...).
    
    """

    def __init__(
            self, 
            df_run: str | Path | pd.DataFrame, 
            prices: Optional[str | Path | pd.DataFrame] = None
    ):
        #print("Current working directory:", Path.cwd())
        self.df = (pd.read_csv(df_run, parse_dates=['timestamp']) if isinstance(df_run, (str, Path)) else df_run.copy())
        if 'timestamp' in self.df.columns:
            self.df.set_index('timestamp', inplace=True)

        if prices is not None:
            self.prices = (pd.read_csv(prices, parse_dates=['timestamp']) if isinstance(prices, (str, Path)) else prices.copy())
            self.prices.reset_index()
            if 'timestamp' in self.prices.columns:
                self.prices.set_index('timestamp', inplace=True)
            self.prices.index = pd.to_datetime(self.prices.index)
        elif {"import_price", "export_price"}.issubset(self.df.columns):
            # attach implicit prices (already applied in SimulationEngine)
            if 'timestamp' in self.df.columns:
                self.prices = self.df[["timestamp", "import_price", "export_price"]].set_index("timestamp")
            self.prices = self.df[["import_price", "export_price"]].copy()
        else:
            self.prices = None

        # get the total timespan of operation in hours
        self.total_timespan = (self.df.index[-1] - self.df.index[0]).total_seconds() / 3600
        self.t_start = self.df.index[0]
        self.t_end = self.df.index[-1]

    def get_costs(self) -> dict:
        """ Calculates the total energy costs/revenue of the optimization results. """
        
        if self.prices is None:
            raise ValueError("Prices data is required to calculate energy costs.")
        
        if self.df.index.freq is None:
            # get the frequency by taking the timestamp stored in index of two rows
            self.df.index.freq = pd.infer_freq(self.df.index)

        # The prices and df do not have the same resolution. We need to map each row of the df to prices of the latest timestamp.
        # whenever df['pg'] is positive, we buy energy, otherwise we sell it. ffill is forward filling to account for frequency mismatch of prices/df
        self.df['costs_buy'] = self.df['pg'].clip(lower=0) * self.prices['import_price'].reindex(self.df.index, method='ffill') * pd.Timedelta(self.df.index.freq).total_seconds() / 3600
        self.df['costs_sell'] = -self.df['pg'].clip(upper=0) * self.prices['export_price'].reindex(self.df.index, method='ffill') * pd.Timedelta(self.df.index.freq).total_seconds() / 3600

        # Calculate cashflow
        self.df['cashflow'] = self.df['costs_buy'] - self.df['costs_sell']

        # Calculate total costs/revenue
        import_cost = float(self.df['costs_buy'].sum())
        export_revenue = float(self.df['costs_sell'].sum())
        total_cost = import_cost - export_revenue

        costs_summary = {
            'import_cost': import_cost,
            'export_revenue': export_revenue,
            'net_cost': total_cost
        }

        return costs_summary
    
    def get_df(self) -> pd.DataFrame:
        """ Returns the DataFrame of the optimization results. This DataFrame can already contain the costs/revenues. """
        return self.df.copy()
    

    def plot_results(self) -> None:

        
        pass

    def plot_battery_soe(self, min_soe=0.0, max_soe=None) -> None:

        # make a plotly figure with the battery state of energy (soe) over time
        # for the battery, we need to append a row to the DataFrame with the soe of the end
        # check if 'action' of the last row is not a float, then we need to append a row with the last soe value
        if not isinstance(self.df['action'].iloc[-1], float):
            freq = self.df.index.freq
            next_timestamp = self.df.index[-1] + freq
            new_row = pd.DataFrame([{
                'soe_now': self.df['soe_new'].iloc[-1],
            }], index=[next_timestamp])
            
            self.df = pd.concat([self.df, new_row])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['soe_now'],
            mode='lines',
            name='SoE [kWh]'
        ))

        # add horizontal lines for the maximum and minimum SoE? Do we need a config file for this? :(
        if min_soe is not None and max_soe is not None:
            fig.add_hline(y=min_soe, line_dash="dash", line_color="red", annotation_text="Min Capacity")
            fig.add_hline(y=max_soe, line_dash="dash", line_color="red", annotation_text="Max Capacity")
            fig.update_yaxes(range=[min_soe - 0.3, max_soe + 0.3])

        fig.update_layout(
            title='Battery State of Energy (SoE) Over Time',
            xaxis_title='Time',
            yaxis_title='State of Energy (SoE)',
            legend_title='Legend',
            showlegend=True,
        )
        fig.show()





        



