from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
        elif {"import_quad", "export_quad", "import_lin", "export_lin"}.issubset(self.df.columns):
            if 'timestamp' in self.df.columns:
                self.prices = self.df[["timestamp", "import_quad", "export_quad", "import_lin", "export_lin"]].set_index("timestamp")
            self.prices = self.df[["import_quad", "export_quad", "import_lin", "export_lin"]].copy()
        elif {"import_A", "export_A", "import_k", "export_k", "import_c"}.issubset(self.df.columns):
            if 'timestamp' in self.df.columns:
                self.prices = self.df[["timestamp", "import_A", "export_A", "import_k", "export_k", "import_c"]].set_index("timestamp")
            self.prices = self.df[["import_A", "export_A", "import_k", "export_k", "import_c"]].copy()
        else:
            self.prices = None

        # get the total timespan of operation in hours
        self.total_timespan = (self.df.index[-1] - self.df.index[0]).total_seconds() / 3600
        self.t_start = self.df.index[0]
        self.t_end = self.df.index[-1]
        self.e_end = round(self.df['soe_new'].iloc[-1], 4) if 'soe_new' in self.df.columns else None
        self.pg_export_total = self.df['pg'].clip(upper=0).sum()
        self.pg_import_total = self.df['pg'].clip(lower=0).sum()

    def get_costs(self) -> dict:
        """ Calculates the total energy costs/revenue of the optimization results. """
        
        if self.prices is None:
            raise ValueError("Prices data is required to calculate energy costs.")
        
        # Get frequency of solution in hours
        if self.df.index.freq is None:
            # get the frequency by taking the timestamp stored in index of two rows
            self.df.index.freq = pd.infer_freq(self.df.index)
        freq = self.df.index.freq
        dt_hours = pd.Timedelta(freq).total_seconds() / 3600  # convert frequency to hours


        # Map prices to df-index (minutely)
        prices = self.prices.reindex(self.df.index, method='ffill')


        # Consider Import/export separately
        P_imp = self.df["pg"].clip(lower=0)  # Pg >= 0
        P_exp = (-self.df["pg"].clip(upper=0))  # Pg < 0 => positive Exportpower

        
        # Linear or quadratic objective
        has_linear = {"import_price", "export_price"}.issubset(prices.columns)
        has_quadr = {"import_quad", "export_quad", "import_lin", "export_lin"}.issubset(prices.columns)
        has_exp = {"import_A", "export_A", "import_k", "export_k", "import_c"}.issubset(prices.columns)

        if has_linear and not has_quadr:
            # Costs Import C = P_imp * c_imp * t;  Revenue Export R = P_exp * r_exp *t
            c_imp = prices["import_price"]  # €/kWh
            r_exp = prices["export_price"]  # €/kWh
            self.df['costs_buy'] = P_imp * c_imp * dt_hours  # costs for buying energy
            self.df['rev_sell'] = P_exp * r_exp * dt_hours  # revenue for selling energy

        elif has_quadr and not has_linear:
            # Costs Import: C = (c0 + c1 * P_imp) *P_imp * t
            c0 = prices["import_lin"]  # €/kWh
            c1 = prices["import_quad"]  # €/(kW * kWh)
            self.df["costs_buy"] = (c0 + c1 * P_imp) * P_imp * dt_hours  # costs for buying energy

            # Revenue Export: R = (r0 - b*P_exp) * P_exp * t
            r0 = prices["export_lin"]  # €/kWh
            b = prices["export_quad"]  # €/(kW * kWh)
            self.df["rev_sell"] = (r0 - b * P_exp) * P_exp * dt_hours  # revenue for selling energy
        
        elif has_exp and not (has_linear or has_quadr):
            # Costs Import: C = c * P_imp - A*(1-exp[-k*P_imp])
            c = prices["import_c"]
            A_imp = prices["import_A"]
            k_imp = prices["import_k"]
            self.df["costs_buy"] = (c * P_imp - A_imp * (1 - np.exp(-k_imp * P_imp))) * dt_hours  # costs for buying energy

            # Revenue Export: R = A*(1-exp[-k*P_exp])
            A_exp = prices["export_A"]
            k_exp = prices["export_k"]
            self.df["rev_sell"] = A_exp * (1 - np.exp(-k_exp * P_exp)) * dt_hours  # revenue for selling energy

        else:
            raise ValueError("Price column missing. Need either ['import_price', 'export_price'] OR ['import_quad', 'export_quad', 'import_lin', 'export_lin']")


        # Cashflow (Costs and Revenue are positive)
        self.df["cashflow"] = self.df.get("costs_buy", 0.0) - self.df.get("rev_sell", 0.0)

        import_cost = float(self.df.get("costs_buy", 0.0).sum())
        export_revenue = float(self.df.get("rev_sell", 0.0).sum())
        net_cost = import_cost - export_revenue

        costs_summary = {"import_cost": import_cost, "export_revenue": export_revenue, "net_cost": net_cost}

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





        



