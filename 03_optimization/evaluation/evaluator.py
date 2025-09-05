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
            prices: Optional[str | Path | pd.DataFrame] = None,
            battery_cfg: Optional[dict] = None
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

        #if self.prices is not None and {"c_bat_deg"}.issubset(self.df.columns):
        if {"c_bat_deg"}.issubset(self.df.columns):
            self.c_deg = self.df["c_bat_deg"].iloc[0]
        else:
            self.c_deg = float(0.0)
            print("!C_DEG = 0")

            # Get frequency of solution in hours
        if self.df.index.freq is None:
            # get the frequency by taking the timestamp stored in index of two rows
            self.df.index.freq = pd.infer_freq(self.df.index)
        freq = self.df.index.freq
        self.dt_hours = pd.Timedelta(freq).total_seconds() / 3600  # convert frequency to hours

        # get the total timespan of operation in hours
        self.total_timespan = (self.df.index[-1] - self.df.index[0]).total_seconds() / 3600
        self.t_start = self.df.index[0]
        self.t_end = self.df.index[-1]
        self.e_end = round(self.df['soe_new'].iloc[-1], 4) if 'soe_new' in self.df.columns else None
        pg = self.df["pg"].astype(float)
        pb = self.df["action"].astype(float)

        # Positive import / positive export magnitudes
        P_imp = pg.clip(lower=0.0)          # kW (>=0)
        P_exp = (-pg.clip(upper=0.0))       # kW (>=0)

        # Total energies over the horizon (kWh)
        self.e_import_total = round(float((P_imp * self.dt_hours).sum()), 4)
        self.e_export_total = round(float((P_exp * self.dt_hours).sum()), 4)

        self.pg_export_total = self.df['pg'].clip(upper=0).sum()
        self.pg_import_total = -self.df['pg'].clip(upper=0).sum()


        # Calculate the total energy throughput in kWh
        self.e_throughput_total = round(float((pb.abs() * self.dt_hours).sum()), 4)
        # get the total discharged energy to calculate battery losses
        pb_dis = pb.clip(lower=0.0)  # discharged kW (>=0)
        self.e_discharged_total = round(float((pb_dis * self.dt_hours).sum()), 4)
        self.battery_deg_costs = round(float(np.array((self.e_discharged_total * self.c_deg)).sum()), 4)



        # Battery specs for final SoE to € transformation
        self.cap_min = 0.0
        self.eta_dis = 0.95
        if battery_cfg is not None: # TODO: CAREFUL IF THIS IS NOT USED
            self.cap_min = float(battery_cfg.get('capacity_min', self.cap_min))
            self.eta_dis = float(battery_cfg.get('discharge_efficiency', self.eta_dis))

    def get_costs(self) -> dict:
        """ Calculates the total energy costs/revenue of the optimization results. """
        
        if self.prices is None:
            raise ValueError("Prices data is required to calculate energy costs.")
        



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
            self.df['costs_buy'] = P_imp * c_imp * self.dt_hours  # costs for buying energy
            self.df['rev_sell'] = P_exp * r_exp * self.dt_hours  # revenue for selling energy

        elif has_quadr and not has_linear:
            # Costs Import: C = (c0 + c1 * P_imp) *P_imp * t
            c0 = prices["import_lin"]  # €/kWh
            c1 = prices["import_quad"]  # €/(kW * kWh)
            self.df["costs_buy"] = (c0 + c1 * P_imp) * P_imp * self.dt_hours  # costs for buying energy

            # Revenue Export: R = (r0 - b*P_exp) * P_exp * t
            r0 = prices["export_lin"]  # €/kWh
            b = prices["export_quad"]  # €/(kW * kWh)
            self.df["rev_sell"] = (r0 - b * P_exp) * P_exp * self.dt_hours  # revenue for selling energy
        
        elif has_exp and not (has_linear or has_quadr):
            # Costs Import: C = c * P_imp - A*(1-exp[-k*P_imp])
            c = prices["import_c"]
            A_imp = prices["import_A"]
            k_imp = prices["import_k"]
            self.df["costs_buy"] = (c * P_imp - A_imp * (1 - np.exp(-k_imp * P_imp))) * self.dt_hours  # costs for buying energy

            # Revenue Export: R = A*(1-exp[-k*P_exp])
            A_exp = prices["export_A"]
            k_exp = prices["export_k"]
            self.df["rev_sell"] = A_exp * (1 - np.exp(-k_exp * P_exp)) * self.dt_hours  # revenue for selling energy

        else:
            raise ValueError("Price column missing. Need either ['import_price', 'export_price'] OR ['import_quad', 'export_quad', 'import_lin', 'export_lin']")


        # Cashflow (Costs and Revenue are positive)
        self.df["cashflow"] = self.df.get("costs_buy", 0.0) - self.df.get("rev_sell", 0.0)

        import_cost = float(self.df.get("costs_buy", 0.0).sum())
        export_revenue = float(self.df.get("rev_sell", 0.0).sum())
        net_cost = import_cost - export_revenue

        # Calculate terminal SoE value
        terminal_value = 0.0
        if self.e_end is not None:
            deliverable_kwh = max(self.e_end - self.cap_min, 0.0) * self.eta_dis
            # take the last timesteps price with 1kW
            last_price_row = prices.iloc[-1]
            price_last = self._terminal_price_at_1kW(last_price_row)
            terminal_value = float(deliverable_kwh * price_last)
        net_cost_adj = net_cost - terminal_value


        # Calculate grid-friendliness factors
        # Get the sum of all squared pg imports and pg_exports
        P_imp_sq = (P_imp ** 2).sum()
        P_exp_sq = (P_exp ** 2).sum()


        pg_rolling = self.compute_pg_metrics_rolling(window="15min")

        # Add degradation costs
        net_cost_final = net_cost_adj + self.battery_deg_costs

        costs_summary = {"import_cost": import_cost, "export_revenue": export_revenue, "net_cost": net_cost, "net_cost_adj": net_cost_adj, "import_squared": P_imp_sq, "export_squared": P_exp_sq, "e_discharged_total": self.e_discharged_total, "battery_deg_costs": self.battery_deg_costs, "net_cost_final": net_cost_final}

        costs_summary.update(pg_rolling)

        return costs_summary
    

    def _terminal_price_at_1kW(self, prices_row: pd.Series) -> float:
        """ Compute the theoretical export 'price per kWh' at 1 kW from the last timestep. """
        # Linear Price
        if {"export_price"}.issubset(prices_row.index):
            return float(prices_row["export_price"])
        # Quadratic Revenue for 1kW during 1h: (r0 - b * 1) * 1
        elif {"export_quad", "export_lin"}.issubset(prices_row.index):
            return float((prices_row["export_lin"] - prices_row["export_quad"]) * 1.0)
        # Exponential Revenue for 1kW during 1h: A*(1 - exp(-k*1))
        elif {"export_A", "export_k"}.issubset(prices_row.index):
            A = float(prices_row["export_A"])
            k = float(prices_row["export_k"])
            return float(A * (1.0 - np.exp(-k * 1.0)))
        else:
            raise ValueError("Price row is missing required columns.")



    def compute_pg_metrics_rolling(self, window="15min"):

        pg = self.df["pg"].astype(float)

        minp = self._expected_samples_in_window(window) 
        pg_avg = pg.rolling(window=window, min_periods=minp).mean()

        dt_h = self._infer_dt_hours()
        k = self._kpis_from_series(pg_avg, window_hours=dt_h)

        # add prefix to keys to avoid collisions
        #prefix: str = "roll15"
        prefix=None
        return {f"{k_}": v for k_, v in k.items()}



    def _kpis_from_series(self, s: pd.Series, window_hours: float) -> dict:
        """ Comput Key Performance Indicators on an averaged power series s. """

        s = s.dropna().astype(float)
        n = len(s)

        mss = float((s**2).mean())
        rms = float(np.sqrt(mss))

        pmax_import = float(s.clip(lower=0).max())
        pmax_export = float((-s.clip(upper=0)).max()) # TODO: Check this metric

        q95_abs = float(s.abs().quantile(0.95))

        threshold_kw = q95_abs

        exceed_mask = (s.abs() > threshold_kw)
        exceed_frac = float(exceed_mask.mean())
        exceed_hours = float(exceed_frac * n * window_hours)

        return {
        "count": n,
        "mss_kw2": mss,
        "rms_kw": rms,
        "q95_abs_kw": q95_abs,
        "pmax_import_kw": pmax_import,
        "pmax_export_kw": pmax_export,
        "exceed_frac": exceed_frac,
        "exceed_hours": exceed_hours,
        "threshold_kw": float(threshold_kw),
        }
    





    def _infer_dt_hours(self) -> float:
        if self.df.index.freq is None:
            self.df.index.freq = pd.infer_freq(self.df.index)
        return pd.Timedelta(self.df.index.freq).total_seconds() / 3600.0

    def _expected_samples_in_window(self, window: str) -> int:
        """  """
        dt_h = self._infer_dt_hours()
        win_h = pd.Timedelta(window).total_seconds() / 3600.0

        return max(1, int(round(win_h / max(dt_h, 1e-12))))
















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





        



