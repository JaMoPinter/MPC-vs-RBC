# Rule-Based enhanced deterministic MPC. Rule is to follow grid schedule.


from .mpc_det import MpcDetOptimizer
import pandas as pd
from pathlib import Path
from utils import map_building_to_pv_num_orientation


class MpcRuleConstGridOptimizer(MpcDetOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def update_soe(self, t_now, decision, gt):
        ''' Take pg as main decision that wants to be followed. Rule-based following of the grid schedule. '''

        pg_desired = decision.get('pg')
        pb_desired = gt - pg_desired

        # Ensure pb is within limits
        pb_desired = max(self.pb_min, min(self.pb_max, pb_desired))

        if pb_desired < 0.0:  # CHARGING
            alpha = self.eta_ch

            # Check if we can charge more or if we hit our battery limits
            available_cap = self.cap_max - self.soe_now

            pb_energy_min = - available_cap / (alpha * self.gt_inc)  # Most negative pb possible given battery limits
            pb = max(pb_desired, pb_energy_min)  # select a pb such that pg is followed as closely as possible!

        else:  # DISCHARGING
            alpha = 1/self.eta_dis

            # Check here if we can discharge more or if we hit our battery limits!
            available_cap = self.soe_now - self.cap_min

            pb_energy_max = available_cap / (alpha * self.gt_inc)  # Most positive pb possible given battery limits
            pb = min(pb_desired, pb_energy_max)

        pg_actual = gt - pb
        soe_new = self.soe_now - pb * alpha * self.gt_inc

        if round(soe_new, 5) > self.cap_max or round(soe_new, 5) < self.cap_min:
            raise ValueError(f"State of charge out of bounds: {soe_new} kWh. Should be between {self.cap_min} and {self.cap_max} kWh.")

        if soe_new > self.cap_max:
            soe_new = self.cap_max
        if soe_new < self.cap_min:
            soe_new = self.cap_min

        self.results_realization[t_now] = {
            'timestamp': t_now,
            'action': pb,       # Power setpoint for the battery at t_now
            'pb': pb,   # Power setpoint for the battery at t_now
            'pg': pg_actual,           # Grid power after applying the action
            'gt': gt,               # Ground truth at t_now
            'soe_now': self.soe_now,  # Current state of energy before applying the action
            'soe_new': soe_new,      # New state of energy after applying the action
            'pb_mpc': decision.get('pb'),  # Original MPC battery power setpoint
            'pg_mpc': decision.get('pg')   # Original MPC grid power setpoint
        }   

        self.soe_now = soe_new
        return soe_new



class IdealRuleConstGridOptimizer(MpcRuleConstGridOptimizer):

    def __init__(self, *args, **kwargs):
        self.super_fc_long = None  # Store the ground truth to decrease number of file access
        super().__init__(*args, **kwargs)



    def _prepare_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        ''' Instead of using the provided forecast => use ground truth as forecast. '''

        if self.super_fc_long is None:
            # Load the full gt
            num_pv_modules, orientation = map_building_to_pv_num_orientation(self.b)

            path = Path(f'01_data/prosumption_data/{self.mpc_freq}min/prosumption_{self.b}_num_pv_modules_{num_pv_modules}_pv_{orientation}_hp_1.0.csv')
            df = pd.read_csv(path, parse_dates=['index'], index_col='index', usecols=['index', 'P_TOT'])
            df.index.name = 'timestamp'
            self.super_fc_long = df
        
        self.super_fc = self.super_fc_long.copy()
        self.super_fc = self.super_fc.loc[forecast.index.get_level_values('timestamp')]
        self.super_fc['P_TOT'] = self.super_fc['P_TOT'] / 1000.0  # Convert from W to kW

        self.super_fc.rename(columns={'P_TOT': 'expected_value'}, inplace=True)
        return self.super_fc
