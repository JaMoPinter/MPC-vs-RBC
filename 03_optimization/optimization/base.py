from abc import ABC, abstractmethod
import pandas as pd

class BaseOptimizer(ABC):
    """
    Abstract base class for rolling-horizon optimizers.
    """

    def __init__(self, battery_cfg: dict, mpc_freq: int, prices, param_assumption: str = None):
        """
        Initialize the optimizer.
        """
        self.battery_cfg = battery_cfg
        self.c_buy_long = prices['import_price']  # Import price for a longer than necessary period
        self.c_sell_long = prices['export_price']
        
        # TODO: Implement here all the parameters that are needed for all the optimizers! 
        self.mpc_freq = mpc_freq  # MPC frequency in minutes
        self.t_inc = self.mpc_freq / 60  # Convert minutes to hours

        self.param_assumption = param_assumption  # Assumption for parametric forecasts, e.g., 'normal', 'sum2gaussian', etc.
        
        self.cap_max = self.battery_cfg['capacity_max']
        self.cap_min = self.battery_cfg['capacity_min']
        self.pb_max = self.battery_cfg['power_max']
        self.pb_min = self.battery_cfg['power_min']

        self.eta_ch = self.battery_cfg['charge_efficiency']
        self.eta_dis = self.battery_cfg['discharge_efficiency']


        self.soe_now = self.battery_cfg['soe_initial']  # Current state of energy (SoE) in kWh
        self.soe_initial = self.battery_cfg['soe_initial']  # Current state of energy (SoE) in kWh



        self.results_schedule = {}  # Results from optimization
        self.results_realization = {}  # Results according to scheduled actions and ground truth


        # TODO: Load the prices
        # TODO: Resample the prices to the right frequency.


        # self.pb_high_limit = config.get('pb_high_limit')
        # self.pb_low_limit = config.get('pb_low_limit')
        # self.e_low_limit = config.get('e_low_limit')
        # self.e_high_limit = config.get('e_high_limit')
        
        # self.freq = config.get('freq')



    # def initialize(self, initial_soe: float):
    #     """
    #     Set the initial state of energy (SoE) before simulation starts.

    #     Args:
    #         initial_soe (float): Initial state of energy in kWh.
    #     """
    #     self.soe = initial_soe


    @abstractmethod
    def optimize(self, t_now: pd.Timestamp, fc: pd.DataFrame) -> dict: # Maybe for some optimizers, the gt is needed? Maybe only for update_soe?
        """
        Perform the optimization for the current timestamp, soe and forecast. 
        
        Args:
            t_now (pd.Timestamp): Current timestamp to start optimization.
            fc (pd.DataFrame): Forecast data at the current timestamp with length of mpc_horizon.
            soe_now (float): Current state of charge of the battery in kWh.
        
        Returns a dict containing at least:
            - 'action': float, the battery power setpoint at t_now
        """
        pass



    @abstractmethod
    def update_soe(self, t: pd.Timestamp, decision, gt) -> dict: # Some methods will need gt as well, e.g., Rule-Based? Should it be included here or in the specific optimizer?
        """
        Update the state of energy (soe) based on the action taken. # TODO: Correct description to return a dict
        
        Args:
            soe (float): Current state of charge of the battery in kWh.
            action: Action taken, e.g., battery power setpoint in kW.
        
        Returns:
            float: Updated state of charge after applying the action.
        """
        # TODO: What forms can action take? Sometimes it is a float, sometimes action consists of at least two values (e.g., interval optimizer)
        pass




    
    # @abstractmethod
    # def run_optimization(self, t_now: pd.Timestamp) -> None:
    #     """
    #     Run the optimization to calculate the optimal schedule.
    #     Args:
    #         t_now (pd.Timestamp): Current timestamp to start optimization.
    #     """
    #     pass

    # @abstractmethod
    # def get_next_action(self) -> float:
    #     """
    #     Return the battery power setpoint according to the latest schedule.
        
    #     Returns:
    #         float: Battery power setpoint in kW.
    #     """
    #     pass


