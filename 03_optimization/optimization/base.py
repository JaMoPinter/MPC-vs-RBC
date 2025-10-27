from abc import ABC, abstractmethod
import pandas as pd
import pyomo.environ as pyo
from collections import Counter

class BaseOptimizer(ABC):
    """
    Abstract base class for rolling-horizon optimizers.
    """

    def __init__(self, battery_cfg: dict, mpc_freq: int, gt_freq: int, prices, objective: str, building, param_assumption: str = None):
        """
        Initialize the optimizer.
        """
        self.battery_cfg = battery_cfg
        self.objective = objective

        if self.objective == 'linear':
            self.c_buy_long = prices['import_price']  # Import price for a longer than necessary period
            self.c_sell_long = prices['export_price']
        elif self.objective == 'quadratic':
            self.c_quad_buy_long = prices['import_quad']
            self.c_lin_buy_long = prices['import_lin']
            self.c_quad_sell_long = prices['export_quad']
            self.c_lin_sell_long = prices['export_lin']
        elif self.objective == 'exponential':
            self.c_imp_c_long = prices['import_c']
            self.c_imp_A_long = prices['import_A']
            self.c_imp_k_long = prices['import_k']
            self.c_exp_A_long = prices['export_A']
            self.c_exp_k_long = prices['export_k']

        self.c_deg = self.battery_cfg["c_bat_deg"]  # €/kWh degradation costs

        self.b = building
        
        self.mpc_freq = mpc_freq  # MPC frequency in minutes
        self.t_inc = self.mpc_freq / 60  # Convert minutes to hours
        self.gt_inc = gt_freq / 60 # frequency of ground truth data in hours

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

        # Keeping track of solver successes/failures
        self.solver_attampts: int = 0
        self.solver_failures: int = 0
        self.solver_fail_messages: Counter[str] = Counter()  # Count of different solver failure messages
        self.last_solver_ok: bool | None = None  # Was last solver run successful?
        self.last_solver_status: str | None = None  # Why did it (not) work? Short messages
        self.last_solver_message: str | None = None  # What exactly happened? => More detailed



    @abstractmethod
    def optimize(self, t_now: pd.Timestamp, fc: pd.DataFrame) -> dict:
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
    def update_soe(self, t: pd.Timestamp, decision, gt) -> dict: # Some methods will need gt as well, e.g., Rule-Based
        """
        Update the state of energy (soe) based on the action taken. # TODO: Correct description to return a dict
        
        Args:
            soe (float): Current state of charge of the battery in kWh.
            action: Action taken, e.g., battery power setpoint in kW.
        
        Returns:
            float: Updated state of charge after applying the action.
        """
        pass


    def solve(self):
        """ Solve the optimization using pyomo and IPOPT. """
        solver = pyo.SolverFactory('ipopt')
        solver.options['max_iter'] = 8000

        try: 
            result = solver.solve(self.model, tee=True)
        except Exception as e:
            self._log_solver_result(False, status="exception", message=repr(e))
            return None
        
        ok = pyo.check_optimal_termination(result)
        status = str(getattr(result.solver, "status", "unknown"))
        term = str(getattr(result.solver, "termination_condition", "unknown"))
        msg = f"{status} / {term}"

        self._log_solver_result(bool(ok), status=status, message=msg)
        return result if ok else None
    

    def _get_prices(self, time_index):
        
        if self.objective == 'linear':
            c_buy = self.c_buy_long[time_index]  
            c_sell = self.c_sell_long[time_index]  
            return c_buy, c_sell, None, None, None

        elif self.objective == 'quadratic':
            c_quad_buy = self.c_quad_buy_long[time_index]
            c_lin_buy = self.c_lin_buy_long[time_index]
            c_quad_sell = self.c_quad_sell_long[time_index]
            c_lin_sell = self.c_lin_sell_long[time_index]
            return c_quad_buy, c_quad_sell, c_lin_buy, c_lin_sell, None

        elif self.objective == 'exponential':
            c_imp_c = self.c_imp_c_long[time_index]
            c_imp_A = self.c_imp_A_long[time_index]
            c_imp_k = self.c_imp_k_long[time_index]
            c_exp_A = self.c_exp_A_long[time_index]
            c_exp_k = self.c_exp_k_long[time_index]
            return c_imp_A, c_exp_A, c_imp_k, c_exp_k, c_imp_c


    def _fallback_decision(self) -> float:
        """ 
        Emergency action when the solver fails. Log occurences and see if they can distort the results.
        
        Set battery power to zero.
        """
        pb = 0.0
        pg = 0.0 

        fb_decision = {
            "pb": pb,
            "pg": pg,
            "solver_ok": False,
            "solver_status": "error1",
            "solver_error": "fallback_decision"
        }
        return fb_decision
    

    def _log_solver_result(self, ok:bool, status: str | None = None, message: str | None = None) -> None:
        """ Record the outcome of a solver attempt (per MPC step). """
        self.solver_attampts += 1
        if not ok:
            self.solver_failures += 1
            if message:
                # Keep message short
                self.solver_fail_messages[message[:160]] += 1

        self.last_solver_ok = ok
        self.last_solver_status = status
        self.last_solver_message = message

