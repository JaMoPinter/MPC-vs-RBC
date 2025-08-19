from ..base import BaseOptimizer

import numpy as np
import pandas as pd
import pyomo.environ as pyo


class MpcDetOptimizer(BaseOptimizer):
    """
    Deterministic MPC based on deterministic forecasts.

    If forecast is probabilistic, use the expected value. 
    
    To prevent the model from discharging at the end of the horizon, we do XXX
    
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)




    def _prepare_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """
        If forecast is probabilistic, return expected value. Otherwise return unchanged forecast.
        """
        if 'timestamp' in forecast.columns:
            forecast = forecast.set_index('timestamp')
        
        # check if forecast only has one column
        if forecast.shape[1] == 1:
            return forecast

        # Forecast is probabilistic
        fc = forecast.copy()
        if self.param_assumption == 'sum2gaussian':
            for t, row in fc.iterrows():
                mu1 = row['mu1']
                mu2 = row['mu2']
                w1 = row['w1']
                w2 = row['w2']
                # Compute the expected value as a weighted sum
                expected_value = w1 * mu1 + w2 * mu2
                fc.at[t, 'expected_value'] = expected_value
        return fc[['expected_value']]





    def _build_model(self):

        self.model = pyo.ConcreteModel(name='mpc_det')

        self._define_sets()

        self._define_parameters()

        self._define_decision_variables()

        self._define_constraints()

        self._define_objective()
        

    def _define_sets(self):
        self.model.time = pyo.Set(initialize=self.time_index)
        self.model.time_e = pyo.Set(initialize=self.time_index.append(pd.Index([self.time_index[-1] + self.time_index.freq])))
        self.model.time_e0 = pyo.Set(initialize=self.time_index.to_list()[:1])

    def _define_parameters(self):
        self.model.e0 = pyo.Param(self.model.time_e0, initialize=self.soe_now)
        self.model.pl = pyo.Param(self.model.time, initialize=self.fc_exp['expected_value'].to_dict())

    def _define_decision_variables(self):
        self.model.pb = pyo.Var(self.model.time, domain=pyo.Reals)
        self.model.pb_ch = pyo.Var(self.model.time, domain=pyo.Reals)
        self.model.pb_dis = pyo.Var(self.model.time, domain=pyo.Reals)

        self.model.pg = pyo.Var(self.model.time, domain=pyo.Reals)
        self.model.pg_sell = pyo.Var(self.model.time, domain=pyo.Reals)
        self.model.pg_buy = pyo.Var(self.model.time, domain=pyo.Reals)

        self.model.e = pyo.Var(self.model.time_e, domain=pyo.Reals)



    def _define_constraints(self):
        # Battery Evolution
        def constr_e_evo(model, t):
            ''' e[t] = e[t-1] -pb_ch[t-1] * dt * eta_ch - pb_dis[t-1] * dt / eta_dis '''
            if t == model.time_e.first():
                return model.e[t] == model.e0[t]
            else:
                t_prev = model.time_e.prev(t)
                return model.e[t] == model.e[t_prev] - model.pb_ch[t_prev] * self.t_inc * self.eta_ch - model.pb_dis[t_prev] * self.t_inc / self.eta_dis
        self.model.constr_e_evo = pyo.Constraint(self.model.time_e, rule=constr_e_evo)

        # Battery Limits
        def constr_e_max_limit(model, t):
            ''' e[t] <= e_max '''
            return model.e[t] <= self.cap_max
        self.model.constr_e_max_limit = pyo.Constraint(self.model.time_e, rule=constr_e_max_limit)

        def constr_e_min_limit(model, t):
            ''' e[t] >= e_min '''
            return model.e[t] >= self.cap_min
        self.model.constr_e_min_limit = pyo.Constraint(self.model.time_e, rule=constr_e_min_limit)

        def constr_pb_max(model, t):
            ''' pb[t] <= pb_max '''
            return model.pb[t] <= self.pb_max
        self.model.constr_pb_max = pyo.Constraint(self.model.time, rule=constr_pb_max)

        def constr_pb_min(model, t):
            ''' pb[t] >= pb_min '''
            return model.pb[t] >= self.pb_min
        self.model.constr_pb_min = pyo.Constraint(self.model.time, rule=constr_pb_min)

        # Complementary Constraints
        def constr_pb_split(model, t):
            ''' pb[t] = pb_ch[t] + pb_dis[t] '''
            return model.pb[t] == model.pb_ch[t] + model.pb_dis[t]
        self.model.constr_pb_split = pyo.Constraint(self.model.time, rule=constr_pb_split)

        def constr_pb_relaxation(model, t):
            ''' -1e-8 <= pb_ch[t] * pb_dis[t] <= 0 '''
            return pyo.inequality(-1e-8, model.pb_ch[t] * model.pb_dis[t], 0)
        self.model.constr_pb_relaxation = pyo.Constraint(self.model.time, rule=constr_pb_relaxation)


        def constr_pg_split(model, t):
            ''' pg[t] = pg_sell[t] + pg_buy[t] '''
            return model.pg[t] == model.pg_sell[t] + model.pg_buy[t]
        self.model.constr_pg_split = pyo.Constraint(self.model.time, rule=constr_pg_split)

        def constr_pg_relaxation(model, t):
            ''' -1e-8 <= pg_sell[t] * pg_buy[t] <= 0 '''
            return pyo.inequality(-1e-8, model.pg_sell[t] * model.pg_buy[t], 0)
        self.model.constr_pg_relaxation = pyo.Constraint(self.model.time, rule=constr_pg_relaxation)


        # Charging / Discharging Constraints
        def constr_pb_ch(model, t):
            ''' pb_ch[t] <= 0 '''
            return model.pb_ch[t] <= 0
        self.model.constr_pb_ch = pyo.Constraint(self.model.time, rule=constr_pb_ch)

        def constr_pb_dis(model, t):
            ''' pb_dis[t] >= 0 '''
            return model.pb_dis[t] >= 0
        self.model.constr_pb_dis = pyo.Constraint(self.model.time, rule=constr_pb_dis)


        def constr_pg_sell(model, t):
            ''' pg_sell[t] <= 0 '''
            return model.pg_sell[t] <= 0
        self.model.constr_pg_sell = pyo.Constraint(self.model.time, rule=constr_pg_sell)

        def constr_pg_buy(model, t):
            ''' pg_buy[t] >= 0 '''
            return model.pg_buy[t] >= 0
        self.model.constr_pg_buy = pyo.Constraint(self.model.time, rule=constr_pg_buy)

        # Power Balance
        def constr_power_balance(model, t):
            ''' pg[t] + pb[t] = pl[t] '''
            return model.pg[t] + model.pb[t] == model.pl[t]
        self.model.constr_power_balance = pyo.Constraint(self.model.time, rule=constr_power_balance)

        # # E-end should be equal to e0
        # def constr_e_end(model):
        #     ''' e_end = e0 '''
        #     return model.e[model.time_e.last()] == model.e0[model.time_e0.first()]
        # self.model.constr_e_end = pyo.Constraint(rule=constr_e_end)

    def _define_objective(self):

        if self.objective == 'linear':
            def objective(model):
                sum_costs = sum(
                    self.c_sell1[t] * model.pg_sell[t]
                    + self.c_buy1[t] * model.pg_buy[t]
                    for t in model.time
            )
                return sum_costs
        elif self.objective == 'quadratic':
            def objective(model):
                ''' TBD '''
                sum_costs = sum(
                    self.c_buy1[t] * model.pg_buy[t]**2 + self.c_buy2[t] * model.pg_buy[t]
                    + self.c_sell1[t] * model.pg_sell[t]**2 + self.c_sell2[t] * model.pg_sell[t]
                    for t in model.time
                )
                return sum_costs
        else:
            raise ValueError(f"Unknown objective function: {self.objective}. Choose 'linear' or 'quadratic'.")
        self.model.objective = pyo.Objective(rule=objective, sense=pyo.minimize)


    # def solve(self):
    #     solver = pyo.SolverFactory('ipopt')
    #     solver.options['max_iter'] = 5000
    #     result = solver.solve(self.model, tee=True)
    #     return result




    def optimize(self, t_now: pd.Timestamp, forecast: pd.DataFrame) -> dict:

        # Step 1: Get the forecast in correct format
        self.fc_exp = self._prepare_forecast(forecast)
        self.fc_exp.index = pd.DatetimeIndex(self.fc_exp.index.get_level_values('timestamp'), freq=str(self.mpc_freq) + 'min')
        self.time_index = self.fc_exp.index

        # Step 2: Get the prices
        # self.c_buy = self.c_buy_long[self.time_index]
        # self.c_sell = self.c_sell_long[self.time_index]

        self.c_buy1, self.c_sell1, self.c_buy2, self.c_sell2 = self._get_prices(self.time_index) # cbuy2=csell2=None for linear prices

        


        # Step 3: Build the model
        self._build_model()

        # Step 4: Solve the model
        result = self.solve()

        # Step 5: Emergency fallback decision if solver fails
        if result is None:
            fb_decision = self._fallback_decision()
            return fb_decision

        # Step 6: Get and return decision
        decision = {'pb': [pyo.value(self.model.pb[t]) for t in self.model.time][0], 'solver_ok': True, "solver_status": "ok"}
        return decision
    


    def update_soe(self, t_now, decision, gt):
        
        pb = decision.get('pb')

        if pb <= 0:
            eta = self.eta_ch
        else:
            eta = 1/self.eta_dis

        pg = gt - pb

        soe_new = self.soe_now - pb *self.gt_inc * eta

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
            'pg': pg,           # Grid power after applying the action
            'gt': gt,               # Ground truth at t_now
            'soe_now': self.soe_now,  # Current state of energy before applying the action
            'soe_new': soe_new      # New state of energy after applying the action
        }


        self.soe_now = soe_new
        return soe_new
        
        
