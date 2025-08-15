from .interval_optimizer import IntervalOptimizer
import pyomo.environ as pyo


class MpcProbOptimizer(IntervalOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _define_interval_constraints(self):
        def constr_y_width(model, t):
            ''' y_high[t] - y_low[t] = 0 '''
            return model.y_high[t] - model.y_low[t] == 0
        self.model.constr_y_width = pyo.Constraint(self.model.time, rule=constr_y_width)
