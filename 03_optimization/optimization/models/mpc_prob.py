from ..base import BaseOptimizer
#from interval_optimizer import IntervalOptimizer
import pyomo.environ as pyo

# Maybe inherit from the IntervalOptimizer. Only needs to have one additional constraint that specifies y_low == y_high

class MpcProbOptimizer(BaseOptimizer):
    


    def _build_model(self):

        self._define_sets()

        self._define_parameters()

        self._define_decision_variables()

        self._define_constraints()

        # Prohibit interval scheduling
        self._prohibit_interval_scheduling()

        self._define_objective_function()


    def _prohibit_interval_scheduling(self):
        """
        Prohibit interval scheduling by setting y_low == y_high.
        """
        def constr_y_order(model, t):
            ''' y_low[t] <= y_high[t] '''
            return model.y_low[t] <= model.y_high[t]
        self.model.constr_y_order = pyo.Constraint(self.model.time, rule=constr_y_order)