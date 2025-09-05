# map optimizer names to classes

from .models.rule_based import RuleBasedOptimizer
from .models.mpc_det import MpcDetOptimizer
from .models.mpc_prob import MpcProbOptimizer
from .models.interval_optimizer import IntervalOptimizer
from .models.ideal_fc import IdealOptimizer
from .models.mpc_det_rule_based import MpcRuleOptimizer
#from .models.mpc_det_rule_based import MpcRuleConsumerOptimizer
from .models.mpc_det_rule_based import IdealRuleOptimizer

class OptimizerRegistry:
    _registry = {
        'rule-based': RuleBasedOptimizer,
        'mpc_det': MpcDetOptimizer,
        'mpc_det-fc': MpcDetOptimizer,  # For using a deterministic forecast
        'mpc_prob': MpcProbOptimizer,
        'interval': IntervalOptimizer,
        'ideal': IdealOptimizer,
        'mpc_det_rule': MpcRuleOptimizer,
#        'mpc_det_rule_consumer': MpcRuleConsumerOptimizer,
        'ideal_rule': IdealRuleOptimizer
    }

    @classmethod
    def get_optimizer(cls, name: str):
        """
        Retrieve an optimizer class by its name.

        Args:
            name (str): The name of the optimizer.

        Returns:
            Optimizer class corresponding to the given name.

        Raises:
            KeyError: If the optimizer name is not registered.
        """
        try:
            return cls._registry[name]
        except KeyError:
            raise ValueError(f"Unknown optimizer '{name}'. Available: {list(cls._registry)}")