# map optimizer names to classes

from .models.rule_based import RuleBasedOptimizer
from .models.ideal_fc import IdealOptimizer
from .models.mpc_det_rule_based_constGrid import MpcRuleConstGridOptimizer
from .models.mpc_det_rule_based_constGrid import IdealRuleConstGridOptimizer

class OptimizerRegistry:
    _registry = {
        'rule-based': RuleBasedOptimizer,
        'ideal': IdealOptimizer,
        'mpc_det_rule_constGrid': MpcRuleConstGridOptimizer,
        'ideal_rule_constGrid': IdealRuleConstGridOptimizer

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