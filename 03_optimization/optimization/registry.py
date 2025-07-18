# map optimizer names to classes

from .models.rule_based import RuleBasedOptimizer
from .models.mpc_det import MpcDetOptimizer
from .models.mpc_prob import MpcProbOptimizer
from .models.interval_optimizer import IntervalOptimizer

class OptimizerRegistry:
    _registry = {
        'rule-based': RuleBasedOptimizer,
        'mpc_det': MpcDetOptimizer,
        'mpc_prob': MpcProbOptimizer,
        'interval': IntervalOptimizer
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