"""Public algorithm names and their separate implementations."""

from .adaptive_hill_climb import AdaptiveHillClimbOptimizer
from .adaptive_random_search import AdaptiveRandomSearchOptimizer
from .hill_climb import HillClimbOptimizer
from .nelder_mead import NelderMeadOptimizer
from .nelder_mead_manual import ManualNelderMeadOptimizer
from .pybobyqa import PyBobyqaOptimizer
from .random_search import RandomSearchOptimizer
from .stochastic_hill_climb import StochasticHillClimbOptimizer


OPTIMIZERS = {
    'hill_climb': StochasticHillClimbOptimizer,
    'hill_climb_manual': HillClimbOptimizer,
    'adaptive_hill_climb': AdaptiveHillClimbOptimizer,
    'nelder_mead': NelderMeadOptimizer,
    'nelder_mead_manual': ManualNelderMeadOptimizer,
    'pybobyqa': PyBobyqaOptimizer,
    'random_search': RandomSearchOptimizer,
    'simple_adaptive_random_search': AdaptiveRandomSearchOptimizer,
}
