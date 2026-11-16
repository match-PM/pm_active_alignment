"""Adaptive coordinate hill climbing without duplicating its search loop."""

from .hill_climb import HillClimbOptimizer


class AdaptiveHillClimbOptimizer(HillClimbOptimizer):
    """Uses the adaptive_hill_climb parameter namespace and its own defaults."""

    # The search loop is identical to coordinate hill climbing. Its namespace
    # enables adaptation and provides different increase/decrease factors.
