"""The original main-branch stochastic hill-climbing strategy."""

import numpy as np

from .base import Optimizer


class StochasticHillClimbOptimizer(Optimizer):
    """Perturb all coordinates and retain only improvements."""

    def optimize(self, session):
        """Search around the best point using bounded random perturbations."""
        settings = session.parameters
        step = settings['step_size']
        failures = 0
        while session.evaluations < settings['max_evaluations']:
            previous = session.best.cost
            # Sample around the best point, not the last rejected candidate.
            candidate = session.best.position + session.rng.uniform(
                -step, step, len(session.initial_position))
            session.evaluate(np.clip(candidate, 0.0, 1.0))
            session.iterations += 1
            if session.best.cost < previous:
                failures = 0
                if settings['adaptive_step']:
                    step = min(step * settings['step_increase_factor'], settings['max_step'])
            else:
                failures += 1
                # Consecutive failures reduce the step only when adaptation is enabled.
                if settings['adaptive_step'] and failures >= settings['adaptation_window']:
                    if step <= settings['min_step']:
                        session.stop_reason = 'step_too_small'
                        return
                    step = max(step * settings['step_decrease_factor'], settings['min_step'])
                    failures = 0
