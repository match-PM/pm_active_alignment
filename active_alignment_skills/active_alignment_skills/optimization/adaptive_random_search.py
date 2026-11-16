"""Working-branch random search with a sliding-window one-fifth success rule."""

from collections import deque

import numpy as np

from .base import Optimizer


class AdaptiveRandomSearchOptimizer(Optimizer):
    """Adapt bounded uniform perturbations around the best observed point."""

    def optimize(self, session):
        """Adjust perturbation width from the recent fraction of successful moves."""
        settings = session.parameters
        sigma = settings['sigma']
        successes = deque(maxlen=settings['window_size'])
        while session.evaluations < settings['max_evaluations']:
            before = session.best.cost
            candidate = np.clip(session.best.position + session.rng.uniform(
                -sigma, sigma, len(session.initial_position)), 0, 1)
            session.evaluate(candidate)
            successes.append(session.best.cost < before)
            session.iterations += 1
            if len(successes) == settings['window_size']:
                # The one-fifth rule increases exploration above 20% successes
                # and shrinks the search neighborhood below 20% successes.
                rate = sum(successes) / settings['window_size']
                if rate > 0.2:
                    sigma = min(sigma * settings['alpha'], settings['max_sigma'])
                elif rate < 0.2:
                    sigma = max(sigma * settings['gamma'], settings['min_sigma'])
            session.solver_details['final_sigma'] = sigma
