"""Global or local uniform random-search baseline from the working branch."""

import numpy as np

from .base import Optimizer


class RandomSearchOptimizer(Optimizer):
    """Keep the best evaluated point while sampling bounded candidates."""

    def optimize(self, session):
        """Sample the whole search box or a neighborhood of the current best point."""
        settings = session.parameters
        dimension = len(session.initial_position)
        while session.evaluations < settings['max_evaluations']:
            if settings['use_global_search']:
                # Global sampling is a simple baseline without local search assumptions.
                candidate = session.rng.uniform(0, 1, dimension)
            else:
                # Local sampling keeps the same fixed perturbation width each time.
                candidate = np.clip(session.best.position + session.rng.uniform(
                    -settings['perturbation_scale'], settings['perturbation_scale'], dimension),
                    0, 1)
            session.evaluate(candidate)
            # Session.evaluate records rejected points too and updates the best
            # point only when the measured objective improves.
            session.iterations += 1
