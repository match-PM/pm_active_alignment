"""Classical bounded Nelder-Mead, reviewed from the working branch."""

import numpy as np

from .base import Optimizer, initial_simplex


class ManualNelderMeadOptimizer(Optimizer):
    """Reflection, expansion, outside/inside contraction, and shrink."""

    def optimize(self, session):
        """Maintain an n+1 point simplex in the normalized joint search box."""
        settings = session.parameters
        simplex = initial_simplex(session.initial_position, settings['simplex_step'])
        values = np.empty(len(simplex))
        values[0] = session.history[0].cost
        for i in range(1, len(simplex)):
            values[i] = session.evaluate(simplex[i])
        for _ in range(settings['max_iterations']):
            session.check_cancelled()
            order = np.argsort(values)
            # Sort from best (smallest objective) to worst before each simplex step.
            simplex, values = simplex[order], values[order]
            if (np.max(np.abs(simplex[1:] - simplex[0])) <= settings['xatol']
                    and np.max(np.abs(values[1:] - values[0])) <= settings['fatol']):
                session.stop_reason = 'converged'
                return
            centroid = np.mean(simplex[:-1], axis=0)
            # Reflection replaces the worst vertex with a point across the
            # centroid of the remaining vertices.
            reflected = np.clip(centroid + settings['alpha'] * (centroid - simplex[-1]), 0, 1)
            reflected_value = session.evaluate(reflected)
            session.iterations += 1
            if reflected_value < values[0]:
                # A better-than-best reflection may justify a longer expansion.
                expanded = np.clip(centroid + settings['beta'] * (reflected - centroid), 0, 1)
                expanded_value = session.evaluate(expanded)
                if expanded_value < reflected_value:
                    simplex[-1], values[-1] = expanded, expanded_value
                else:
                    simplex[-1], values[-1] = reflected, reflected_value
            elif reflected_value < values[-2]:
                # An ordinary improving reflection replaces only the worst vertex.
                simplex[-1], values[-1] = reflected, reflected_value
            else:
                # Contract toward the reflection if it beat the old worst point;
                # otherwise contract toward the old worst point itself.
                outside = reflected_value < values[-1]
                anchor = reflected if outside else simplex[-1]
                contracted = np.clip(centroid + settings['gamma'] * (anchor - centroid), 0, 1)
                contracted_value = session.evaluate(contracted)
                accepted = (contracted_value <= reflected_value if outside
                            else contracted_value < values[-1])
                if accepted:
                    simplex[-1], values[-1] = contracted, contracted_value
                else:
                    # A failed contraction shrinks all other vertices toward the best.
                    for i in range(1, len(simplex)):
                        simplex[i] = np.clip(
                            simplex[0] + settings['delta'] * (simplex[i] - simplex[0]), 0, 1)
                        values[i] = session.evaluate(simplex[i])
        session.stop_reason = 'max_iterations'
