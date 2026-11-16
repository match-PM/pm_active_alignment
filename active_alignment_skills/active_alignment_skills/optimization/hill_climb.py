"""Working-branch coordinate/simultaneous hill climbing."""

import numpy as np

from .base import Optimizer


class HillClimbOptimizer(Optimizer):
    """Try each direction from the current best point, with optional adaptation."""

    def optimize(self, session):
        """Sweep the active joints, accepting only strictly smaller objective values."""
        settings = session.parameters
        step = settings['step_size']
        dimension = len(session.initial_position)
        while session.evaluations < settings['max_evaluations']:
            session.check_cancelled()
            improved = False
            # Coordinate mode tests one joint at a time. Simultaneous mode moves
            # all coordinates together, matching the working-branch variant.
            if settings['coordinate_wise']:
                directions = np.eye(dimension)
            else:
                directions = np.ones((1, dimension))
            for direction in directions:
                origin = session.best.position.copy()
                previous = session.best.cost
                for sign in (1, -1):
                    candidate = np.clip(origin + sign * step * direction, 0.0, 1.0)
                    if np.array_equal(candidate, origin):
                        continue
                    # Try the opposite direction only if the positive step fails.
                    session.evaluate(candidate)
                    if session.best.cost < previous:
                        improved = True
                        break
            session.iterations += 1
            if settings['adaptive_step']:
                # Successful sweeps explore farther; unsuccessful sweeps refine
                # the search until no improvement is found at the minimum step.
                if improved:
                    step = min(step * settings['step_increase_factor'], settings['max_step'])
                else:
                    if step <= settings['min_step']:
                        session.stop_reason = 'step_too_small'
                        return
                    step = max(step * settings['step_decrease_factor'], settings['min_step'])
            elif not improved:
                session.stop_reason = 'no_improvement'
                return
