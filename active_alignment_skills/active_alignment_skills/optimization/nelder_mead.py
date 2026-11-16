"""Library-based Nelder-Mead retained from the main branch."""

from scipy.optimize import minimize
import numpy as np

from .base import Optimizer, initial_simplex


class NelderMeadOptimizer(Optimizer):
    """Use SciPy while accounting for actual physical evaluations centrally."""

    def optimize(self, session):
        """Let SciPy manage the simplex while the session manages robot evaluations."""
        settings = session.parameters
        first_call = True

        def objective(position):
            nonlocal first_call
            session.check_cancelled()
            if first_call:
                first_call = False
                if not np.array_equal(position, session.initial_position):
                    raise RuntimeError('SciPy changed the first simplex vertex before evaluation.')
                # SciPy first evaluates initial_simplex[0], already measured.
                return session.history[0].cost
            return session.evaluate(position)

        result = minimize(
            objective, session.initial_position, method='Nelder-Mead',
            bounds=[(0.0, 1.0)] * len(session.initial_position),
            options={
                'maxiter': settings['max_iterations'],
                'maxfev': settings['max_evaluations'] + 1,
                'xatol': settings['xatol'], 'fatol': settings['fatol'],
                'adaptive': settings['adaptive'],
                'initial_simplex': initial_simplex(
                    session.initial_position, settings['simplex_step']),
            })
        session.iterations = int(result.nit)
        # Reaching either limit is a normal completed run; unexpected solver
        # failures propagate to the action server as errors.
        session.solver_details = {'message': str(result.message), 'nfev': int(result.nfev)}
        if result.success:
            session.stop_reason = 'converged'
        elif result.status == 1:
            session.stop_reason = 'max_evaluations'
        elif result.status == 2:
            session.stop_reason = 'max_iterations'
        else:
            raise RuntimeError(f'SciPy Nelder-Mead failed: {result.message}')
