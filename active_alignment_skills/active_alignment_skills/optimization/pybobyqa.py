"""Optional library-based bounded trust-region optimization."""

from contextlib import contextmanager
import threading

import numpy as np

from .base import Optimizer


_RANDOM_LOCK = threading.Lock()


@contextmanager
def solver_random_seed(seed):
    """Py-BOBYQA uses NumPy's legacy global RNG rather than an injected generator."""
    with _RANDOM_LOCK:
        previous = np.random.get_state()
        np.random.seed(seed % (2**32))
        try:
            yield
        finally:
            np.random.set_state(previous)


class PyBobyqaOptimizer(Optimizer):
    """Keep dependency loading local so other optimizers remain usable."""

    def optimize(self, session):
        """Fit bounded quadratic models using Py-BOBYQA's trust-region solver."""
        try:
            import pybobyqa
        except ImportError as exc:
            raise RuntimeError('Py-BOBYQA is unavailable; install the Py-BOBYQA package.') from exc
        settings = session.parameters
        dimension = len(session.initial_position)
        maximum_npt = (dimension + 1) * (dimension + 2) // 2
        # npt=0 selects the full quadratic interpolation set for the active dimension.
        npt = settings['npt'] or maximum_npt
        if not dimension + 1 <= npt <= maximum_npt:
            raise ValueError(f'npt must be between {dimension + 1} and {maximum_npt}.')
        if settings['max_evaluations'] < npt - 1:
            raise ValueError('max_evaluations is too small for the interpolation set.')
        first_call = True

        def objective(position):
            nonlocal first_call
            session.check_cancelled()
            if first_call:
                first_call = False
                if not np.allclose(position, session.initial_position, rtol=0, atol=1e-12):
                    raise RuntimeError('Py-BOBYQA changed x0 before its first evaluation.')
                return session.history[0].cost
            return session.evaluate(position)

        with solver_random_seed(settings['random_seed']):
            # The library's initial request reuses the separate starting sample;
            # every later objective request consumes a physical measurement.
            result = pybobyqa.solve(
                objective, session.initial_position.copy(),
                bounds=(np.zeros(dimension), np.ones(dimension)), npt=npt,
                rhobeg=settings['initial_tr_radius'], rhoend=settings['final_tr_radius'],
                maxfun=settings['max_evaluations'] + 1,
                objfun_has_noise=settings['noise_aware'],
                user_params={
                    # Session owns the optional target; cost can have any finite scale/sign.
                    'model.abs_tol': -float('inf'),
                    # Symmetric axis samples around a narrow peak can make the
                    # interpolation set singular during geometry repair/restarts.
                    # Seeded random directions break that symmetry.
                    'init.random_initial_directions': settings['random_initial_directions'],
                    'restarts.use_restarts': settings['use_restarts'],
                    'noise.quit_on_noise_level': settings['quit_on_noise_level'],
                }, do_logging=False, print_progress=False)
        session.solver_details = {'flag': int(result.flag), 'message': str(result.msg),
                                  'nfev': int(result.nf)}
        if result.flag < 0:
            message = f'Py-BOBYQA failed: {result.msg}'
            if result.flag == result.EXIT_LINALG_ERROR:
                message += (
                    ' Hint: set pybobyqa.random_initial_directions=true and retry with '
                    f'a different general.random_seed (currently {settings["random_seed"]}). '
                    'Check history.csv: if most signal samples are flat/zero, reduce '
                    'pybobyqa.initial_tr_radius to match the signal width; keep '
                    '0 < pybobyqa.final_tr_radius < pybobyqa.initial_tr_radius and '
                    'avoid steps below the robot/sensor resolution. '
                    'If it persists, try nelder_mead or adaptive_hill_climb.')
            raise RuntimeError(message)
        session.stop_reason = ('max_evaluations' if session.evaluations
                               >= settings['max_evaluations'] else 'solver_termination')
