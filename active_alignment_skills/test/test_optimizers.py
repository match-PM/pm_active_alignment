"""Numerical and lifecycle regression checks without ROS or robot hardware."""

import numpy as np
import pytest

from active_alignment_skills.optimization.base import (
    OptimizationCancelled, OptimizationSession, initial_simplex, run_optimizer)
from active_alignment_skills.optimization.registry import OPTIMIZERS
from active_alignment_skills.helpers.parameters import load_defaults, optimizer_parameters


def settings(name, **changes):
    return {**optimizer_parameters(load_defaults(), name), **changes}


@pytest.mark.parametrize('name', OPTIMIZERS)
def test_known_quadratic_improves_and_stays_inside_bounds(name):
    if name == 'pybobyqa':
        pytest.importorskip('pybobyqa')
    target = np.array([0.73, 0.29])
    observed = []

    def objective(position):
        assert np.all((position >= 0) & (position <= 1))
        observed.append(position.copy())
        return float(np.sum((position - target)**2))

    session = OptimizationSession([0.5, 0.5], settings(name, max_evaluations=200), objective)
    run_optimizer(OPTIMIZERS[name](), session)
    assert session.best.cost < 0.005
    assert session.best.cost == min(record.cost for record in session.history)
    assert session.evaluations <= 200
    assert len(observed) == session.evaluations + 1 == len(session.history)
    if name.startswith('nelder_mead'):
        assert np.linalg.norm(session.best.position - target) < 1e-4


@pytest.mark.parametrize('name', OPTIMIZERS)
@pytest.mark.parametrize('budget', [0, 1, 5])
def test_budget_excludes_exactly_one_initial_measurement(name, budget):
    if name == 'pybobyqa':
        pytest.importorskip('pybobyqa')
        if 0 < budget < 5:
            pytest.skip('Py-BOBYQA needs five post-initial points for a 2D quadratic model.')
    session = OptimizationSession([0.5, 0.5], settings(name, max_evaluations=budget),
                                  lambda x: float(np.sum((x - 0.31)**2)))
    run_optimizer(OPTIMIZERS[name](), session)
    assert session.evaluations <= budget
    assert len(session.history) == session.evaluations + 1
    assert session.stop_reason in (
        'max_evaluations',
        'no_improvement',
        'converged',
        'solver_termination')


@pytest.mark.parametrize('name', OPTIMIZERS)
@pytest.mark.parametrize('cost, threshold, reached', [
    (0.2, 0.3, True), (0.4, 0.3, False),
    (-0.8, -0.9, False), (-1.0, -0.9, True),
    (-0.4, -0.3, True), (-0.2, -0.3, False),
    (0.5, 0.5, True), (-0.5, -0.5, True),
])
def test_signed_objective_target_on_initial_point(name, cost, threshold, reached):
    session = OptimizationSession([0.3], settings(
        name, max_evaluations=0,
        stop_on_objective_threshold=True, objective_threshold=threshold),
        lambda _: cost)
    run_optimizer(OPTIMIZERS[name](), session)
    assert session.stop_reason == ('objective_threshold_reached' if reached else 'max_evaluations')
    assert session.evaluations == 0
    assert len(session.history) == 1


@pytest.mark.parametrize('name', OPTIMIZERS)
@pytest.mark.parametrize('minimize', [True, False])
def test_signed_objective_target_stops_after_the_measurement_that_reaches_it(name, minimize):
    if name == 'pybobyqa':
        pytest.importorskip('pybobyqa')
    raw_values = iter([1.0, 0.2] if minimize else [0.2, 0.9])

    def measure(position):
        raw = next(raw_values)
        return (raw if minimize else -raw), {'topic:/signal': raw}

    session = OptimizationSession([0.5], settings(
        name, max_evaluations=20, stop_on_objective_threshold=True,
        objective_threshold=0.3 if minimize else -0.8), measure)
    run_optimizer(OPTIMIZERS[name](), session)
    assert session.stop_reason == 'objective_threshold_reached'
    assert session.evaluations == 1
    assert len(session.history) == 2
    assert session.best is session.history[-1]


@pytest.mark.parametrize('name', OPTIMIZERS)
def test_cancellation_propagates_instead_of_returning_success(name):
    session = OptimizationSession([0.5], settings(name), lambda _: 1.0, cancelled=lambda: True)
    with pytest.raises(OptimizationCancelled):
        run_optimizer(OPTIMIZERS[name](), session)
    assert not session.history


@pytest.mark.parametrize('name', ['nelder_mead', 'nelder_mead_manual', 'pybobyqa'])
def test_bound_initialization_finds_interior_optimum(name):
    if name == 'pybobyqa':
        pytest.importorskip('pybobyqa')
    session = OptimizationSession([1.0, 0.0], settings(name, max_evaluations=200),
                                  lambda x: float(np.sum((x - [0.8, 0.2])**2)))
    run_optimizer(OPTIMIZERS[name](), session)
    assert session.best.cost < 1e-8


def test_manual_and_scipy_classical_simplex_agree_away_from_bounds():
    histories = []
    for name in ('nelder_mead', 'nelder_mead_manual'):
        session = OptimizationSession([0.5, 0.5], settings(name),
                                      lambda x: float(np.sum((x - [0.6, 0.4])**2)))
        run_optimizer(OPTIMIZERS[name](), session)
        histories.append(session)
    np.testing.assert_allclose(histories[0].best.position, histories[1].best.position, atol=1e-10)


def test_initial_simplex_is_not_degenerate_at_upper_bound():
    simplex = initial_simplex([1.0, 0.0, 1.0], 0.05)
    assert np.linalg.matrix_rank(simplex[1:] - simplex[0]) == 3


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf')])
def test_nonfinite_measurements_are_errors(bad_value):
    session = OptimizationSession([0.5], settings('hill_climb'), lambda _: bad_value)
    with pytest.raises(ValueError, match='non-finite'):
        run_optimizer(OPTIMIZERS['hill_climb'](), session)


def test_seeded_search_is_reproducible():
    histories = []
    for _ in range(2):
        session = OptimizationSession([0.5, 0.5], settings(
            'random_search'), lambda x: float(x @ x))
        run_optimizer(OPTIMIZERS['random_search'](), session)
        histories.append([record.position for record in session.history])
    np.testing.assert_array_equal(histories[0], histories[1])


def test_pybobyqa_handles_negative_scaled_cost_and_restores_global_rng():
    pytest.importorskip('pybobyqa')
    previous = np.random.get_state()
    session = OptimizationSession([0.5], settings(
        'pybobyqa', noise_aware=False, use_restarts=False, max_evaluations=30),
        lambda x: float(-1e21 + 1e20 * (x[0] - 0.7)**2))
    run_optimizer(OPTIMIZERS['pybobyqa'](), session)
    assert session.evaluations > 0
    assert session.best.cost < session.history[0].cost
    current = np.random.get_state()
    np.testing.assert_array_equal(current[1], previous[1])
    assert current[0] == previous[0]
    assert current[2:] == previous[2:]


@pytest.mark.parametrize('seed', [4, 12, 13, 16])
def test_pybobyqa_handles_noisy_narrow_peak_without_singular_geometry(seed):
    pytest.importorskip('pybobyqa')
    import warnings
    from scipy.linalg import LinAlgWarning

    # A micrometre-scale signal over millimetre-scale travel: the axis samples
    # are nearly flat, while the starting point is near the peak. Coordinate
    # initialization in Py-BOBYQA 1.5.0 fails on these seeded observations.
    center = np.array([0.500100012962504, 0.5001000175328759, 0.5001000222904899])
    noise = np.random.default_rng(seed)

    def objective(position):
        return float(-0.306 * np.exp(-np.sum(((position - center) / 0.0002)**2))
                     + noise.normal(0, 0.0003))

    session = OptimizationSession(center, settings('pybobyqa', random_seed=seed), objective)
    with warnings.catch_warnings():
        warnings.simplefilter('error', LinAlgWarning)
        run_optimizer(OPTIMIZERS['pybobyqa'](), session)
    assert session.solver_details['flag'] >= 0
    assert session.evaluations <= session.parameters['max_evaluations']
    assert len(session.history) == session.evaluations + 1
    assert session.best.cost <= session.history[0].cost


@pytest.mark.parametrize('flag', [-3, -1])
def test_pybobyqa_failure_explains_singular_geometry_without_hiding_errors(monkeypatch, flag):
    pybobyqa = pytest.importorskip('pybobyqa')
    from types import SimpleNamespace

    def fail_solver(objective, initial_position, **kwargs):
        objective(initial_position)
        return SimpleNamespace(flag=flag, msg='injected solver failure', nf=1,
                               EXIT_LINALG_ERROR=-3)

    monkeypatch.setattr(pybobyqa, 'solve', fail_solver)
    session = OptimizationSession([0.5, 0.5], settings('pybobyqa', random_seed=7),
                                  lambda position: float(np.sum(position**2)))
    with pytest.raises(RuntimeError, match='injected solver failure') as error:
        run_optimizer(OPTIMIZERS['pybobyqa'](), session)
    message = str(error.value)
    if flag == -3:
        assert 'random_initial_directions=true' in message
        assert 'general.random_seed (currently 7)' in message
        assert 'history.csv' in message
        assert 'pybobyqa.initial_tr_radius' in message
        assert 'robot/sensor resolution' in message
        assert 'nelder_mead or adaptive_hill_climb' in message
    else:
        assert 'Hint:' not in message
    assert session.solver_details['flag'] == flag
    assert session.evaluations == 0
    assert len(session.history) == 1
