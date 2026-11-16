"""Shared evaluation accounting and result types for every optimizer."""

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np


class OptimizationCancelled(Exception):
    """The client cancelled the current run."""


class BudgetReached(Exception):
    """All permitted post-initial measurements have been consumed."""


class ObjectiveThresholdReached(Exception):
    """The weighted minimization objective reached its optional stopping threshold."""


@dataclass
class Measurement:
    """One evaluated candidate; positions and observations belong together."""

    position: np.ndarray
    cost: float
    elapsed: float
    metrics: dict = field(default_factory=dict)


class OptimizationSession:
    """Apply uniform bounds, cancellation, budgets, and best-point tracking.

    The objective always minimizes. Maximization is converted once by the
    measurement adapter, using each AlignTopic's minimize_not_maximize flag.
    The initial measurement is recorded but excluded from max_evaluations.
    """

    def __init__(self, initial_position, parameters: dict, measure: Callable,
                 cancelled: Callable = lambda: False, observe: Callable = None):
        self.initial_position = np.asarray(initial_position, dtype=float)
        if (self.initial_position.ndim != 1 or self.initial_position.size == 0
                or not np.all(np.isfinite(self.initial_position))
                or np.any(self.initial_position < 0) or np.any(self.initial_position > 1)):
            raise ValueError('Initial position must be a finite vector within [0, 1].')
        self.parameters = dict(parameters)
        self.measure = measure
        self.cancelled = cancelled
        self.observe = observe
        self.rng = np.random.default_rng(parameters['random_seed'])
        self.history = []
        self.best = None
        self.evaluations = 0
        self.iterations = 0
        self.stop_reason = 'not_started'
        self.solver_details = {}
        self.started = time.monotonic()

    def check_cancelled(self):
        if self.cancelled():
            raise OptimizationCancelled('Optimization cancelled.')

    def evaluate(self, position, *, initial=False):
        """Move and measure one candidate through the supplied measurement adapter."""
        self.check_cancelled()
        if initial and self.history:
            raise ValueError('The initial measurement can only be recorded once.')
        if not initial and self.evaluations >= self.parameters['max_evaluations']:
            raise BudgetReached()
        candidate = np.asarray(position, dtype=float)
        if (candidate.shape != self.initial_position.shape
                or not np.all(np.isfinite(candidate))):
            raise ValueError('Candidate has an invalid shape or non-finite values.')
        candidate = np.clip(candidate, 0.0, 1.0)
        # The adapter waits for verified motion, optional settling, and fresh data.
        # Algorithms never read cached ROS topics or implement their own sleeps.
        observation = self.measure(candidate.copy())
        if isinstance(observation, tuple):
            cost, metrics = observation
        else:
            cost, metrics = observation, {}
        if not np.isfinite(cost):
            raise ValueError('Objective returned a non-finite value.')
        record = Measurement(candidate.copy(), float(cost),
                             time.monotonic() - self.started, dict(metrics))
        self.history.append(record)
        # Keep the position, objective, and diagnostics in one record so the
        # returned best point and the best logged row always describe the same move.
        if not initial:
            self.evaluations += 1
        if self.best is None or record.cost < self.best.cost:
            self.best = record
        if self.observe is not None:
            self.observe(record)
        self.check_cancelled()
        if (self.parameters['stop_on_objective_threshold']
                and record.cost <= self.parameters['objective_threshold']):
            raise ObjectiveThresholdReached()
        return record.cost


class Optimizer:
    """Small strategy interface; mutable search state belongs to one run."""

    def optimize(self, session: OptimizationSession):
        raise NotImplementedError


def run_optimizer(optimizer: Optimizer, session: OptimizationSession):
    """Record the initial point and handle normal search termination only."""
    try:
        # The starting point is measured once, outside the search budget.
        session.evaluate(session.initial_position, initial=True)
        if session.parameters['max_evaluations'] == 0:
            session.stop_reason = 'max_evaluations'
            return session
        optimizer.optimize(session)
        if session.stop_reason == 'not_started':
            session.stop_reason = ('max_evaluations' if session.evaluations
                                   >= session.parameters['max_evaluations'] else 'converged')
    # Budget and threshold exits are normal outcomes. Cancellation and errors
    # propagate to the ROS node so it can report the correct action state.
    except BudgetReached:
        session.stop_reason = 'max_evaluations'
    except ObjectiveThresholdReached:
        session.stop_reason = 'objective_threshold_reached'
    return session


def initial_simplex(position, step):
    """Build distinct vertices even when an initial coordinate is at a bound."""
    position = np.asarray(position, dtype=float)
    simplex = np.tile(position, (len(position) + 1, 1))
    for axis in range(len(position)):
        direction = 1 if position[axis] <= 0.5 else -1
        simplex[axis + 1, axis] = np.clip(position[axis] + direction * step, 0.0, 1.0)
    return simplex
