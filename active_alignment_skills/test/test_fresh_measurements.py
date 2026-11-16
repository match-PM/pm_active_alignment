"""Freshness is tied to movement completion, independently of settling duration."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import time
from types import SimpleNamespace

import pytest
from active_alignment_interfaces.msg import AlignTopic

from active_alignment_skills.py_modules.OptimizationValueManager import OptimizationValueManager


class ObservationNode:
    """Minimal ROS adapter: this test drives receipt times and topic callbacks directly."""

    cb_subscriptions = None

    def create_publisher(self, *args):
        return SimpleNamespace(publish=lambda message: None)

    def create_subscription(self, *args, **kwargs):
        return object()

    def destroy_subscription(self, subscription):
        pass


@pytest.mark.parametrize('minimize, expected_cost', [(True, 3.0), (False, -3.0)])
def test_positive_signal_uses_requested_optimization_direction(minimize, expected_cost):
    manager = OptimizationValueManager(ObservationNode())
    manager.init([AlignTopic(topic_name='/signal', minimize_not_maximize=minimize)])
    after = time.monotonic()
    manager._receive('/signal', SimpleNamespace(data=3.0))
    cost, raw = manager.wait_for_fresh(after, 0.1)
    assert cost == expected_cost
    assert raw == {'/signal': 3.0}
    manager.clear()


def test_message_defaults_use_unit_weight_and_minimization():
    manager = OptimizationValueManager(ObservationNode())
    manager.init([AlignTopic(topic_name='/signal')])
    assert manager.topics == {'/signal': (1.0, True)}
    manager.clear()


def test_every_topic_must_be_newer_than_completed_movement():
    manager = OptimizationValueManager(ObservationNode())
    manager.init([
        SimpleNamespace(topic_name='/first', weight_factor=1.0, minimize_not_maximize=True),
        SimpleNamespace(topic_name='/second', weight_factor=1.0, minimize_not_maximize=True),
    ])
    # Both cached values exist before the movement-completed boundary. Neither
    # may satisfy the next measurement, even though no settling delay is used.
    manager._receive('/first', SimpleNamespace(data=100.0))
    manager._receive('/second', SimpleNamespace(data=200.0))
    movement_finished = time.monotonic()

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(manager.wait_for_fresh, movement_finished, 1.0)
        with pytest.raises(FutureTimeout):
            result.result(timeout=0.02)

        # Updating only one source must still leave the measurement waiting.
        manager._receive('/first', SimpleNamespace(data=2.0))
        with pytest.raises(FutureTimeout):
            result.result(timeout=0.02)

        manager._receive('/second', SimpleNamespace(data=4.0))
        cost, values = result.result(timeout=0.5)
    assert cost == 3.0
    assert values == {'/first': 2.0, '/second': 4.0}
    manager.clear()
