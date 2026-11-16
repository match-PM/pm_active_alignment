"""Optional simulation diagnostics, separate from the optimization objective."""

import math
import threading
import time

from std_msgs.msg import Float32

from ..optimization.base import OptimizationCancelled


class MetricsReader:
    """Read fresh diagnostic values after a completed movement."""

    SUFFIXES = ('eta_lateral', 'eta_angular', 'eta_axial', 'eta_clean',
                'distance', 'angle_deviation', 'distance_percentage',
                'distance_lateral_percentage', 'distance_axial_percentage', 'angle_percentage')

    def __init__(self, node, prefix):
        self.node = node
        self.condition = threading.Condition()
        self.values = {}
        self.subscriptions = []
        if prefix:
            for suffix in self.SUFFIXES:
                self.subscriptions.append(node.create_subscription(
                    Float32, prefix.rstrip('/') + '/' + suffix,
                    lambda msg, key=suffix: self.receive(key, msg), 10,
                    callback_group=node.cb_subscriptions))

    def receive(self, key, message):
        with self.condition:
            self.values[key] = (float(message.data), time.monotonic())
            self.condition.notify_all()

    def snapshot(self, after, timeout, cancelled):
        if not self.subscriptions:
            return {}
        deadline = time.monotonic() + timeout
        with self.condition:
            while not all(
                    key in self.values and self.values[key][1] > after for key in self.SUFFIXES):
                if cancelled():
                    raise OptimizationCancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        'No fresh simulation diagnostics from metrics_topic_prefix.')
                self.condition.wait(min(0.05, remaining))
            result = {f'metric:{key}': self.values[key][0] for key in self.SUFFIXES}
            if not all(math.isfinite(value) for value in result.values()):
                raise ValueError('Simulation diagnostics contain non-finite values.')
            return result

    def close(self):
        for subscription in self.subscriptions:
            self.node.destroy_subscription(subscription)
        self.subscriptions.clear()
