"""Wait for fresh setup-verification samples without polling stale defaults."""

import math
import threading
import time

from std_msgs.msg import Float32

from active_alignment_skills.optimization.base import OptimizationCancelled
from active_alignment_skills.helpers.subscription_worker import SubscriptionWorker


class SignalVerifier:
    """Verify the clean simulated signal, independently of measurement noise."""

    def __init__(self, node, topic):
        self.worker = SubscriptionWorker(node, 'batch_verification')
        self.node = self.worker.node
        self.condition = threading.Condition()
        self.value = None
        self.received = 0.0
        self.subscription = self.node.create_subscription(
            Float32, topic, self.receive, 10, callback_group=self.node.cb_subscriptions)
        self.worker.start()

    def receive(self, message):
        with self.condition:
            self.value = float(message.data)
            self.received = time.monotonic()
            self.condition.notify_all()

    def verify(self, expected, tolerance, timeout, cancelled):
        after = time.monotonic()
        deadline = after + timeout
        with self.condition:
            while True:
                if cancelled():
                    raise OptimizationCancelled()
                if (self.received > after and math.isfinite(self.value)
                        and abs(self.value - expected) <= tolerance):
                    return self.value
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'Initial signal verification failed: expected {expected}, '
                                       f'observed {self.value}.')
                self.condition.wait(min(0.05, remaining))

    def close(self):
        self.worker.close()
