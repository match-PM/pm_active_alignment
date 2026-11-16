"""Fresh, finite, weighted observations from all requested objective topics."""

import math
import threading
import time

from std_msgs.msg import Float32

from active_alignment_skills.optimization.base import OptimizationCancelled


class OptimizationValueManager:
    """Convert maximization topics to a common minimization cost exactly once."""

    def __init__(self, node):
        self.node = node
        self.condition = threading.Condition()
        self.topics = {}
        self.values = {}
        self.subscriptions = []
        self.value_output_history = []
        self.last_read = 0.0
        self.eval_value_pub = node.create_publisher(Float32, 'active_alignment/eval_value', 10)

    def init(self, alignment_topics):
        self.clear()
        if not alignment_topics:
            raise ValueError('At least one alignment topic is required.')
        for topic in alignment_topics:
            if not topic.topic_name or topic.topic_name in self.topics:
                raise ValueError('Alignment topics must have unique, nonempty names.')
            if not math.isfinite(topic.weight_factor) or topic.weight_factor <= 0:
                raise ValueError('Every topic weight must be finite and positive.')
            self.topics[topic.topic_name] = (topic.weight_factor, topic.minimize_not_maximize)
        for name in self.topics:
            self.subscriptions.append(self.node.create_subscription(
                Float32, name, lambda msg, key=name: self._receive(key, msg), 10,
                callback_group=self.node.cb_subscriptions))

    def _receive(self, name, msg):
        with self.condition:
            if name in self.topics:
                self.values[name] = (float(msg.data), time.monotonic())
                self.condition.notify_all()

    def wait_for_fresh(self, after, timeout, cancelled=lambda: False):
        """Wait until every objective source has a receipt newer than after.

        The caller captures after when motion and optional settling have finished.
        No fixed measurement delay is used: the condition wakes on topic updates.
        """
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if cancelled():
                    raise OptimizationCancelled('Cancelled while waiting for measurements.')
                ready = self.topics and all(
                    name in self.values and self.values[name][1] > after
                    for name in self.topics)
                if ready:
                    raw = {name: self.values[name][0] for name in self.topics}
                    if not all(math.isfinite(value) for value in raw.values()):
                        raise ValueError('An alignment topic published a non-finite measurement.')
                    total_weight = sum(weight for weight, _ in self.topics.values())
                    cost = sum(raw[name] * weight * (1 if minimize else -1)
                               for name, (weight, minimize) in self.topics.items()) / total_weight
                    self.last_read = time.monotonic()
                    self.value_output_history.append(cost)
                    self.eval_value_pub.publish(Float32(data=float(cost)))
                    return float(cost), raw
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [name for name in self.topics
                               if name not in self.values or self.values[name][1] <= after]
                    raise TimeoutError(f'No fresh post-movement data from: {missing}')
                self.condition.wait(min(0.05, remaining))

    def get_eval_value(self, wait_for_update_sec=5, check_for_new_values=False):
        after = self.last_read if check_for_new_values else -math.inf
        return self.wait_for_fresh(after, wait_for_update_sec)[0]

    def clear(self):
        for subscription in self.subscriptions:
            self.node.destroy_subscription(subscription)
        self.subscriptions.clear()
        with self.condition:
            self.topics.clear()
            self.values.clear()
            self.value_output_history.clear()
            self.last_read = 0.0
            self.condition.notify_all()
