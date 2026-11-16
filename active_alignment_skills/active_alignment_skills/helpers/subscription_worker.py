"""Own short-lived subscriptions without destroying live executor entities."""

import threading
import uuid

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node


class SubscriptionWorker:
    """Stop the executor before destroying any of its subscriptions."""

    def __init__(self, owner, label):
        self.node = Node(label + '_' + uuid.uuid4().hex[:8], context=owner.context,
                         use_global_arguments=False)
        self.node.cb_subscriptions = MutuallyExclusiveCallbackGroup()
        self.executor = SingleThreadedExecutor(context=owner.context)
        self.executor.add_node(self.node)
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def stop(self):
        self.executor.shutdown()
        if self.thread is not None:
            self.thread.join()

    def close(self):
        self.stop()
        self.node.destroy_node()
