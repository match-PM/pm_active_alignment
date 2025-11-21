import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from active_alignment_interfaces.msg import AlignTopic  # <-- your action interface
import time

class AlignTopicEntry(AlignTopic):
    def __init__(self, align_topic: AlignTopic, node:Node):
        super().__init__()
        self.topic_name = align_topic.topic_name
        self.weight_factor = align_topic.weight_factor
        self.minimize_not_maximize = align_topic.minimize_not_maximize
        self.node = node

        self.subscription = self.node.create_subscription(
            Float32,
            self.topic_name,
            self.topic_callback,
            10
        )
        self.node.get_logger().info(f"Created subscription to topic: {self.topic_name}")
        self.current_value = None
        self.current_timestamp = None
        self.value_history: list[float] = []

    def topic_callback(self, msg: Float32):
        self.current_value = msg.data
        self.current_timestamp = self.node.get_clock().now()
        #self.node.get_logger().info(f"Received message on {self.topic_name}: {msg.data}")

    def destroy_subscription(self):
        self.node.destroy_subscription(self.subscription)
        self.node.get_logger().info(f"Destroyed subscription to topic: {self.topic_name}")
    

# class AlginTopicManager:
#     """
#     Manages alignment topics for active alignment skills.
#     """

#     def __init__(self, node: Node):
#         self.node = node
#         self.alignment_topics: list[AlignTopicEntry] = []

#     def append_topic(self, topic: AlignTopic):
#         self.alignment_topics.append(AlignTopicEntry(topic))
#         self.node.get_logger().info(f"Appended alignment topic: {topic.topic_name}")
    
class OptimizationValueManager:
    """
    Manages optimization values for active alignment skills.
    """

    def __init__(self, node: Node):
        self.node = node
        self.optimization_values: dict[str, float] = {}
        
        self.eval_value_pub = self.node.create_publisher(Float32, 'active_alignment/eval_value', 10)
        self.eval_value_timer = self.node.create_timer(0.1, self._publish_eval_value)

        self.alignment_topics: list[AlignTopicEntry] = []

        self.last_update_dict: dict[str, rclpy.time.Time] = {}

        self.value_output_history: list[float] = []

    def _append_topic(self, topic: AlignTopic):
        if self._check_topic_exists(topic.topic_name):
            self.node.get_logger().error(f"Topic {topic.topic_name} already exists in OptimizationValueManager. Cannot append duplicate.")
            raise ValueError(f"Topic {topic.topic_name} already exists in OptimizationValueManager.")
        self.alignment_topics.append(AlignTopicEntry(topic,self.node))
        self.node.get_logger().info(f"Appended alignment topic: {topic.topic_name}")

    def _check_topic_exists(self, topic_name: str) -> bool:
        for topic_entry in self.alignment_topics:
            if topic_entry.topic_name == topic_name:
                return True
        return False
    
    def _publish_eval_value(self):

        if not self._check_values_available():
            return  # don’t publish yet
        eval_value = self._calc_output_signal()
        msg = Float32()
        msg.data = eval_value
        self.eval_value_pub.publish(msg)
        self.node.get_logger().info(f"Published eval value: {eval_value}")

    def _check_values_available(self) -> bool:
        if len(self.alignment_topics) == 0:
            return False
        for topic_entry in self.alignment_topics:
            if topic_entry.current_value is None:
                return False
        return True
    
    def _get_eval_value(self, add_to_history = False) -> float:
        if not self._check_values_available():
            return None
        return self._calc_output_signal(add_to_history = add_to_history)

        
    def _calc_output_signal(self, add_to_history = False) -> float:
        total_weight = 0.0
        weighted_sum = 0.0

        for topic_entry in self.alignment_topics:
            if topic_entry.current_value is not None:
                weight = topic_entry.weight_factor
                value = topic_entry.current_value
                
                if weight <= 0:
                    self.node.get_logger().warning(f"Weight factor for topic {topic_entry.topic_name} is non-positive ({weight}). Skipping this topic in eval calculation.")
                    continue
                    
                if add_to_history:
                    topic_entry.value_history.append(value)

                # If minimizing, invert the value
                if not topic_entry.minimize_not_maximize:
                    value = -value

                weighted_sum += weight * value
                total_weight += weight

        if total_weight == 0:
            return 0.0  # Avoid division by zero

        return weighted_sum / total_weight
    
    # def get_eval_value(self, wait_for_update_sec: int = 0, check_for_new_values = False) -> float:
    #     """
    #     Get the current evaluation value.
    #     If wait_for_update_sec > 0, waits up to that many seconds for a valid value.
    #     If no valid value is received, raises a ValueError.
    #     If wait_for_update_sec == 0, returns the current value or raises ValueError if none.
        
    #     Returns:
    #         float: The current evaluation value.
    #         Raises:
    #             ValueError: If no valid evaluation value is received within the wait time.
    #     """
            
    #     if wait_for_update_sec > 0:
    #         best_eval = None

    #         for _ in range(wait_for_update_sec * 10):
    #             best_eval = self._get_eval_value()
    #             if best_eval is not None:
    #                 return best_eval
    #             time.sleep(0.1)  # wait for eval to update
    #         raise ValueError("No eval value received during optimization.")
        
    #     else:   
    #         value = self._get_eval_value()

    #         if value is None:
    #             raise ValueError("No eval value received during optimization.")
    #         return value

    def get_eval_value(self, 
                       wait_for_update_sec: int = 0, 
                       check_for_new_values: bool = False):
        """
        Returns the evaluation value.
        
        If check_for_new_values=True:
            waits until at least ONE topic has published NEW data since the LAST call.
        If wait_for_update_sec > 0:
            waits up to that many seconds for a valid evaluation value.
        If both are set, it first waits for new data, then for a valid eval value.
        If neither is set, returns the current eval value or raises ValueError if none.
        """
        sleep_interval = 0.1
        # ----------------------------------------------
        # If "check_for_new_values" is active — wait for update
        # ----------------------------------------------
        if check_for_new_values:
            waited = 0.0
            while waited < wait_for_update_sec:
                if self._topics_have_new_data():
                    break
                time.sleep(sleep_interval)
                waited += sleep_interval
                if self._check_values_available():
                    self.node.get_logger().warning("Waiting for new data after the last read... This has a negative impact on optimization speed. Consider increasing the time for publishing alignment topics.")
            else:
                raise ValueError(f"Timeout: No data received from topics publisher in {waited} seconds.")

        # ----------------------------------------------
        # Regular wait for eval value if needed
        # ----------------------------------------------
        if wait_for_update_sec > 0 and not check_for_new_values:
            waited = 0.0
            while waited < wait_for_update_sec:
                value = self._get_eval_value(add_to_history=True)
                if value is not None:
                    # Save new timestamps
                    self._update_last_timestamps()
                    self.value_output_history.append(value)
                    return value
                
                time.sleep(sleep_interval)
                waited += sleep_interval
            raise ValueError("Timeout: No evaluation value received.")

        # ----------------------------------------------
        # Instant check (no waiting)
        # ----------------------------------------------
        value = self._get_eval_value(add_to_history=True)
        if value is None:
            raise ValueError("No evaluation value available.")
        
        # Save timestamps AFTER successfully getting value
        self._update_last_timestamps()
        self.value_output_history.append(value)
        
        return value

    def _update_last_timestamps(self):
        """
        Store the current timestamps after a successful evaluation read.
        """
        for topic_entry in self.alignment_topics:
            self.last_update_dict[topic_entry.topic_name] = topic_entry.current_timestamp

    def _topics_have_new_data(self) -> bool:
        """
        Returns True if ANY topic has a timestamp newer than self.last_update_dict.
        """
        for topic_entry in self.alignment_topics:
            name = topic_entry.topic_name
            previous = self.last_update_dict.get(name, None)
            current = topic_entry.current_timestamp

            if current is None:
                continue  # topic has not published yet → no new data

            if previous is None:
                return True  # first-time publication counts as new

            if current > previous:
                return True

        return False

    def init(self, alignment_topics: list[AlignTopic]):
        """
        Initialize the optimization value manager with alignment topics.
        Creates subscriptions to the specified topics.
        """
        #self.eval_dict.clear()
        for topic in alignment_topics:
            topic: AlignTopic
            self._append_topic(topic)
        self.last_update_dict.clear()

    def clear(self):
        """
        Clear all subscriptions and internal data.
        """
        for topic_entry in self.alignment_topics:
            topic_entry.destroy_subscription()
        self.alignment_topics.clear()
        self.last_update_dict.clear()

        # for sub in self.subscription_list:
        #     self.node.destroy_subscription(sub["subscription"])
        #     self.node.get_logger().info(f"Destroyed subscription to topic: {sub['topic_name']}")
        # self.subscription_list = []
        # self.eval_dict.clear()


    # def publish_eval_value(self):
    #     valid_values = [v for v in self.eval_dict.values() if v is not None]
    #     if not valid_values:
    #         return  # don’t publish yet

    #     eval_value = sum(valid_values) / len(valid_values)
    #     msg = Float32()
    #     msg.data = eval_value
    #     self.eval_value_pub.publish(msg)
    #     #self.get_logger().info(f"Published eval value: {eval_value}")

    # def _get_eval_value(self) -> float:
    #     valid_values = [v for v in self.eval_dict.values() if v is not None]
    #     if not valid_values:
    #         return None
    #     return sum(valid_values) / len(valid_values)
    
    # def get_eval_value(self, wait_for_update_sec: int = 0) -> float:
    #     """
    #     Get the current evaluation value.
    #     If wait_for_update_sec > 0, waits up to that many seconds for a valid value.
    #     If no valid value is received, raises a ValueError.
    #     If wait_for_update_sec == 0, returns the current value or raises ValueError if none.
        
    #     Returns:
    #         float: The current evaluation value.
    #         Raises:
    #             ValueError: If no valid evaluation value is received within the wait time.
    #     """
            
    #     if wait_for_update_sec > 0:
    #         best_eval = None

    #         for _ in range(wait_for_update_sec * 10):
    #             best_eval = self._get_eval_value()
    #             if best_eval is not None:
    #                 return best_eval
    #             time.sleep(0.1)  # wait for eval to update
    #         raise ValueError("No eval value received during optimization.")
        
    #     else:   
    #         value = self._get_eval_value()

    #         if value is None:
    #             raise ValueError("No eval value received during optimization.")
    #         return value


    # def create_subscriptions(self, alignment_topics: list[AlignTopic]):
    #     self.eval_dict.clear()
    #     for topic in alignment_topics:
    #         topic: AlignTopic
    #         topic_name = topic.topic_name
    #         self.eval_dict[topic_name] = None

    #         _sub = self.node.create_subscription(
    #                 Float32,
    #                 topic_name,
    #                 lambda msg, t=topic_name: self.topic_callback(msg, t),
    #                 10
    #             )
            
    #         _sub_dict = {"topic_name": topic_name, "subscription": _sub}
            
    #         self.subscription_list.append(_sub_dict)

    #         self.node.get_logger().info(f"Subscribed to topic: {topic.topic_name}")

    # def destroy_subscriptions(self):
    #     for sub in self.subscription_list:
    #         self.node.destroy_subscription(sub["subscription"])
    #         self.node.get_logger().info(f"Destroyed subscription to topic: {sub['topic_name']}")
    #     self.subscription_list = []
    #     self.eval_dict.clear()

    # def topic_callback(self, 
    #                    msg: Float32, 
    #                    topic: str):
    #     self.eval_dict[topic] = msg.data