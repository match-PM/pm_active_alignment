"""Publish test measurements calculated from the poses of configured TF frame pairs."""

import threading

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .simulation.policies import make_policy
from .simulation.policies.optical import OpticalPolicy
from .simulation.publisher_configuration import (
    create_frame_pairs, load_publisher_overrides, publisher_defaults)


class MultiFrameDeviationNode(Node):
    """Read TF, call the selected policy, and publish its calculated measurements."""

    def __init__(self, **kwargs):
        super().__init__('multi_frame_deviation_node', **kwargs)
        self.policy_lock = threading.RLock()

        # Declare ordinary ROS parameters first. The optional simulation YAML can
        # then override model settings and frame pairs, just as on the working branch.
        defaults = publisher_defaults()
        self.settings = {}
        for name, default in defaults.items():
            self.declare_parameter(name, default)
            self.settings[name] = self.get_parameter(name).value

        config_path = self.settings['simulation_parameter_file']
        if config_path:
            overrides = load_publisher_overrides(config_path)
            parameters = []
            for name, value in overrides.items():
                parameters.append(Parameter(name, value=value))
            response = self.set_parameters_atomically(parameters)
            if not response.successful:
                raise ValueError(response.reason)
            self.settings.update(overrides)

        self.frame_pairs = create_frame_pairs(self.settings)
        self.policy = make_policy(self.settings)

        # The listener maintains the latest transform for every configured pair.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.clear_buffer_service = self.create_service(
            Trigger, 'clear_tf_buffer', self.clear_buffer)

        # Advertise the complete topic set once. Switching policies changes which
        # values are published, without creating or destroying ROS publishers.
        self.publishers_by_pair = {}
        for pair in self.frame_pairs:
            self.publishers_by_pair[pair.name] = {}
            for topic in OpticalPolicy.TOPICS:
                topic_name = f'{pair.name}/{topic}'
                publisher = self.create_publisher(Float32, topic_name, 10)
                self.publishers_by_pair[pair.name][topic] = publisher

        period = 1.0 / self.settings['publish_rate']
        self.timer = self.create_timer(period, self.compute_all_frame_deviations)
        self.add_on_set_parameters_callback(self.update_parameters)
        self.get_logger().info(f"Dummy publisher using {self.settings['policy']} policy.")

    def update_parameters(self, parameters):
        """Validate a complete policy update before replacing the active policy."""
        startup_only = (
            'pair_names', 'frames_a', 'frames_b', 'publish_rate', 'simulation_parameter_file')

        try:
            with self.policy_lock:
                updated = dict(self.settings)
                for parameter in parameters:
                    if parameter.name == 'use_sim_time':
                        continue
                    if parameter.name not in updated:
                        raise ValueError(f'Unknown publisher parameter: {parameter.name}')
                    if parameter.name in startup_only:
                        if parameter.value != self.settings[parameter.name]:
                            raise ValueError(f'{parameter.name} requires a publisher restart.')
                    updated[parameter.name] = parameter.value

                # Building a policy also validates its physical constants and
                # noise settings. A failed update leaves the current policy intact.
                policy = make_policy(updated)
                self.settings = updated
                self.policy = policy
            return SetParametersResult(successful=True)
        except (ValueError, TypeError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))

    def clear_buffer(self, request, response):
        self.tf_buffer.clear()
        response.success = True
        response.message = 'TF buffer cleared.'
        return response

    def compute_all_frame_deviations(self):
        """Measure every frame pair using the selected geometric or optical policy."""
        for pair in self.frame_pairs:
            try:
                transform = self.tf_buffer.lookup_transform(
                    pair.frame_a, pair.frame_b, rclpy.time.Time())
            except TransformException as exc:
                self.get_logger().warning(
                    f'TF unavailable for {pair.name}: {exc}', throttle_duration_sec=5.0)
                continue

            translation = transform.transform.translation
            rotation = transform.transform.rotation
            position = [translation.x, translation.y, translation.z]
            quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]

            # Policy equations live in simulation/policies/, not in this ROS node.
            # optical.py calculates coupling and noise; geometric.py keeps the
            # original scaled-distance and full-rotation measurements.
            with self.policy_lock:
                measurements = self.policy.evaluate(position, quaternion)

            for topic, value in measurements.items():
                message = Float32(data=float(value))
                self.publishers_by_pair[pair.name][topic].publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = MultiFrameDeviationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
