"""Controller discovery and verified physical robot movement."""

import math
import threading
import time

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from ..optimization.base import OptimizationCancelled
from ..py_modules.AlignmentControllerManager import AlignmentControllerManager, ControllerJoint
from .robot_description import parse_robot_description
from .ros_helpers import wait_future, wait_service


class RobotMotion:
    """Retain inactive controller joints and confirm fresh measured positions."""

    def __init__(self, node):
        self.node = node
        self.condition = threading.Condition()
        self.positions = {}
        self.timestamps = {}
        self.clients = {}
        self.pending = set()
        self.controller_client = node.create_client(
            ListControllers, '/controller_manager/list_controllers',
            callback_group=node.cb_reentrant)
        self.description_client = node.create_client(
            GetParameters, '/robot_state_publisher/get_parameters',
            callback_group=node.cb_reentrant)
        self.subscription = node.create_subscription(
            JointState, '/joint_states', self._receive, 10,
            callback_group=node.cb_subscriptions)

    def _receive(self, msg):
        with self.condition:
            for name, value in zip(msg.name, msg.position):
                if math.isfinite(value):
                    self.positions[name] = value
                    self.timestamps[name] = time.monotonic()
            self.condition.notify_all()

    def initialize(self, active_joints, parameters, cancelled):
        if not active_joints or len({j.joint_name for j in active_joints}) != len(active_joints):
            raise ValueError('Active joints must be nonempty and unique.')
        timeout = parameters['trajectory_timeout']
        for client in (self.controller_client, self.description_client):
            wait_service(client, timeout, cancelled)
        controllers = wait_future(
            self.controller_client.call_async(ListControllers.Request()), timeout, cancelled)
        description = wait_future(self.description_client.call_async(
            GetParameters.Request(names=['robot_description'])), timeout, cancelled)
        if not description.values or not description.values[0].string_value:
            raise ValueError('robot_description is unavailable.')
        robot = parse_robot_description(description.values[0].string_value)
        limits = {joint.name: joint.limit for joint in robot.joints if joint.limit is not None}
        mapping = {}
        controller_joints = {}
        for controller in controllers.controller:
            if (controller.type != 'joint_trajectory_controller/JointTrajectoryController'
                    or controller.state != 'active'):
                continue
            joints = sorted({interface.rsplit('/', 1)[0]
                             for interface in controller.claimed_interfaces
                             if interface.endswith('/position')})
            controller_joints[controller.name] = joints
            mapping.update({name: controller.name for name in joints})
        involved = set()
        for joint in active_joints:
            if joint.joint_name not in mapping:
                raise ValueError(f'No active trajectory controller for {joint.joint_name}.')
            involved.add(mapping[joint.joint_name])
        required = {name for controller in involved for name in controller_joints[controller]}
        deadline = time.monotonic() + timeout
        with self.condition:
            while not all(name in self.positions and time.monotonic() - self.timestamps[name] < 2.0
                          for name in required):
                if cancelled():
                    raise OptimizationCancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        'Fresh joint states are unavailable for all controller joints.')
                self.condition.wait(0.05)
            initial_positions = dict(self.positions)
        manager = AlignmentControllerManager()
        requested = {joint.joint_name: joint for joint in active_joints}
        from active_alignment_interfaces.msg import JointAlign
        for controller in sorted(involved):
            for name in controller_joints[controller]:
                active = name in requested
                specification = requested[name] if active else JointAlign(joint_name=name)
                joint = ControllerJoint(active, initial_positions[name], specification)
                if active:
                    limit = limits.get(name)
                    if limit is None or not all(
                            value is not None and math.isfinite(value)
                            for value in (limit.lower, limit.upper)):
                        raise ValueError(f'Finite URDF limits are required for {name}.')
                    if not all(math.isfinite(value) for value in
                               (specification.upper_limit, specification.lower_limit)):
                        raise ValueError(f'Non-finite optimization limits for {name}.')
                    joint.initialize_joint_limits(limit.upper, limit.lower)
                    debug = joint.get_debug_info()
                    if (debug['upper_joint_optimization_limit']
                            <= debug['lower_joint_optimization_limit']):
                        raise ValueError(f'Empty optimization interval for {name}.')
                manager.add_joint_to_controller(controller, joint)
        return manager

    def move_manager(self, manager, parameters, cancelled):
        for controller in manager.get_controllers():
            self.move(
                controller.controller_name,
                controller.get_joint_states(),
                parameters,
                cancelled)

    def move(self, controller_name, positions, parameters, cancelled=lambda: False):
        if cancelled():
            raise OptimizationCancelled()
        timeout = parameters['trajectory_timeout']
        if controller_name not in self.clients:
            self.clients[controller_name] = ActionClient(
                self.node, FollowJointTrajectory, f'/{controller_name}/follow_joint_trajectory',
                callback_group=self.node.cb_reentrant)
        client = self.clients[controller_name]
        deadline = time.monotonic() + timeout
        while not client.wait_for_server(timeout_sec=0.05):
            if cancelled():
                raise OptimizationCancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Trajectory action unavailable: {controller_name}')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(positions)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions.values()]
        nanoseconds = round(parameters['movement_time'] * 1e9)
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(nanoseconds, 10**9)
        goal.trajectory.points = [point]
        sent_at = time.monotonic()
        token = object()
        with self.condition:
            self.pending.add(token)

        def finished(_):
            with self.condition:
                self.pending.discard(token)

        try:
            response_future = client.send_goal_async(goal)
        except Exception:
            finished(None)
            raise
        try:
            # Obtain accepted handles even after cancellation, then cancel their motion.
            handle = wait_future(response_future, timeout)
        except TimeoutError:
            def cancel_late_response(future):
                response = future.result()
                if response is not None and response.accepted:
                    response.get_result_async().add_done_callback(finished)
                    response.cancel_goal_async()
                else:
                    finished(future)
            response_future.add_done_callback(cancel_late_response)
            raise
        if handle is None or not handle.accepted:
            finished(response_future)
            raise RuntimeError(f'Trajectory rejected by {controller_name}.')
        result_future = handle.get_result_async()
        result_future.add_done_callback(finished)
        try:
            result = wait_future(result_future, timeout, cancelled)
            if (result.status != GoalStatus.STATUS_SUCCEEDED
                    or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL):
                raise RuntimeError(
                    f'Trajectory failed for {controller_name}: {result.result.error_string}')
            self.verify_positions(
                positions,
                sent_at,
                parameters['joint_tolerance'],
                timeout,
                cancelled)
        except Exception:
            try:
                wait_future(handle.cancel_goal_async(), min(timeout, 5.0))
                wait_future(result_future, min(timeout, 5.0))
            except Exception as exc:
                self.node.get_logger().error(f'Could not cancel trajectory: {exc}')
            raise

    def has_active_motion(self):
        with self.condition:
            return bool(self.pending)

    def verify_positions(self, positions, after, tolerance, timeout, cancelled=lambda: False):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not all(self.timestamps.get(name, 0) > after
                          and abs(self.positions[name] - value) <= tolerance
                          for name, value in positions.items()):
                if cancelled():
                    raise OptimizationCancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError('Robot did not reach the requested joint positions.')
                self.condition.wait(0.05)

    def snapshot(self):
        with self.condition:
            return dict(self.positions)
