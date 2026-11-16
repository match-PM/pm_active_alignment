"""ROS action adapter for independently implemented alignment optimizers."""

from pathlib import Path
from functools import partial
import threading
import time

from active_alignment_interfaces.action import ActiveAlign
from active_alignment_interfaces.msg import JointAlignmentResult
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosidl_runtime_py.convert import message_to_ordereddict
import yaml

from .helpers.motion import RobotMotion
from .helpers.metrics_reader import MetricsReader
from .optimization.base import OptimizationCancelled, OptimizationSession, run_optimizer
from .optimization.registry import OPTIMIZERS
from .helpers.parameters import (
    load_defaults, optimizer_parameters, read_parameter_file,
    resolve_parameter_file, rsap_directory, validate_parameters)
from .py_modules.OptimizationValueManager import OptimizationValueManager
from .helpers.ros_helpers import interruptible_sleep
from .helpers.run_recorder import RunRecorder, unique_run_directory
from .helpers.subscription_worker import SubscriptionWorker


class ActiveAlignmentServer(Node):
    """Serialize hardware runs across all endpoints and snapshot their parameters."""

    def __init__(self, **kwargs):
        super().__init__('active_alignment_server', **kwargs)
        # Optimization runs block their own worker while other executor threads
        # continue receiving controller responses, joint states, and cancellation.
        self.cb_reentrant = ReentrantCallbackGroup()
        self.cb_subscriptions = ReentrantCallbackGroup()
        self.run_lock = threading.Lock()
        self.parameter_lock = threading.RLock()
        self.schema = load_defaults()
        self.declare_parameter('initial_parameter_file', '')
        self.declare_parameter('results_directory', str(Path.home() / '.ros' / 'active_alignment'))
        self.declare_parameter('batch_directory', '')
        for name, default in self.schema.items():
            self.declare_parameter(name, default)
        validate_parameters(self.parameter_snapshot(), self.schema)
        self.add_on_set_parameters_callback(self._validate_parameter_update)
        initial_file = self.get_parameter('initial_parameter_file').value
        if initial_file:
            self.load_parameter_file(Path(initial_file).expanduser().resolve())
        self.motion = RobotMotion(self)
        self.optimization_value_manager = OptimizationValueManager(self)
        # Keep an explicit list so every action server stays alive with the node.
        self.action_servers = []
        for algorithm in OPTIMIZERS:
            server = ActionServer(
                self, ActiveAlign,
                f'{self.get_name()}/exec_active_alignment_{algorithm}',
                execute_callback=partial(self.execute_callback, algorithm=algorithm),
                goal_callback=self.goal_callback,
                cancel_callback=partial(self.cancel_callback, algorithm=algorithm),
                callback_group=self.cb_reentrant)
            self.action_servers.append(server)
        self.get_logger().info(f'Alignment server ready: {", ".join(OPTIMIZERS)}')

    def parameter_snapshot(self):
        with self.parameter_lock:
            return {name: self.get_parameter(name).value for name in self.schema}

    def set_parameters_atomically(self, parameters):
        lock = getattr(self, 'parameter_lock', threading.RLock())
        with lock:
            return super().set_parameters_atomically(parameters)

    def set_parameters(self, parameters):
        lock = getattr(self, 'parameter_lock', threading.RLock())
        with lock:
            return super().set_parameters(parameters)

    def _validate_parameter_update(self, parameters):
        try:
            with self.parameter_lock:
                for parameter in parameters:
                    if parameter.name == 'batch_directory':
                        current = self.get_parameter('batch_directory').value
                        if self.run_lock.locked():
                            raise ValueError(
                                'Cannot change batch reservation during optimization.')
                        if hasattr(self, 'motion') and self.motion.has_active_motion():
                            raise ValueError('A trajectory has not acknowledged termination.')
                        if current and parameter.value and current != parameter.value:
                            raise ValueError('Another batch already owns the alignment server.')
                values = self.parameter_snapshot()
                for parameter in parameters:
                    if parameter.name in self.schema:
                        values[parameter.name] = parameter.value
                validate_parameters(values, self.schema)
            return SetParametersResult(successful=True)
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))

    def load_parameter_file(self, path):
        """Validate the whole update before applying any changes to ROS parameters."""
        with self.parameter_lock:
            updates = read_parameter_file(path, self.get_fully_qualified_name())
            checked = validate_parameters({**self.parameter_snapshot(), **updates}, self.schema)
            result = self.set_parameters_atomically([
                Parameter(name, value=checked[name]) for name in updates])
            if not result.successful:
                raise ValueError(result.reason)
            return self.parameter_snapshot()

    def goal_callback(self, request):
        """Reserve the robot across all algorithms, or reject a competing goal."""
        if not request.active_joints or not request.alignment_topics:
            return GoalResponse.REJECT
        with self.parameter_lock:
            if self.motion.has_active_motion():
                return GoalResponse.REJECT
            reserved = self.get_parameter('batch_directory').value
            if reserved:
                if not request.rsap_path or not rsap_directory(
                        request.rsap_path).is_relative_to(Path(reserved)):
                    return GoalResponse.REJECT
            if not self.run_lock.acquire(blocking=False):
                self.get_logger().warning(
                    'An alignment run is already active; rejecting competing goal.')
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle, algorithm):
        # The execution callback cancels motion and sets the terminal action state.
        self.get_logger().info(f'{algorithm}: cancellation requested; stopping execution.')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle, algorithm):
        """Load settings, run one search, restore its best point, and save the result."""
        goal = goal_handle.request
        started = time.monotonic()
        last_progress = -float('inf')

        def cancelled():
            return goal_handle.is_cancel_requested or not rclpy.ok()

        def topic_values(record):
            return ', '.join(
                f'{topic.topic_name}={record.metrics[f"topic:{topic.topic_name}"]:.6g}'
                for topic in goal.alignment_topics)

        result = ActiveAlign.Result()
        recorder = None
        session = None
        metrics_reader = None
        subscriptions = None
        directory = None
        physical_result = {}
        status, message = 'failed', ''
        try:
            self.get_logger().info(f'{algorithm}: starting execution.')
            # Resolve logs independently of parameter loading, so invalid files
            # still leave a failed-run record in the requested experiment folder.
            parent = (rsap_directory(goal.rsap_path) if goal.rsap_path else
                      Path(self.get_parameter('results_directory').value).expanduser())
            base = parent / 'optimization_logs'
            directory = unique_run_directory(base, algorithm)
            recorder = RunRecorder(directory, algorithm, self.parameter_snapshot(),
                                   message_to_ordereddict(goal))
            parameter_file = resolve_parameter_file(goal.parameter_file, goal.rsap_path)
            with self.parameter_lock:
                snapshot = (self.load_parameter_file(parameter_file) if parameter_file is not None
                            else self.parameter_snapshot())
            with (directory / 'optimization_parameter.yaml').open('w') as stream:
                yaml.safe_dump(snapshot, stream, sort_keys=False)
            settings = optimizer_parameters(snapshot, algorithm)
            self.get_logger().info(
                f'{algorithm}: initializing robot; evaluation budget='
                f'{settings["max_evaluations"]}; results: {directory}')
            manager = self.motion.initialize(goal.active_joints, settings, cancelled)
            names = list(manager.get_all_joint_states_as_mapped())
            subscriptions = SubscriptionWorker(self, 'alignment_measurements')
            self.optimization_value_manager = OptimizationValueManager(subscriptions.node)
            self.optimization_value_manager.init(goal.alignment_topics)
            metrics_reader = MetricsReader(subscriptions.node, settings['metrics_topic_prefix'])
            subscriptions.start()

            def measure(candidate):
                # Search positions are normalized; the manager converts them to
                # physical positions and retains the inactive controller joints.
                manager.set_joint_states_from_mapped(dict(zip(names, candidate)))
                self.motion.move_manager(manager, settings, cancelled)
                # move_manager returns only after controller completion and joint
                # verification. Settling is optional; it is NOT a measurement delay.
                interruptible_sleep(settings['wait_for_settle'], cancelled)
                after = time.monotonic()
                # Even with zero settling, require each topic to receive a sample
                # after this boundary. A cached sample from a previous move cannot
                # satisfy this read. Timeout is a limit, not a fixed waiting period.
                cost, raw = self.optimization_value_manager.wait_for_fresh(
                    after, settings['measurement_timeout'], cancelled)
                metrics = {f'topic:{name}': value for name, value in raw.items()}
                diagnostics = metrics_reader.snapshot(
                    after, settings['measurement_timeout'], cancelled)
                metrics.update(diagnostics)
                metrics.update({f'commanded:{name}': value for name, value in
                                manager.get_all_joint_states(only_active=True).items()})
                measured = self.motion.snapshot()
                metrics.update({f'measured:{name}': measured[name] for name in names})
                return cost, metrics

            def observe(record):
                nonlocal last_progress
                recorder.record(record)
                goal_handle.publish_feedback(ActiveAlign.Feedback(active=True))
                if (session.evaluations == 0 or session.evaluations % 10 == 0
                        or record.elapsed - last_progress >= 2.0):
                    self.get_logger().info(
                        f'{algorithm}: progress {session.evaluations}/'
                        f'{settings["max_evaluations"]} evaluations; '
                        f'current [{topic_values(record)}]; '
                        f'best [{topic_values(session.best)}]; '
                        f'elapsed {record.elapsed:.1f}s')
                    last_progress = record.elapsed

            initial = list(manager.get_all_joint_states_as_mapped().values())
            session = OptimizationSession(initial, settings, measure, cancelled, observe)
            run_optimizer(OPTIMIZERS[algorithm](), session)
            session.check_cancelled()
            # The last candidate need not be the best one. Move back explicitly
            # before returning physical joint positions to the action client.
            self.get_logger().info(
                f'{algorithm}: search stopped ({session.stop_reason}); '
                'restoring best position.')
            manager.set_joint_states_from_mapped(dict(zip(names, session.best.position)))
            self.motion.move_manager(manager, settings, cancelled)
            physical_result = manager.get_all_joint_states(only_active=True)
            result.joint_results = [JointAlignmentResult(joint_name=name, joint_value=float(value))
                                    for name, value in physical_result.items()]
            result.num_of_iterations = session.evaluations
            # Report the observed values at the selected point in their original
            # sign. Direction changes belong only to the optimization objective.
            total_weight = sum(topic.weight_factor for topic in goal.alignment_topics)
            result.final_eval_value = sum(
                topic.weight_factor * session.best.metrics[f'topic:{topic.topic_name}']
                for topic in goal.alignment_topics) / total_weight
            result.success = True
            status = 'succeeded'
        except OptimizationCancelled as exc:
            status, message = 'cancelled', str(exc)
            if session is not None:
                session.stop_reason = 'cancelled'
        except Exception as exc:
            message = str(exc)
            if session is not None:
                session.stop_reason = 'error'
        finally:
            if goal_handle.is_cancel_requested:
                status = 'cancelled'
                if session is not None:
                    session.stop_reason = 'cancelled'
            try:
                # Stop the measurement executor before removing its subscriptions;
                # otherwise a concurrent ROS callback could still hold one of them.
                if subscriptions is not None:
                    subscriptions.stop()
                self.optimization_value_manager.clear()
                if metrics_reader is not None:
                    metrics_reader.close()
                if subscriptions is not None:
                    subscriptions.node.destroy_node()
                if recorder is not None:
                    recorder.finish(status, session, message, physical_result)
            except Exception as exc:
                result.success = False
                message = f'{message}; ' if message else ''
                message += f'Could not finalize optimization results: {exc}'
            try:
                evaluations = session.evaluations if session is not None else 0
                summary = (f'evaluations={evaluations}; '
                           f'elapsed {time.monotonic() - started:.1f}s')
                if session is not None and session.best is not None:
                    summary += f'; best [{topic_values(session.best)}]'
                if directory is not None:
                    summary += f'; results: {directory}'
                if goal_handle.is_cancel_requested:
                    result.success = False
                    self.get_logger().info(
                        f'{algorithm}: execution cancelled; {summary}')
                    goal_handle.canceled()
                elif result.success:
                    self.get_logger().info(
                        f'{algorithm}: execution completed; '
                        f'stop reason={session.stop_reason}; {summary}; '
                        f'final value={result.final_eval_value:.6g}')
                    goal_handle.succeed()
                else:
                    log = (self.get_logger().info if status == 'cancelled'
                           else self.get_logger().error)
                    log(f'{algorithm}: execution stopped; reason={message}; {summary}')
                    goal_handle.abort()
            finally:
                self.run_lock.release()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ActiveAlignmentServer()
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
