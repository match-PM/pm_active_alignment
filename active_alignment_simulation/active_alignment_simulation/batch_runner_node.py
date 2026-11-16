"""Cancellable simulation batches implemented as an independent ROS action node."""

from copy import deepcopy
import json
from pathlib import Path
import threading
import time

from action_msgs.msg import GoalStatus
from active_alignment_interfaces.action import ActiveAlign, RunAlignmentBatch
from active_alignment_interfaces.msg import AlignTopic, JointAlign
from rcl_interfaces.srv import GetParameters, SetParametersAtomically
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter, parameter_value_to_python
from rosidl_runtime_py.convert import message_to_ordereddict
import yaml

from active_alignment_skills.helpers.motion import RobotMotion
from active_alignment_skills.optimization.base import OptimizationCancelled
from active_alignment_skills.helpers.parameters import (
    load_defaults, optimizer_parameters)
from active_alignment_skills.helpers.ros_helpers import (
    interruptible_sleep, wait_future, wait_service)
from active_alignment_skills.helpers.run_recorder import unique_run_directory
from .helpers.config import load_simulation, make_model, task_parameters
from .helpers.scene_frames import SceneFrames, pose_values
from .helpers.seed_generator import generate_seeds
from .helpers.observations import SignalVerifier


class ChildStillRunning(RuntimeError):
    """The child did not acknowledge cancellation; all setup mutation must stop."""


class BatchRunner(Node):
    """Coordinate setup and public optimization actions; keep per-run YAML immutable."""

    def __init__(self, **kwargs):
        super().__init__('active_alignment_batch', **kwargs)
        self.cb_reentrant = ReentrantCallbackGroup()
        self.cb_subscriptions = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.declare_parameter('simulation_parameter_file', '')
        self.motion = RobotMotion(self)
        self.frames = SceneFrames(self)
        self.action_clients = {}
        self.parameter_clients = {}
        self.child_active = False
        self.action = ActionServer(
            self, RunAlignmentBatch, 'active_alignment_batch/execute',
            execute_callback=self.execute, goal_callback=self.accept,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self.cb_reentrant)

    def accept(self, goal):
        if self.child_active or self.motion.has_active_motion():
            return GoalResponse.REJECT
        if not (goal.simulation_parameter_file or self.get_parameter(
                'simulation_parameter_file').value):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT if self.lock.acquire(False) else GoalResponse.REJECT

    def get_parameters_from(self, node_name, names, timeout):
        endpoint = node_name.rstrip('/') + '/get_parameters'
        if endpoint not in self.parameter_clients:
            self.parameter_clients[endpoint] = self.create_client(
                GetParameters, endpoint, callback_group=self.cb_reentrant)
        client = self.parameter_clients[endpoint]
        wait_service(client, timeout)
        response = wait_future(client.call_async(GetParameters.Request(names=names)), timeout)
        return {name: parameter_value_to_python(value)
                for name, value in zip(names, response.values)}

    def set_parameters_on(self, node_name, values, timeout):
        endpoint = node_name.rstrip('/') + '/set_parameters_atomically'
        if endpoint not in self.parameter_clients:
            self.parameter_clients[endpoint] = self.create_client(
                SetParametersAtomically, endpoint, callback_group=self.cb_reentrant)
        client = self.parameter_clients[endpoint]
        wait_service(client, timeout)
        request = SetParametersAtomically.Request(parameters=[
            Parameter(name, value=value).to_parameter_msg() for name, value in values.items()])
        response = wait_future(client.call_async(request), timeout)
        if not response.result.successful:
            raise ValueError(response.result.reason)

    def run_goal(self, config, goal, algorithm, cancelled):
        endpoint = config['alignment_server'].rstrip('/') + '/exec_active_alignment_' + algorithm
        if endpoint not in self.action_clients:
            self.action_clients[endpoint] = ActionClient(
                self, ActiveAlign, endpoint, callback_group=self.cb_reentrant)
        client = self.action_clients[endpoint]
        deadline = time.monotonic() + config['service_timeout']
        while not client.wait_for_server(timeout_sec=0.05):
            if cancelled():
                raise OptimizationCancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Optimization endpoint unavailable: {endpoint}')
        response = client.send_goal_async(goal)
        self.child_active = True
        try:
            handle = wait_future(response, config['service_timeout'])
        except TimeoutError:
            response.add_done_callback(lambda future: future.result().cancel_goal_async()
                                       if future.result() and future.result().accepted else None)
            raise ChildStillRunning('Optimization goal response timed out; '
                                    'retaining the reservation until its state is checked.')
        if handle is None or not handle.accepted:
            self.child_active = False
            raise RuntimeError('Alignment server rejected the goal (another run may be active).')
        self.child_active = True
        result_future = handle.get_result_async()
        try:
            wrapped = wait_future(result_future, config['run_timeout'], cancelled)
            self.confirm_server_idle(config)
            self.child_active = False
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED or not wrapped.result.success:
                raise RuntimeError(f'Optimization failed with action status {wrapped.status}.')
            return wrapped.result
        except Exception:
            # Do not reset joints or launch another task until the optimizer
            # acknowledges cancellation.
            try:
                wait_future(handle.cancel_goal_async(), config['service_timeout'])
                wait_future(result_future, config['service_timeout'])
                self.confirm_server_idle(config)
                self.child_active = False
            except Exception as exc:
                raise ChildStillRunning(
                    'Child optimization did not stop; retaining batch reservation. '
                    'Stop the child before restoring frames or running another batch.') from exc
            raise

    def confirm_server_idle(self, config):
        deadline = time.monotonic() + config['service_timeout']
        while True:
            try:
                self.set_parameters_on(config['alignment_server'],
                                       {'batch_directory': self.reservation},
                                       config['service_timeout'])
                return
            except ValueError:
                if time.monotonic() >= deadline:
                    raise ChildStillRunning('A child trajectory has not stopped.')
                time.sleep(0.02)

    def execute(self, handle):
        def cancelled():
            return handle.is_cancel_requested or not rclpy.ok()
        result = RunAlignmentBatch.Result()
        config = None
        directory = None
        reset_positions = None
        reset_manager = None
        reset_parameters = None
        publisher_original = None
        baseline_parameters = None
        reserved = False
        verifier = None
        summary = {'runs': [], 'status': 'running', 'message': ''}
        interrupted = False
        try:
            path = handle.request.simulation_parameter_file or self.get_parameter(
                'simulation_parameter_file').value
            config = load_simulation(path)
            base = Path(config.get('results_directory', 'simulation_results')).expanduser()
            if not base.is_absolute():
                base = Path(config['_path']).parent / base
            directory = unique_run_directory(base, 'batch')
            self.reservation = str(directory)
            result.results_path = str(directory)
            with (directory / 'simulation_parameter.yaml').open('w') as stream:
                yaml.safe_dump({key: value for key, value in config.items()
                               if key != '_path'}, stream, sort_keys=False)
            model = make_model(config)
            seeds = generate_seeds(config, model)
            with (directory / 'seed_poses.yaml').open('w') as stream:
                yaml.safe_dump(seeds, stream, sort_keys=False)
            baseline_parameters = self.get_parameters_from(
                config['alignment_server'], list(load_defaults()), config['service_timeout'])
            # Resolve every task before touching frames or the robot.
            tasks = [(task, task_parameters(config, task, baseline_parameters))
                     for task in config['tasks'] if task['iterations']]
            if not tasks:
                raise ValueError('All task repetition counts are zero.')
            self.set_parameters_on(config['alignment_server'], {'batch_directory': str(directory)},
                                   config['service_timeout'])
            reserved = True
            total = len(seeds) * sum(task['iterations'] for task, _ in tasks)
            publisher_values = model.to_parameters()
            publisher_values.update(config.get('model', {}))
            # Target-power experiments require the optical policy. The publisher's
            # previous policy and settings are restored when the batch finishes.
            publisher_values['policy'] = 'optical'
            # Structural publisher settings are startup-only, not per-run overrides.
            for key in ('publish_rate', 'legacy_distance_scale'):
                publisher_values.pop(key, None)
            publisher_values.setdefault('noise_mode', 'none')
            publisher_values['noise_seed'] = config['noise_seed_base']
            publisher_original = self.get_parameters_from(
                config['publisher_node'], list(publisher_values), config['service_timeout'])
            self.set_parameters_on(
                config['publisher_node'],
                publisher_values,
                config['service_timeout'])
            verifier = SignalVerifier(self, config['signal_verification']['topic'])
            self.frames.prepare(config, cancelled)
            with (directory / 'initial_frames.json').open('w') as stream:
                json.dump({name: message_to_ordereddict(frame) if frame is not None else None
                           for name, frame in self.frames.originals.items()}, stream, indent=2)
            specifications = [JointAlign(**joint) for joint in config['active_joints']]
            reset_parameters = optimizer_parameters(tasks[0][1], tasks[0][0]['algorithm'])
            reset_parameters['movement_time'] = float(config.get('reset_movement_time', 1.0))
            if reset_parameters['movement_time'] <= 0:
                raise ValueError('reset_movement_time must be positive.')
            reset_manager = self.motion.initialize(specifications, reset_parameters, cancelled)
            reset_positions = reset_manager.get_all_joint_states(only_active=True)
            explicit = config.get('initial_joint_positions')
            if explicit:
                reset_positions.update(explicit)
                # Physical reset positions may lie outside the original optimization window;
                # ControllerJoint.set_current_value still enforces the URDF limits.
                reset_manager.set_joint_states({name: value for name, value in explicit.items()
                                                if name in reset_positions})
            else:
                reset_manager.set_joint_states_from_mapped(
                    {name: config['reset_position'] for name in reset_positions})
                reset_positions = reset_manager.get_all_joint_states(only_active=True)
            self.motion.move_manager(reset_manager, reset_parameters, cancelled)
            interruptible_sleep(config['setup_settle_time'], cancelled)
            reference = config['target'].get('reference_frame', config['target']['frame_name'])
            baseline_pose = self.frames.world_pose(config['world_frame'], reference,
                                                   config['service_timeout'], cancelled)
            summary['baseline_world_pose'] = dict(
                zip(('position', 'quaternion'), pose_values(baseline_pose)))
            summary['initial_joint_positions'] = reset_positions
            run_number = 0
            for seed in seeds:
                for task, parameters in tasks:
                    algorithm = task['algorithm']
                    for repetition in range(task['iterations']):
                        if cancelled():
                            raise OptimizationCancelled()
                        run_number += 1
                        feedback = RunAlignmentBatch.Feedback(
                            total_runs=total, completed_runs=result.completed_runs,
                            failed_runs=result.failed_runs, algorithm=algorithm,
                            repetition=repetition + 1, seed=seed['seed'])
                        handle.publish_feedback(feedback)
                        run_directory = directory / (
                            f'run_{run_number:05d}_{algorithm}'
                            f'_seed_{seed["seed"]}_rep_{repetition+1}')
                        run_directory.mkdir()
                        run_parameters = deepcopy(parameters)
                        if not run_parameters['general.metrics_topic_prefix']:
                            run_parameters['general.metrics_topic_prefix'] = (
                                config['signal_verification']['topic'].rsplit('/', 1)[0])
                        # Same repetition has the same RNG seed across algorithms.
                        if 'random_seed' not in task.get('general', {}):
                            run_parameters['general.random_seed'] = (
                                config['optimizer_seed_base'] + repetition)
                        parameter_path = run_directory / 'optimization_parameter.yaml'
                        with parameter_path.open('w') as stream:
                            yaml.safe_dump(run_parameters, stream, sort_keys=False)
                        record = {'algorithm': algorithm, 'repetition': repetition + 1,
                                  'seed': seed, 'directory': str(run_directory),
                                  'status': 'running',
                                  'optimizer_seed': run_parameters['general.random_seed'],
                                  'noise_seed': config['noise_seed_base'] + repetition}
                        try:
                            reset_manager.set_joint_states(reset_positions)
                            self.motion.move_manager(reset_manager, reset_parameters, cancelled)
                            self.frames.apply_seed(config, baseline_pose, seed, cancelled)
                            self.set_parameters_on(config['publisher_node'], {
                                'noise_seed': config['noise_seed_base'] + repetition},
                                config['service_timeout'])
                            interruptible_sleep(config['setup_settle_time'], cancelled)
                            record['verified_initial_signal'] = verifier.verify(
                                seed['expected_signal'], config['signal_verification'].get(
                                    'tolerance', 0.001),
                                config['signal_verification'].get('timeout', 5.0), cancelled)
                            goal = ActiveAlign.Goal(
                                active_joints=specifications,
                                alignment_topics=[AlignTopic(**topic)
                                                  for topic in config['alignment_topics']],
                                parameter_file=str(parameter_path), rsap_path=str(run_directory))
                            outcome = self.run_goal(config, goal, algorithm, cancelled)
                            record.update(
                                status='succeeded',
                                result=message_to_ordereddict(outcome))
                            result.completed_runs += 1
                        except (OptimizationCancelled, ChildStillRunning):
                            record['status'] = 'interrupted'
                            raise
                        except Exception as exc:
                            record.update(status='failed', message=str(exc))
                            result.failed_runs += 1
                            if self.motion.has_active_motion():
                                raise ChildStillRunning(
                                    'A setup trajectory has not stopped.') from exc
                            if not config['continue_on_failure']:
                                raise
                        finally:
                            summary['runs'].append(record)
                            with (run_directory / 'result.json').open('w') as stream:
                                json.dump(record, stream, indent=2)
                            self._save_summary(directory, summary)
                        reset_manager.set_joint_states(reset_positions)
                        self.motion.move_manager(reset_manager, reset_parameters, cancelled)
            result.success = result.failed_runs == 0
            summary['status'] = 'succeeded' if result.success else 'failed'
        except OptimizationCancelled:
            interrupted = True
            summary.update(status='cancelled', message='Batch cancelled.')
        except Exception as exc:
            summary.update(status='failed', message=str(exc))
            self.get_logger().error(f'Batch failed: {exc}')
        finally:
            cleanup_errors = []
            if verifier is not None:
                verifier.close()
            motion_active = self.child_active or self.motion.has_active_motion()
            if config is not None and reserved and not motion_active:
                # Cancellation stops motion; do not issue new robot moves after a cancel.
                if (not interrupted and not handle.is_cancel_requested
                        and reset_manager is not None and reset_positions is not None):
                    try:
                        reset_manager.set_joint_states(reset_positions)
                        self.motion.move_manager(reset_manager, reset_parameters, lambda: False)
                    except Exception as exc:
                        cleanup_errors.append(str(exc))
                try:
                    self.frames.restore(config['service_timeout'])
                except Exception as exc:
                    cleanup_errors.append(str(exc))
                if publisher_original is not None:
                    try:
                        self.set_parameters_on(
                            config['publisher_node'],
                            publisher_original,
                            config['service_timeout'])
                    except Exception as exc:
                        cleanup_errors.append(str(exc))
                if baseline_parameters is not None:
                    try:
                        self.set_parameters_on(
                            config['alignment_server'],
                            baseline_parameters,
                            config['service_timeout'])
                    except Exception as exc:
                        cleanup_errors.append(str(exc))
                try:
                    self.set_parameters_on(config['alignment_server'], {'batch_directory': ''},
                                           config['service_timeout'])
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            elif motion_active:
                cleanup_errors.append(
                    'Child still active; frame, robot, publisher, and server state retained.')
            if cleanup_errors:
                result.success = False
                summary.update(status='failed', cleanup_errors=cleanup_errors)
            if summary['status'] != 'succeeded':
                result.success = False
            result.message = summary.get('message', '') or summary['status']
            if directory is not None:
                try:
                    self._save_summary(directory, summary)
                except OSError as exc:
                    result.success = False
                    result.message = f'Cannot save batch summary: {exc}'
            try:
                if handle.is_cancel_requested:
                    result.success = False
                    handle.canceled()
                elif result.success:
                    handle.succeed()
                else:
                    handle.abort()
            finally:
                self.lock.release()
        return result

    @staticmethod
    def _save_summary(directory, summary):
        temporary = directory / 'batch_summary.json.tmp'
        with temporary.open('w') as stream:
            json.dump(summary, stream, indent=2)
        temporary.replace(directory / 'batch_summary.json')


def main(args=None):
    rclpy.init(args=args)
    node = BatchRunner()
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
