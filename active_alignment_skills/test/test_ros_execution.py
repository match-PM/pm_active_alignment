"""Exercise real ROS actions against fake controllers, without touching hardware.

Run in an isolated ROS_DOMAIN_ID after building active_alignment_interfaces.
"""

from active_alignment_skills.helpers.ros_helpers import wait_future
from active_alignment_skills.active_alignment_skill_node import ActiveAlignmentServer
from active_alignment_skills.optimization.registry import OPTIMIZERS
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from rclpy.parameter import Parameter
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterValue
from controller_manager_msgs.srv import ListControllers
from controller_manager_msgs.msg import ControllerState
from control_msgs.action import FollowJointTrajectory
from active_alignment_interfaces.msg import AlignTopic, JointAlign
from active_alignment_interfaces.action import ActiveAlign
from action_msgs.msg import GoalStatus
import csv
import json
from pathlib import Path
import threading
import time

import pytest
import yaml

rclpy = pytest.importorskip('rclpy')


JOINTS = ['SP_X_Joint', 'SP_Y_Joint', 'inactive_joint']


class FakeRobot(Node):
    """Physical coordinates differ from normalized coordinates; one joint is inactive."""

    def __init__(self):
        super().__init__('fake_alignment_robot')
        self.cb = ReentrantCallbackGroup()
        self.positions = dict(zip(JOINTS, [0.0, 0.0, 0.4]))
        self.delay = 0.0
        self.fail_motion = False
        self.moves = []
        self.urdf = '<robot name="test">' + ''.join(
            f'<link name="link_{i}"/><joint name="{name}" type="prismatic">'
            f'<parent link="base"/><child link="link_{i}"/><axis xyz="1 0 0"/>'
            '<limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
            for i, name in enumerate(JOINTS)) + '<link name="base"/></robot>'
        self.create_service(GetParameters, '/robot_state_publisher/get_parameters',
                            self.description,
                            callback_group=self.cb)
        self.create_service(ListControllers, '/controller_manager/list_controllers',
                            self.controllers,
                            callback_group=self.cb)
        self.trajectory = ActionServer(
            self, FollowJointTrajectory, '/test_controller/follow_joint_trajectory',
            execute_callback=self.move, cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self.cb)
        self.state_publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.value_publisher = self.create_publisher(Float32, '/test/cost', 10)
        self.create_timer(0.01, self.publish, callback_group=self.cb)

    def description(self, request, response):
        response.values = [ParameterValue(type=4, string_value=self.urdf)]
        return response

    def controllers(self, request, response):
        response.controller = [ControllerState(
            name='test_controller', state='active',
            type='joint_trajectory_controller/JointTrajectoryController',
            claimed_interfaces=[name + '/position' for name in JOINTS])]
        return response

    def move(self, handle):
        if self.delay:
            time.sleep(self.delay)
        result = FollowJointTrajectory.Result()
        if handle.is_cancel_requested:
            handle.canceled()
            return result
        if self.fail_motion:
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            result.error_string = 'Injected controller failure'
            handle.abort()
            return result
        values = dict(zip(handle.request.trajectory.joint_names,
                          handle.request.trajectory.points[-1].positions))
        self.positions.update(values)
        self.moves.append(values)
        self.publish()
        handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def publish(self):
        self.state_publisher.publish(JointState(
            name=list(self.positions), position=list(self.positions.values())))
        cost = (self.positions[JOINTS[0]] - 0.3)**2 + (self.positions[JOINTS[1]] + 0.2)**2
        self.value_publisher.publish(Float32(data=float(cost)))


@pytest.fixture
def ros_system(tmp_path):
    rclpy.init()
    robot = FakeRobot()
    server = ActiveAlignmentServer(parameter_overrides=[
        Parameter('results_directory', value=str(tmp_path))])
    client = Node('test_alignment_client')
    executor = MultiThreadedExecutor(num_threads=10)
    for node in (robot, server, client):
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield robot, server, client, executor
    finally:
        executor.shutdown(timeout_sec=10)
        thread.join(timeout=10)
        for node in (client, server, robot):
            node.destroy_node()
        rclpy.shutdown()


def make_goal(tmp_path, algorithm, evaluations=50, objective_threshold=None):
    parameter_file = tmp_path / 'optimization_parameter.yaml'
    general = {
        'max_evaluations': evaluations, 'movement_time': 0.01,
        'wait_for_settle': 0.0, 'measurement_timeout': 2.0, 'trajectory_timeout': 2.0,
    }
    if objective_threshold is not None:
        general.update(stop_on_objective_threshold=True, objective_threshold=objective_threshold)
    parameter_file.write_text(yaml.safe_dump({'general': general}))
    return ActiveAlign.Goal(
        active_joints=[JointAlign(joint_name=name, has_constraint=True,
                                  lower_limit=-0.8, upper_limit=0.8) for name in JOINTS[:2]],
        alignment_topics=[
            AlignTopic(
                topic_name='/test/cost',
                minimize_not_maximize=True,
                weight_factor=1.0)],
        parameter_file=str(parameter_file), rsap_path=str(tmp_path))


def send(client, algorithm, goal):
    action = ActionClient(client, ActiveAlign,
                          '/active_alignment_server/exec_active_alignment_' + algorithm)
    assert action.wait_for_server(timeout_sec=5)
    handle = wait_future(action.send_goal_async(goal), 5)
    return action, handle


@pytest.mark.parametrize('algorithm', [
    'nelder_mead', 'nelder_mead_manual', 'adaptive_hill_climb', 'pybobyqa'])
def test_physical_results_best_position_and_inactive_joint(ros_system, tmp_path, algorithm):
    if algorithm == 'pybobyqa':
        pytest.importorskip('pybobyqa')
    robot, server, client, _ = ros_system
    action, handle = send(client, algorithm, make_goal(tmp_path, algorithm))
    assert handle.accepted
    outcome = wait_future(handle.get_result_async(), 30)
    assert outcome.status == GoalStatus.STATUS_SUCCEEDED
    assert outcome.result.success
    physical = {item.joint_name: item.joint_value for item in outcome.result.joint_results}
    assert physical[JOINTS[0]] == pytest.approx(robot.positions[JOINTS[0]], abs=1e-6)
    assert physical[JOINTS[1]] == pytest.approx(robot.positions[JOINTS[1]], abs=1e-6)
    assert abs(physical[JOINTS[0]] - 0.3) < 0.03
    assert abs(physical[JOINTS[1]] + 0.2) < 0.03
    assert robot.positions['inactive_joint'] == 0.4
    assert all(move['inactive_joint'] == 0.4 for move in robot.moves)
    assert server.get_parameter('general.max_evaluations').value == 50
    metadata_path = next(tmp_path.glob('optimization_logs/*/metadata.json'))
    metadata = json.loads(metadata_path.read_text())
    assert metadata['status'] == 'succeeded'
    assert metadata['total_evaluations'] == metadata['optimization_evaluations'] + 1
    with (metadata_path.parent / 'history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    best = history[metadata['best_history_index']]
    assert outcome.result.final_eval_value == pytest.approx(float(best['topic:/test/cost']))
    assert not server.optimization_value_manager.subscriptions
    action.destroy()


@pytest.mark.parametrize('algorithm', OPTIMIZERS)
@pytest.mark.parametrize('minimize', [True, False])
def test_result_returns_raw_topic_value_for_every_algorithm(
        ros_system, tmp_path, algorithm, minimize, capfd):
    _, _, client, _ = ros_system
    goal = make_goal(tmp_path, algorithm, 0)
    goal.alignment_topics[0].minimize_not_maximize = minimize
    action, handle = send(client, algorithm, goal)
    outcome = wait_future(handle.get_result_async(), 5)
    assert outcome.status == GoalStatus.STATUS_SUCCEEDED
    assert outcome.result.final_eval_value == pytest.approx(0.13, abs=1e-6)
    metadata = json.loads(next(tmp_path.glob('optimization_logs/*/metadata.json')).read_text())
    assert metadata['best_objective'] == pytest.approx(0.13 if minimize else -0.13, abs=1e-6)
    output = capfd.readouterr()
    logs = output.out + output.err
    assert f'{algorithm}: starting execution.' in logs
    assert f'{algorithm}: progress 0/0 evaluations;' in logs
    assert 'current [/test/cost=0.13]; best [/test/cost=0.13]' in logs
    assert f'{algorithm}: search stopped (max_evaluations); restoring best position.' in logs
    assert f'{algorithm}: execution completed; stop reason=max_evaluations;' in logs
    action.destroy()


def test_progress_logs_include_evaluation_updates(ros_system, tmp_path, capfd):
    _, _, client, _ = ros_system
    action, handle = send(client, 'random_search', make_goal(tmp_path, 'random_search', 12))
    outcome = wait_future(handle.get_result_async(), 10)
    assert outcome.status == GoalStatus.STATUS_SUCCEEDED
    output = capfd.readouterr()
    logs = output.out + output.err
    assert 'random_search: progress 10/12 evaluations;' in logs
    assert 'random_search: execution completed; stop reason=max_evaluations; evaluations=12;' in logs
    action.destroy()


@pytest.mark.parametrize('algorithm', OPTIMIZERS)
@pytest.mark.parametrize('minimize', [True, False])
def test_signed_objective_target_stops_every_algorithm(ros_system, tmp_path, algorithm, minimize):
    _, server, client, _ = ros_system
    threshold = 0.14 if minimize else -0.12
    goal = make_goal(tmp_path, algorithm, 20, objective_threshold=threshold)
    goal.alignment_topics[0].minimize_not_maximize = minimize
    action, handle = send(client, algorithm, goal)
    outcome = wait_future(handle.get_result_async(), 5)
    assert outcome.status == GoalStatus.STATUS_SUCCEEDED
    assert outcome.result.num_of_iterations == 0
    assert outcome.result.final_eval_value == pytest.approx(0.13, abs=1e-6)
    metadata_path = next(tmp_path.glob('optimization_logs/*/metadata.json'))
    metadata = json.loads(metadata_path.read_text())
    assert metadata['stop_reason'] == 'objective_threshold_reached'
    assert server.get_parameter('general.objective_threshold').value == threshold
    saved = yaml.safe_load((metadata_path.parent / 'optimization_parameter.yaml').read_text())
    assert saved['general.objective_threshold'] == threshold
    action.destroy()


@pytest.mark.parametrize('minimize', [True, False])
def test_signed_target_uses_weighted_average_for_same_direction_topics(
        ros_system, tmp_path, minimize):
    _, _, client, _ = ros_system
    publisher = client.create_publisher(Float32, '/test/other', 10)
    timer = client.create_timer(0.01, lambda: publisher.publish(Float32(data=0.4)))
    goal = make_goal(tmp_path, 'random_search', 20,
                     objective_threshold=0.23 if minimize else -0.21)
    goal.alignment_topics = [
        AlignTopic(topic_name='/test/cost', minimize_not_maximize=minimize, weight_factor=2.0),
        AlignTopic(topic_name='/test/other', minimize_not_maximize=minimize, weight_factor=1.0)]
    action, handle = send(client, 'random_search', goal)
    outcome = wait_future(handle.get_result_async(), 5)
    assert outcome.status == GoalStatus.STATUS_SUCCEEDED
    assert outcome.result.num_of_iterations == 0
    assert outcome.result.final_eval_value == pytest.approx(0.22, abs=1e-6)
    metadata = json.loads(next(tmp_path.glob('optimization_logs/*/metadata.json')).read_text())
    assert metadata['stop_reason'] == 'objective_threshold_reached'
    assert timer is not None
    action.destroy()


@pytest.mark.parametrize('threshold, stop_reason', [
    (0.05, 'objective_threshold_reached'), (-0.05, 'max_evaluations')])
def test_signed_objective_target_supports_mixed_directions(
        ros_system, tmp_path, threshold, stop_reason):
    _, _, client, _ = ros_system
    publisher = client.create_publisher(Float32, '/test/other', 10)
    timer = client.create_timer(0.01, lambda: publisher.publish(Float32(data=0.4)))
    goal = make_goal(tmp_path, 'random_search', 0, objective_threshold=threshold)
    goal.alignment_topics = [
        AlignTopic(topic_name='/test/cost', minimize_not_maximize=False, weight_factor=2.0),
        AlignTopic(topic_name='/test/other', minimize_not_maximize=True, weight_factor=1.0)]
    action, handle = send(client, 'random_search', goal)
    outcome = wait_future(handle.get_result_async(), 5)
    assert outcome.status == GoalStatus.STATUS_SUCCEEDED
    assert outcome.result.success
    assert outcome.result.final_eval_value == pytest.approx(0.22, abs=1e-6)
    metadata = json.loads(next(tmp_path.glob('optimization_logs/*/metadata.json')).read_text())
    assert metadata['stop_reason'] == stop_reason
    assert metadata['best_objective'] == pytest.approx((-2 * 0.13 + 0.4) / 3, abs=1e-6)
    assert timer is not None
    action.destroy()


def test_cancel_stops_and_competing_goal_is_rejected(ros_system, tmp_path, capfd):
    robot, server, client, _ = ros_system
    robot.delay = 0.15
    goal = make_goal(tmp_path, 'random_search', 100)
    first, handle = send(client, 'random_search', goal)
    assert handle.accepted
    second, competing = send(client, 'nelder_mead', goal)
    assert not competing.accepted
    wait_future(handle.cancel_goal_async(), 5)
    outcome = wait_future(handle.get_result_async(), 10)
    assert outcome.status == GoalStatus.STATUS_CANCELED
    assert not outcome.result.success
    assert not server.run_lock.locked()
    output = capfd.readouterr()
    logs = output.out + output.err
    assert 'random_search: cancellation requested; stopping execution.' in logs
    assert 'random_search: execution cancelled;' in logs
    assert 'random_search: execution completed;' not in logs
    first.destroy()
    second.destroy()


def test_controller_failure_aborts_and_keeps_partial_results(ros_system, tmp_path, capfd):
    robot, server, client, _ = ros_system
    robot.fail_motion = True
    action, handle = send(client, 'nelder_mead', make_goal(tmp_path, 'nelder_mead'))
    outcome = wait_future(handle.get_result_async(), 10)
    assert outcome.status == GoalStatus.STATUS_ABORTED
    assert not outcome.result.success
    metadata = json.loads(next(tmp_path.glob('optimization_logs/*/metadata.json')).read_text())
    assert metadata['status'] == 'failed'
    assert 'Trajectory failed' in metadata['message']
    assert not server.run_lock.locked()
    output = capfd.readouterr()
    logs = output.out + output.err
    assert 'nelder_mead: execution stopped; reason=Trajectory failed' in logs
    assert 'nelder_mead: execution completed;' not in logs
    action.destroy()


def test_invalid_yaml_update_is_atomic_and_never_moves_robot(ros_system, tmp_path):
    robot, server, client, _ = ros_system
    previous = server.get_parameter('general.max_evaluations').value
    goal = make_goal(tmp_path, 'nelder_mead')
    Path(goal.parameter_file).write_text(
        'general: {max_evaluations: 7}\nnelder_mead: {xatol: -1.0}\n')
    action, handle = send(client, 'nelder_mead', goal)
    outcome = wait_future(handle.get_result_async(), 10)
    assert outcome.status == GoalStatus.STATUS_ABORTED
    assert server.get_parameter('general.max_evaluations').value == previous
    assert not robot.moves
    action.destroy()


def test_unterminated_trajectory_rejects_new_goals(ros_system, tmp_path):
    robot, server, client, _ = ros_system
    robot.delay = 0.5
    goal = make_goal(tmp_path, 'random_search', 1)
    path = Path(goal.parameter_file)
    document = yaml.safe_load(path.read_text())
    document['general']['trajectory_timeout'] = 0.05
    path.write_text(yaml.safe_dump(document))
    first, handle = send(client, 'random_search', goal)
    outcome = wait_future(handle.get_result_async(), 5)
    assert outcome.status == GoalStatus.STATUS_ABORTED
    assert server.motion.has_active_motion()
    second, rejected = send(client, 'random_search', goal)
    assert not rejected.accepted
    deadline = time.monotonic() + 3
    while server.motion.has_active_motion():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    first.destroy()
    second.destroy()


def test_weighted_maximization_and_minimization_are_combined_once(ros_system, tmp_path):
    robot, server, client, _ = ros_system
    publisher = client.create_publisher(Float32, '/test/other', 10)
    timer = client.create_timer(0.01, lambda: publisher.publish(Float32(data=0.4)))
    goal = make_goal(tmp_path, 'random_search', 0)
    goal.alignment_topics = [
        AlignTopic(topic_name='/test/cost', minimize_not_maximize=False, weight_factor=2.0),
        AlignTopic(topic_name='/test/other', minimize_not_maximize=True, weight_factor=1.0)]
    action, handle = send(client, 'random_search', goal)
    outcome = wait_future(handle.get_result_async(), 5)
    assert outcome.status == GoalStatus.STATUS_SUCCEEDED
    assert outcome.result.final_eval_value == pytest.approx((2 * 0.13 + 0.4) / 3, abs=1e-6)
    metadata = json.loads(next(tmp_path.glob('optimization_logs/*/metadata.json')).read_text())
    assert metadata['best_objective'] == pytest.approx((-2 * 0.13 + 0.4) / 3, abs=1e-6)
    assert outcome.result.num_of_iterations == 0
    # Entities remain owned by the fixture's node until its executor has stopped.
    assert timer is not None
    action.destroy()


@pytest.mark.parametrize('scenario', ['success', 'verification_failure', 'cancelled'])
def test_complete_batch_updates_and_restores_scene_and_parameters(ros_system, tmp_path, scenario):
    pytest.importorskip('active_alignment_simulation')
    from copy import deepcopy
    from assembly_manager_interfaces.msg import RefFrame
    from assembly_manager_interfaces.srv import (
        CreateRefFrame, DeleteRefFrame, GetScene, ModifyPoseAbsolut)
    from geometry_msgs.msg import Pose, TransformStamped
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
    from active_alignment_interfaces.action import RunAlignmentBatch
    from active_alignment_simulation.batch_runner_node import BatchRunner
    from active_alignment_skills.dummy_value_publisher import MultiFrameDeviationNode

    robot, server, client, executor = ros_system
    original = RefFrame(frame_name='batch_target', parent_frame='world', pose=Pose())
    original.pose.position.x = 0.7
    original.pose.orientation.w = 1.0
    frames = {'batch_target': deepcopy(original)}
    assembly = Node('fake_assembly_manager')
    dynamic = TransformBroadcaster(assembly)
    static = StaticTransformBroadcaster(assembly)
    cb = ReentrantCallbackGroup()

    def broadcast():
        reference = TransformStamped()
        reference.header.frame_id = 'world'
        reference.child_frame_id = 'batch_reference'
        reference.header.stamp = assembly.get_clock().now().to_msg()
        reference.transform.translation.x = robot.positions['SP_X_Joint']
        reference.transform.translation.y = robot.positions['SP_Y_Joint']
        reference.transform.rotation.w = 1.0
        dynamic.sendTransform(reference)
        for frame in frames.values():
            transform = TransformStamped()
            transform.header.frame_id = frame.parent_frame
            transform.child_frame_id = frame.frame_name
            transform.header.stamp = reference.header.stamp
            transform.transform.translation.x = frame.pose.position.x
            transform.transform.translation.y = frame.pose.position.y
            transform.transform.translation.z = frame.pose.position.z
            transform.transform.rotation = frame.pose.orientation
            static.sendTransform(transform)

    def get_scene(request, response):
        response.scene.ref_frames_in_scene = list(frames.values())
        response.success = True
        return response

    def modify(request, response):
        frames[request.frame_name].pose = deepcopy(request.pose)
        response.success = True
        broadcast()
        return response

    def create(request, response):
        frames[request.ref_frame.frame_name] = deepcopy(request.ref_frame)
        response.success = True
        broadcast()
        return response

    def delete(request, response):
        frames.pop(request.frame_name, None)
        response.success = True
        return response

    assembly.create_service(GetScene, '/assembly_manager/get_scene', get_scene, callback_group=cb)
    assembly.create_service(
        ModifyPoseAbsolut,
        '/assembly_manager/modify_frame_absolut',
        modify,
        callback_group=cb)
    assembly.create_service(
        CreateRefFrame,
        '/assembly_manager/create_ref_frame',
        create,
        callback_group=cb)
    assembly.create_service(
        DeleteRefFrame,
        '/assembly_manager/delete_ref_frame',
        delete,
        callback_group=cb)
    assembly.create_timer(0.01, broadcast, callback_group=cb)
    config = {
        'mode': '2D_TRANS', 'model': {
            'mode_field_radius': 0.1, 'wavelength': 0.1, 'search_factor': 2.0,
            'noise_mode': 'none', 'publish_rate': 100.0},
        'frame_pairs': [{
            'name': 'BatchTest', 'frame_a': 'batch_reference', 'frame_b': 'batch_target'}],
        'target': {'frame_name': 'batch_target', 'reference_frame': 'batch_reference'},
        'frames': [{'frame_name': 'temporary_frame', 'parent_frame': 'world'}],
        'signal_verification': {
            'topic': '/BatchTest/eta_clean', 'timeout': 3.0, 'tolerance': 0.001},
        'initial_joint_positions': {'SP_X_Joint': 0.0, 'SP_Y_Joint': 0.0},
        'alignment_topics': [{'topic_name': '/BatchTest/signal', 'minimize_not_maximize': False,
                              'weight_factor': 1.0}],
        'seed_groups': [{'power_percent': 60.0, 'num_seeds': 1}],
        'setup_settle_time': 0.05, 'reset_movement_time': 0.01,
        'tasks': [{'algorithm': name, 'iterations': 1, 'general': {
            'max_evaluations': 12, 'movement_time': 0.01, 'wait_for_settle': 0.0}}
            for name in ('nelder_mead', 'nelder_mead_manual')],
    }
    config_path = tmp_path / 'simulation_parameter.yaml'
    if scenario == 'verification_failure':
        config['signal_verification'] = {'topic': '/missing/eta_clean', 'timeout': 0.15}
    if scenario == 'cancelled':
        robot.delay = 0.1
        config['tasks'][0]['general']['max_evaluations'] = 200
    config_path.write_text(yaml.safe_dump(config))
    publisher = MultiFrameDeviationNode(parameter_overrides=[
        Parameter('simulation_parameter_file', value=str(config_path)),
        Parameter('policy', value='geometric')])
    batch = BatchRunner()
    for node in (assembly, publisher, batch):
        executor.add_node(node)
    previous_budget = server.get_parameter('general.max_evaluations').value
    action = ActionClient(client, RunAlignmentBatch, '/active_alignment_batch/execute')
    try:
        assert action.wait_for_server(timeout_sec=5)
        handle = wait_future(action.send_goal_async(RunAlignmentBatch.Goal(
            simulation_parameter_file=str(config_path))), 5)
        assert handle.accepted
        if scenario == 'cancelled':
            deadline = time.monotonic() + 10
            while not batch.child_active:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            wait_future(handle.cancel_goal_async(), 5)
        outcome = wait_future(handle.get_result_async(), 40)
        expected = {'success': GoalStatus.STATUS_SUCCEEDED,
                    'verification_failure': GoalStatus.STATUS_ABORTED,
                    'cancelled': GoalStatus.STATUS_CANCELED}[scenario]
        assert outcome.status == expected, outcome.result.message
        if scenario == 'success':
            assert outcome.result.completed_runs == 2
            assert outcome.result.failed_runs == 0
        elif scenario == 'verification_failure':
            assert outcome.result.failed_runs == 2
        assert frames['batch_target'] == original
        assert 'temporary_frame' not in frames
        if scenario != 'cancelled':
            assert robot.positions['SP_X_Joint'] == pytest.approx(0.0)
            assert robot.positions['SP_Y_Joint'] == pytest.approx(0.0)
        assert server.get_parameter('general.max_evaluations').value == previous_budget
        assert server.get_parameter('batch_directory').value == ''
        assert publisher.get_parameter('policy').value == 'geometric'
        directory = Path(outcome.result.results_path)
        summary = json.loads((directory / 'batch_summary.json').read_text())
        if scenario == 'success':
            assert summary['status'] == 'succeeded'
            assert len(summary['runs']) == 2
            assert len(list(directory.glob('run_*/optimization_logs/*/history.csv'))) == 2
            assert len(list(directory.glob('run_*/optimization_parameter.yaml'))) == 2
            assert all(record['verified_initial_signal'] == pytest.approx(0.6, abs=0.001)
                       for record in summary['runs'])
    finally:
        action.destroy()
        executor.shutdown(timeout_sec=10)
        for node in (batch, publisher, assembly):
            executor.remove_node(node)
            node.destroy_node()
