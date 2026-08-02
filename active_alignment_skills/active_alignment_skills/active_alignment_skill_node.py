#!/usr/bin/env python3

# Standard library
import asyncio
import csv
import datetime
import json
import math
import os
import random
import shutil
import threading
import time
from functools import partial
from pathlib import Path
from copy import deepcopy
import gc

# Third-party
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from urdf_parser_py.urdf import URDF, Joint as URDFJoint

# ROS2
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration as MsgDuration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.srv import GetParameters
from rclpy.action import (
    ActionClient,
    ActionServer,
    CancelResponse,
    GoalResponse,
)
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import List, Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Project
from active_alignment_interfaces.action import ActiveAlign, ActiveAlignment
from active_alignment_interfaces.msg import (
    AlignTopic,
    JointAlign,
    JointAlignmentResult,
)
from active_alignment_skills.py_modules.AlignmentControllerManager import (
    AlignmentController,
    AlignmentControllerManager,
    ControllerJoint,
)
from active_alignment_skills.py_modules.JointLimitFetcher import JointLimitFetcher
from active_alignment_skills.py_modules.OptimizationValueManager import (
    OptimizationValueManager,
)
from assembly_manager_interfaces.srv import SpawnFramesFromDescription

# ============================================================
# EXCEPTIONS & RESULT CLASS
# ============================================================
class BreakOptimization(Exception):
    def __init__(self, message="Optimization process was interrupted."):
        self.message = message
        super().__init__(self.message)

class StopOptimization(Exception):
    def __init__(self, message="Signal threshold reached."):
        self.message = message
        super().__init__(self.message)

class OptimizationResult:
    def __init__(self, joint_results: dict[str, float], num_of_iterations: int, final_eval_value: float):
        self.joint_results = joint_results
        self.num_of_iterations = num_of_iterations
        self.final_eval_value = final_eval_value

    def to_result_msg(self, result: ActiveAlign.Result):
        result.num_of_iterations = self.num_of_iterations
        result.final_eval_value = self.final_eval_value

        for joint_name, joint_value in self.joint_results.items():
            joint_result_msg = JointAlignmentResult()
            joint_result_msg.joint_name = joint_name
            joint_result_msg.joint_value = joint_value
            result.joint_results.append(joint_result_msg)

# ============================================================
# METRICS READER
# ============================================================
class MetricsReader:
    def __init__(self, node, topic_base):
        self.node = node
        self.lock = threading.Lock()
        self.metrics = {
            'signal': 0.0,
            'eta_lateral': 0.0,
            'eta_angular': 0.0,
            'eta_clean': 0.0,
            'distance': 0.0,
            'angle_deviation': 0.0,
            'dist_perc': 0.0,
            'angle_perc': 0.0,
        }
        self.subs = {}
        for key in self.metrics.keys():
            topic_name = f'{topic_base}/{key}'
            self.subs[key] = node.create_subscription(
                Float32, topic_name, self._make_callback(key), 10)

    def _make_callback(self, key):
        def callback(msg):
            with self.lock:
                self.metrics[key] = msg.data
        return callback

    def get_metrics(self):
        with self.lock:
            return self.metrics.copy()

# ============================================================
# MAIN SERVER CLASS
# ============================================================
class ActiveAlignmentServer(Node):
    TRANSLATION_JOINTS = {"SP_X_Joint", "SP_Y_Joint", "SP_Z_Joint"}
    ROTATION_JOINTS = {"SP_A_Joint", "SP_B_Joint", "SP_C_Joint"}
    ALL_JOINTS = list(TRANSLATION_JOINTS) + list(ROTATION_JOINTS)

    def __init__(self):
        super().__init__('active_alignment_server')
        self.cb_reentrant = ReentrantCallbackGroup()
        self.cb_subscriptions = MutuallyExclusiveCallbackGroup()

        self._joint_limit_fetcher = JointLimitFetcher(self)
        self._action_clients: dict[str, ActionClient] = {}
        self.current_algorithm_name = None

        # ---- Action servers ----
        self._action_server_hill_climb = ActionServer(
            self, ActiveAlign, f'{self.get_name()}/exec_active_alignment_hill_climb',
            execute_callback=self.make_execute_callback(self.hill_climb_optimization),
            goal_callback=self.goal_callback, cancel_callback=self.cancel_callback
        )
        self._action_server_nelder = ActionServer(
            self, ActiveAlign, f'{self.get_name()}/exec_active_alignment_nelder_mead',
            execute_callback=self.make_execute_callback(self.nelder_mead_optimization),
            goal_callback=self.goal_callback, cancel_callback=self.cancel_callback
        )
        self._action_server_random = ActionServer(
            self, ActiveAlign, f'{self.get_name()}/exec_active_alignment_random_search',
            execute_callback=self.make_execute_callback(self.random_search_optimization),
            goal_callback=self.goal_callback, cancel_callback=self.cancel_callback
        )
        self._action_server_batch = ActionServer(
            self, ActiveAlign, f'{self.get_name()}/exec_active_alignment_batch',
            execute_callback=self.make_execute_callback(self.batch_optimization),
            goal_callback=self.goal_callback, cancel_callback=self.cancel_callback
        )

        # ---- ROS clients and subscribers ----
        self.cli = self.create_client(ListControllers, '/controller_manager/list_controllers')
        self._current_joint_state_positions = {}
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.optimization_value_manager = OptimizationValueManager(self)

        # ---- Results logging ----
        self.results_base_dir = Path(get_package_share_directory('active_alignment_skills')) / "optimization_results"
        self.results_base_dir.mkdir(exist_ok=True)
        self.session_folder = None
        self.run_counters = {}
        self.session_start_time = None
        self.metadata = {}
        self.metrics_reader = None

        # ---- Spawn service client ----
        self.spawn_client = self.create_client(
            SpawnFramesFromDescription,
            '/assembly_manager/spawn_frames_from_description',
            callback_group=self.cb_reentrant
        )
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn service...')

        self.get_logger().info("Active Alignment Server is ready.")

    # ============================================================
    # HELPERS
    # ============================================================
    def make_execute_callback(self, optimization_func):
        async def _callback(goal_handle):
            return await self.execute_callback(goal_handle, optimization_func)
        return _callback

    def goal_callback(self, goal_request: ActiveAlign.Goal):
        self.get_logger().info(f"Received goal: {str(goal_request)}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Cancel request received")
        return CancelResponse.ACCEPT

    @staticmethod
    def get_doc_path() -> Path:
        script_path = Path(__file__).resolve()
        if "install" in script_path.parts:
            for parent in script_path.parents:
                if parent.name == "install":
                    ws_root = parent.parent
                    break
            else:
                ws_root = script_path.parents[4]
            doc_path = ws_root / "src" / "pm_active_alignment" / "active_alignment_skills" / "doc"
        else:
            doc_path = script_path.parent.parent / "doc"
        return doc_path

    def read_metadata_from_files(self):
        doc_path = self.get_doc_path()
        config_path = doc_path / "alignment_config.json"
        spawn_path = doc_path / "spawn_test_frames.json"
        metadata = {}
        self.get_logger().info(f"📁 Looking for config at: {config_path}")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                metadata.update(config)
                self.get_logger().info(f"✅ Loaded config from {config_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load config: {e}, using defaults.")
        else:
            self.get_logger().warn(f"❌ Config file not found at {config_path}, using defaults.")
            metadata['W0'] = 5.2e-6
            metadata['LAMBDA'] = 1.55e-6
            metadata['SEARCH_FACTOR'] = 3
            metadata['THETA_EFF'] = 1.55e-6 / (np.pi * 5.2e-6)

        if spawn_path.exists():
            try:
                with open(spawn_path, 'r') as f:
                    spawn = json.load(f)
                trans = spawn['frames'][0]['transformation']['translation']
                rot = spawn['frames'][0]['transformation']['rotation']
                Z_BASE = 90.0 - 32.62570
                dx_mm = trans['X']; dy_mm = trans['Y']; dz_mm = trans['Z'] - Z_BASE
                r = R.from_quat([rot['X'], rot['Y'], rot['Z'], rot['W']])
                metadata['initial_distance_um'] = math.sqrt(dx_mm**2 + dy_mm**2 + dz_mm**2) * 1000
                metadata['initial_angle_deg'] = math.degrees(r.magnitude())
                metadata['initial_signal'] = metadata.get('INITIAL_POWER_PERCENT', 0)
                metadata['initial_translation'] = {'X': dx_mm, 'Y': dy_mm, 'Z': dz_mm}
                metadata['initial_rotation_quat'] = rot
                self.get_logger().info(f"✅ Loaded spawn from {spawn_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load spawn: {e}")
        else:
            self.get_logger().warn(f"❌ Spawn JSON not found at {spawn_path}, cannot read initial misalignment.")
        return metadata

    def load_seed_poses(self):
        doc_path = self.get_doc_path()
        seed_file = doc_path / "seed_poses.json"
        if not seed_file.exists():
            self.get_logger().warn(f"Seed poses file not found: {seed_file}")
            return []
        try:
            with open(seed_file, 'r') as f:
                data = json.load(f)
            self.seed_poses = data
            self.get_logger().info(f"Loaded {len(data)} seed groups from {seed_file}")
            return data
        except Exception as e:
            self.get_logger().error(f"Failed to load seed poses: {e}")
            return []

    def print_subscription_count(self):
        """Log the number of active subscriptions."""
        try:
            sub_count = len(self._subscriptions)
            self.get_logger().info(f"📋 Active subscriptions: {sub_count}")
        except AttributeError:
            self.get_logger().warn("Could not access _subscriptions; using fallback.")
            if hasattr(self, 'metrics_reader') and self.metrics_reader is not None:
                self.get_logger().info(f"📋 MetricsReader subs: {len(self.metrics_reader.subs)}")

    def reset_joints_to_center(self, alignment_controller_mgr, reset_value=0.5):
        self.get_logger().info(f"🔄 Resetting joints to {reset_value}...")
        joints = alignment_controller_mgr.get_all_joint_states_as_mapped()
        for joint_name in joints.keys():
            joints[joint_name] = reset_value
        alignment_controller_mgr.set_joint_states_from_mapped(joints)
        self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=0.1)
        time.sleep(0.01)
        current = alignment_controller_mgr.get_all_joint_states_as_mapped()
        self.get_logger().info(f"✅ Reset complete. Current joints: {current}")

    def compute_dynamic_percentages(self, distance, angle_deviation, joint_names):
        n_trans = sum(1 for j in joint_names if j in self.TRANSLATION_JOINTS)
        n_rot = sum(1 for j in joint_names if j in self.ROTATION_JOINTS)
        SEARCH_FACTOR = self.metadata.get('SEARCH_FACTOR', 3)
        W0 = self.metadata.get('W0', 5.2e-6)
        THETA_EFF = self.metadata.get('THETA_EFF', 1.55e-6 / (np.pi * 5.2e-6))
        max_dist = 2 * math.sqrt(n_trans) * SEARCH_FACTOR * W0 if n_trans > 0 else 1.0
        max_angle = 2 * math.sqrt(n_rot) * SEARCH_FACTOR * THETA_EFF if n_rot > 0 else 1.0
        dist_perc = (distance / max_dist) * 100 if max_dist > 0 else 0.0
        angle_perc = (angle_deviation / max_angle) * 100 if max_angle > 0 else 0.0
        return dist_perc, angle_perc

    def build_dummy_goal_from_metadata(self):
        if not self.metadata:
            self.metadata = self.read_metadata_from_files()
        enabled_joints = self.metadata.get('ENABLED_JOINTS', {})
        if not enabled_joints:
            enabled_joints = {j: True for j in self.ALL_JOINTS}
        SEARCH_FACTOR = self.metadata.get('SEARCH_FACTOR', 3)
        W0 = self.metadata.get('W0', 5.2e-6)
        THETA_EFF = self.metadata.get('THETA_EFF', 1.55e-6/(np.pi*5.2e-6))
        trans_limit = SEARCH_FACTOR * W0
        rot_limit = SEARCH_FACTOR * THETA_EFF
        dummy_goal = ActiveAlign.Goal()
        dummy_goal.alignment_topics = []
        for joint_name, is_enabled in enabled_joints.items():
            if is_enabled:
                ja = JointAlign()
                ja.joint_name = joint_name
                ja.has_constraint = True
                if joint_name in self.TRANSLATION_JOINTS:
                    ja.upper_limit = trans_limit
                    ja.lower_limit = -trans_limit
                elif joint_name in self.ROTATION_JOINTS:
                    ja.upper_limit = rot_limit
                    ja.lower_limit = -rot_limit
                else:
                    continue
                dummy_goal.active_joints.append(ja)
        return dummy_goal

    def get_enabled_counts(self):
        if not self.metadata:
            self.metadata = self.read_metadata_from_files()
        enabled = self.metadata.get('ENABLED_JOINTS', {})
        n_trans = sum(1 for j in self.TRANSLATION_JOINTS if enabled.get(j, False))
        n_rot = sum(1 for j in self.ROTATION_JOINTS if enabled.get(j, False))
        return n_trans, n_rot

    def compute_initial_power_from_seed(self, seed, n_trans, n_rot):
        dx_m, dy_m, dz_m = seed['dx_m'], seed['dy_m'], seed['dz_m']
        quat = seed['quat']
        d_actual = np.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
        r = R.from_quat([quat['X'], quat['Y'], quat['Z'], quat['W']])
        theta_actual = r.magnitude()
        W0 = self.metadata.get('W0', 5.2e-6)
        THETA_EFF = self.metadata.get('THETA_EFF', 1.55e-6 / (np.pi * W0))
        eta_lateral = np.exp(-(d_actual / W0)**2) if n_trans > 0 else 1.0
        eta_angular = np.exp(-(theta_actual / THETA_EFF)**2) if n_rot > 0 else 1.0
        return eta_lateral * eta_angular * 100.0

    async def init_controller_manager(self, alignment_goal: ActiveAlign.Goal):
        alignment_controller_mgr = AlignmentControllerManager()
        active_joints = alignment_goal.active_joints
        if len(active_joints) == 0:
            raise ValueError("No active joints provided in goal.")
        involved_controllers = set()
        for joint_align in active_joints:
            _joint_name = joint_align.joint_name
            _joint_info = await self._joint_limit_fetcher.get_joint_info(_joint_name)
            _controller_for_joint = _joint_info["controller_name"]
            if _joint_info["controller_type"] != "joint_trajectory_controller/JointTrajectoryController":
                raise ValueError(f"Unsupported controller type for joint '{_joint_name}': {_joint_info['controller_type']}")
            _current_joint_value = self.get_current_joint_state(_joint_name)
            _c_joint = ControllerJoint(is_active_joint=True,
                                       joint_align_msg=joint_align,
                                       initial_joint_value=_current_joint_value)
            _c_joint.initialize_joint_limits(upper_limit=_joint_info["upper"], lower_limit=_joint_info["lower"])
            involved_controllers.add(_controller_for_joint)
            alignment_controller_mgr.add_joint_to_controller(controller_name=_controller_for_joint, joint=_c_joint)
        for ctrl in involved_controllers:
            _joints = await self._joint_limit_fetcher.get_joints_for_controller(ctrl)
            for j in _joints:
                if alignment_controller_mgr.is_joint_in_any_controller(j):
                    continue
                _msg = JointAlign()
                _msg.joint_name = j
                _current_joint_value = self.get_current_joint_state(j)
                _c_joint = ControllerJoint(is_active_joint=False,
                                           initial_joint_value=_current_joint_value,
                                           joint_align_msg=_msg)
                alignment_controller_mgr.add_joint_to_controller(controller_name=ctrl, joint=_c_joint)
        self.get_logger().info(f"Alignment Controller Manager Debug Info: {alignment_controller_mgr.get_debug_info()}")
        return alignment_controller_mgr

    async def execute_callback(self, goal_handle, optimization_method):
        if optimization_method != self.batch_optimization:
            goal: ActiveAlign.Goal = goal_handle.request
            if len(goal.alignment_topics) == 0:
                self.get_logger().error("No alignment topics provided.")
                result = ActiveAlign.Result(); result.success = False; goal_handle.abort(); return result
        if self.session_folder is None:
            self.session_start_time = datetime.datetime.now()
            session_timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            self.session_folder = self.results_base_dir / f"session_{session_timestamp}"
            self.session_folder.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"📁 Results session folder: {self.session_folder}")
        start_time = time.time()
        try:
            if optimization_method == self.batch_optimization:
                result = await optimization_method(goal_handle)
                if result.success: goal_handle.succeed()
                else: goal_handle.abort()
                return result
            # Single algorithm calls
            goal: ActiveAlign.Goal = goal_handle.request
            alignment_topics = goal.alignment_topics
            alignment_controller_mgr = await self.init_controller_manager(goal)
            self.optimization_value_manager.init(alignment_topics)
            topic_base = alignment_topics[0].topic_name.rsplit('/', 1)[0]
            self.metrics_reader = MetricsReader(self, topic_base)
            self.metadata = self.read_metadata_from_files()

            algo_name = optimization_method.__name__.replace('_optimization', '').replace('_', ' ').title()
            self.current_algorithm_name = algo_name
            if algo_name not in self.run_counters:
                self.run_counters[algo_name] = 0
            self.run_counters[algo_name] += 1
            run_number = self.run_counters[algo_name]
            run_folder_name = f"{algo_name.replace(' ', '_')}_{run_number}"
            run_folder = self.session_folder / run_folder_name
            run_folder.mkdir(exist_ok=True)

            optim_result = optimization_method(
                alignment_controller_mgr,
                goal_handle,
                metrics_reader=self.metrics_reader,
                metadata=self.metadata,
                run_folder=run_folder,
                start_time=start_time
            )

            self.optimization_value_manager.clear()
            result = ActiveAlign.Result()
            optim_result.to_result_msg(result)
            result.success = True
            goal_handle.succeed()
            return result

        except BreakOptimization:
            self.get_logger().info("Optimization interrupted.")
            result = ActiveAlign.Result(); result.success = False; goal_handle.abort(); return result
        except Exception as e:
            self.get_logger().error(f"Exception in execute_callback: {e}")
            result = ActiveAlign.Result(); result.success = False; goal_handle.abort(); return result
        finally:
            if optimization_method != self.batch_optimization:
                self.optimization_value_manager.clear()

    # ============================================================
    # BATCH OPTIMIZATION – with memory cleanup & spawn client recreation
    # ============================================================

    async def batch_optimization(self, goal_handle):
        self.get_logger().info("🚀 Batch optimization started")
        self.print_subscription_count()
        doc_path = self.get_doc_path()
        config_path = doc_path / "batch_config.json"
        self.get_logger().info(f"📁 Looking for batch config at: {config_path}")
        if not config_path.exists():
            self.get_logger().error(f"Batch config not found at {config_path}")
            result = ActiveAlign.Result()
            result.success = False
            return result
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load batch config: {e}")
            result = ActiveAlign.Result()
            result.success = False
            return result

        tasks = config.get('tasks', [])
        reset_position = config.get('reset_position', 0.5)
        seed_groups_spec = config.get('seed_groups', [])
        overall_success = True

        self.load_seed_poses()
        if not self.seed_poses and seed_groups_spec:
            self.get_logger().error("Seed groups specified but no seed poses loaded.")
            result = ActiveAlign.Result()
            result.success = False
            return result

        if not self.metadata:
            self.metadata = self.read_metadata_from_files()
        dummy_goal = self.build_dummy_goal_from_metadata()
        n_trans, n_rot = self.get_enabled_counts()

        algo_map = {
            "Hill Climb": self.hill_climb_optimization,
            "Nelder-Mead": self.nelder_mead_optimization,
            "Random Search": self.random_search_optimization,
        }

        self.get_logger().info(f"📋 Tasks to run: {[(t['algorithm'], t.get('iterations', 1)) for t in tasks]}")

        # Track if we ever created a manager (so we know if we can reset)
        manager_created = False

        for group_idx, group_spec in enumerate(seed_groups_spec):
            power = group_spec.get('power_percent')
            num_seeds = group_spec.get('num_seeds', 1)
            self.get_logger().info(f"🔍 Processing group {group_idx+1}/{len(seed_groups_spec)}: power={power}, num_seeds={num_seeds}")

            group_data = None
            for g in self.seed_poses:
                if abs(g['power_percent'] - power) < 1e-6:
                    group_data = g
                    break
            if not group_data:
                self.get_logger().warn(f"No seed poses found for power {power}%. Skipping.")
                continue

            seeds = group_data['seeds'][:num_seeds]
            self.get_logger().info(f"🔘 Testing {len(seeds)} seeds at {power}% power level.")

            # If there are no seeds, skip this group entirely (no seed loop, no algorithm runs)
            if not seeds:
                self.get_logger().info(f"⏭️ Skipping group {power}% (num_seeds=0)")
                continue

            for seed_idx, seed in enumerate(seeds):
                self.get_logger().info(f"--- Seed {seed_idx+1}/{len(seeds)} (seed={seed['seed']}) ---")

                # ---- Check cancellation before writing spawn ----
                if goal_handle.is_cancel_requested:
                    self.get_logger().info("Batch cancelled before writing spawn.")
                    result = ActiveAlign.Result()
                    result.success = False
                    return result

                if not self.write_spawn_json_from_seed(seed):
                    self.get_logger().error("Failed to write spawn JSON for seed. Skipping.")
                    continue

                if goal_handle.is_cancel_requested:
                    self.get_logger().info("Batch cancelled before reloading spawn.")
                    result = ActiveAlign.Result()
                    result.success = False
                    return result

                if not await self.reload_spawn_frames():
                    self.get_logger().error("Failed to reload spawn frames. Skipping seed.")
                    continue

                time.sleep(0.2)

                run_metadata = deepcopy(self.metadata)
                initial_power = self.compute_initial_power_from_seed(seed, n_trans, n_rot)
                run_metadata['INITIAL_POWER_PERCENT'] = initial_power
                run_metadata['seed_number'] = seed['seed']
                run_metadata['power_percent'] = group_spec['power_percent']
                run_metadata['dx_m'] = seed['dx_m']; run_metadata['dy_m'] = seed['dy_m']; run_metadata['dz_m'] = seed['dz_m']
                run_metadata['d_actual'] = np.sqrt(seed['dx_m']**2 + seed['dy_m']**2 + seed['dz_m']**2)
                run_metadata['theta_actual'] = R.from_quat([seed['quat']['X'], seed['quat']['Y'], seed['quat']['Z'], seed['quat']['W']]).magnitude()
                run_metadata['initial_distance_um'] = run_metadata['d_actual'] * 1e6
                run_metadata['initial_angle_deg'] = np.degrees(run_metadata['theta_actual'])
                run_metadata['initial_signal'] = initial_power

                # ---- Verify spawn reload with signal check (5% tolerance) ----
                expected_signal = run_metadata['INITIAL_POWER_PERCENT'] / 100.0
                signal_verified = False
                for attempt in range(5):
                    temp_reader = MetricsReader(self, "/Test1")
                    try:
                        current_signal = temp_reader.get_metrics()['signal']
                        self.get_logger().info(f"🔍 Signal check attempt {attempt+1}: current={current_signal:.4f}, expected={expected_signal:.4f}")
                        if abs(current_signal - expected_signal) < 0.05:
                            signal_verified = True
                            self.get_logger().info("✅ Signal verified, TF is consistent.")
                            break
                        else:
                            self.get_logger().warn(f"⚠️ Signal mismatch (attempt {attempt+1}), waiting for TF to propagate...")
                            time.sleep(0.2)
                    finally:
                        for sub in temp_reader.subs.values():
                            self.destroy_subscription(sub)
                        del temp_reader

                if not signal_verified:
                    self.get_logger().warn("⚠️ Signal could not be verified within 5 attempts. Proceeding anyway.")

                # ---- Inner loop over tasks ----
                for task_idx, task in enumerate(tasks):
                    algo_name = task.get('algorithm')
                    iterations = task.get('iterations', 1)
                    params = task.get('params', {})
                    self.get_logger().info(f"  🔹 Task {task_idx+1}/{len(tasks)}: {algo_name} ({iterations} runs)")
                    if algo_name not in algo_map:
                        self.get_logger().error(f"Unknown algorithm: {algo_name}")
                        continue
                    method = algo_map[algo_name]

                    # If iterations is 0, skip this algorithm entirely
                    if iterations == 0:
                        self.get_logger().info(f"  ⏭️ Skipping {algo_name} (iterations=0)")
                        continue

                    # We'll track if we actually ran any runs, so we know if we need to reset
                    ran_any_runs = False

                    for run_idx in range(iterations):
                        if goal_handle.is_cancel_requested:
                            self.get_logger().info("Batch cancelled before run.")
                            result = ActiveAlign.Result()
                            result.success = False
                            return result

                        alignment_controller_mgr = await self.init_controller_manager(dummy_goal)
                        if not alignment_controller_mgr:
                            self.get_logger().error("Failed to initialise controller manager for run.")
                            overall_success = False
                            continue

                        manager_created = True
                        ran_any_runs = True

                        self.reset_joints_to_center(alignment_controller_mgr, reset_position)
                        time.sleep(0.1)

                        self.current_algorithm_name = algo_name
                        if algo_name not in self.run_counters:
                            self.run_counters[algo_name] = 0
                        self.run_counters[algo_name] += 1
                        run_number = self.run_counters[algo_name]
                        power_str = f"{power:.0f}" if power == int(power) else f"{power:.1f}"
                        run_folder_name = f"{algo_name.replace(' ', '_')}_{run_number}_S{seed['seed']}_P{power_str}"
                        if self.session_folder is None:
                            self.session_start_time = datetime.datetime.now()
                            session_timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
                            self.session_folder = self.results_base_dir / f"session_{session_timestamp}"
                            self.session_folder.mkdir(parents=True, exist_ok=True)
                        run_folder = self.session_folder / run_folder_name
                        run_folder.mkdir(exist_ok=True)

                        topic_base = "/Test1"
                        metrics_reader = MetricsReader(self, topic_base)
                        start_time = time.time()

                        kwargs = {
                            'alignment_controller_mgr': alignment_controller_mgr,
                            'goal_handle': goal_handle,
                            'metrics_reader': metrics_reader,
                            'metadata': run_metadata,
                            'run_folder': run_folder,
                            'start_time': start_time,
                        }
                        kwargs.update(params)

                        try:
                            self.get_logger().info(f"    🚀 Calling {algo_name} (run {run_idx+1}/{iterations})...")
                            result = method(**kwargs)
                            self.get_logger().info(f"✅ Run {run_folder_name} completed. Best signal: {result.final_eval_value:.6f}")
                        except BreakOptimization as e:
                            self.get_logger().info(f"Run cancelled: {e}")
                            raise
                        except Exception as e:
                            self.get_logger().error(f"❌ Run {run_folder_name} failed: {e}", exc_info=True)
                            overall_success = False

                        self.reset_joints_to_center(alignment_controller_mgr, reset_position)
                        self.get_logger().info(f"🔄 Reset after run {run_idx+1} for {algo_name}.")

                        try:
                            if metrics_reader is not None:
                                del metrics_reader
                            gc.collect()
                        except Exception as e:
                            self.get_logger().debug(f"Cleanup warning: {e}")

                    # After finishing all runs for this algorithm, reset to center only if we actually ran something
                    if ran_any_runs:
                        self.reset_joints_to_center(alignment_controller_mgr, reset_position)
                        self.get_logger().info(f"🔄 Reset to center after {algo_name} runs for this seed.")

                    self.get_logger().info("♻️ Recreating spawn service client after algorithm batch...")
                    try:
                        if hasattr(self, 'spawn_client') and self.spawn_client is not None:
                            self.spawn_client.destroy()
                            del self.spawn_client
                    except:
                        pass
                    self.spawn_client = self.create_client(
                        SpawnFramesFromDescription,
                        '/assembly_manager/spawn_frames_from_description',
                        callback_group=self.cb_reentrant
                    )
                    while not self.spawn_client.wait_for_service(timeout_sec=1.0):
                        self.get_logger().info('Waiting for spawn service...')
                    gc.collect()

                # ---- Final reset after seed (only if we created a manager) ----
                if manager_created and alignment_controller_mgr is not None:
                    self.reset_joints_to_center(alignment_controller_mgr, reset_position)
                    self.get_logger().info(f"🔄 Final reset after seed {seed_idx+1}.")
                    gc.collect()
                else:
                    self.get_logger().info(f"ℹ️ No manager created for seed {seed_idx+1}, skipping final reset.")

        result_msg = ActiveAlign.Result()
        result_msg.success = overall_success
        self.get_logger().info("🏁 Batch optimization finished.")
        return result_msg

    # ============================================================
    # SPAWN HELPERS
    # ============================================================
    def write_spawn_json_from_seed(self, seed):
        doc_path = self.get_doc_path()
        spawn_path = doc_path / "spawn_test_frames.json"
        dx_m, dy_m, dz_m = seed['dx_m'], seed['dy_m'], seed['dz_m']
        quat = seed['quat']
        Z_BASE = 90.0 - 32.62570
        dx_mm = dx_m * 1000; dy_mm = dy_m * 1000; dz_mm = dz_m * 1000
        z_mm = Z_BASE + dz_mm
        spawn_data = {
            "document_units": "mm",
            "unique_identifier": "AL_Test_Frames_",
            "frames": [
                {"name": "Target_1", "parent_frame": "SmarPod_Origin",
                 "transformation": {"translation": {"X": dx_mm, "Y": dy_mm, "Z": z_mm},
                                    "rotation": quat}, "constraints": {}},
                {"name": "Align_1", "parent_frame": "Smarpod_Part_Spawn",
                 "transformation": {"translation": {"X": 0.0, "Y": 0.0, "Z": 0.0},
                                    "rotation": {"X": 0.0, "Y": 0.0, "Z": 0.0, "W": 1.0}},
                 "constraints": {}}
            ]
        }
        try:
            with open(spawn_path, 'w') as f:
                json.dump(spawn_data, f, indent=2)
            self.get_logger().info(f"✅ Updated spawn JSON with seed offset: dx={dx_m*1e6:.2f}um, dy={dy_m*1e6:.2f}um")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to write spawn JSON: {e}")
            return False

    async def reload_spawn_frames(self):
        if not self.spawn_client.service_is_ready():
            self.get_logger().error("Spawn service not available.")
            return False
        req = SpawnFramesFromDescription.Request()
        req.dict_or_path = str(self.get_doc_path() / "spawn_test_frames.json")
        try:
            future = self.spawn_client.call_async(req)
            response = await future
            if response.success:
                self.get_logger().info("✅ Spawn frames reloaded successfully.")
                return True
            else:
                self.get_logger().error(f"❌ Spawn frames reload failed: {response.message}")
                return False
        except Exception as e:
            self.get_logger().error(f"❌ Spawn service call failed: {e}")
            return False

    # ============================================================
    # OPTIMIZATION METHODS (unchanged)
    # ============================================================

    def hill_climb_optimization(self,
                                alignment_controller_mgr,
                                goal_handle,
                                max_iterations: int = 80,
                                step_size: float = 0.1,
                                stop_threshold: float = 0.99,
                                metrics_reader: MetricsReader = None,
                                metadata: dict = None,
                                run_folder: Path = None,
                                start_time: float = None,
                                maximize: bool = True,
                                coordinate_wise: bool = True,
                                adaptive_step: bool = True,
                                step_increase_factor: float = 1.2,
                                step_decrease_factor: float = 0.5,
                                min_step: float = 1e-4,
                                max_step: float = 0.3,
                                movement_time: float = 0.01,
                                **kwargs) -> OptimizationResult:

        self.current_algorithm_name = "Hill Climb"

        best_positions_mapped = alignment_controller_mgr.get_all_joint_states_as_mapped()
        joint_names = list(best_positions_mapped.keys())
        self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=movement_time)

        # ---- HARDCORED PRE-FLIGHT CHECK ----
        need_fallback = False
        for j in joint_names:
            if j not in self._current_joint_state_positions:
                continue
            if abs(self._current_joint_state_positions[j]) > 1e-4:
                need_fallback = True
                self.get_logger().warn(f"Joint {j} physical position {self._current_joint_state_positions[j]:.6f} not at center (0.0)")
                break

        if need_fallback:
            self.get_logger().warn("⚠️ Joints physically not at center. Forcing each joint to 0.5 with physical verification...")
            current_norm = alignment_controller_mgr.get_all_joint_states_as_mapped()
            for j in joint_names:
                current_norm[j] = 0.5
                alignment_controller_mgr.set_joint_states_from_mapped(current_norm)
                self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=0.05)
                time.sleep(0.05)
                if j in self._current_joint_state_positions and abs(self._current_joint_state_positions[j]) > 1e-4:
                    self.get_logger().error(f"❌ Joint {j} still not at center after forced move! Physical: {self._current_joint_state_positions[j]:.6f}")
                    self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=0.2)
                    time.sleep(0.1)
            # Re‑fetch after fallback
            best_positions_mapped = alignment_controller_mgr.get_all_joint_states_as_mapped()
            joint_names = list(best_positions_mapped.keys())
            self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=movement_time)

        # History containers
        signal_hist = []
        eta_lat_hist = []
        eta_ang_hist = []
        eta_clean_hist = []
        dist_perc_hist = []
        angle_perc_hist = []
        joint_hist = []
        eval_count = 0
        stop_reason = "max_measurements"

        def measure_and_record(position_dict):
            nonlocal eval_count
            eval_count += 1

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                raise BreakOptimization("Cancelled during Hill Climb measurement")

            alignment_controller_mgr.set_joint_states_from_mapped(position_dict)
            self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=movement_time)
            time.sleep(0.05)

            if metrics_reader is not None:
                metrics = metrics_reader.get_metrics()
                signal = metrics['signal']
                dist_perc, angle_perc = self.compute_dynamic_percentages(
                    metrics['distance'], metrics['angle_deviation'], joint_names)
                eta_lat = metrics['eta_lateral']
                eta_ang = metrics['eta_angular']
                eta_clean = metrics['eta_clean']
            else:
                signal = self.optimization_value_manager.get_eval_value(wait_for_update_sec=5, check_for_new_values=True)
                dist_perc = angle_perc = eta_lat = eta_ang = eta_clean = 0.0

            signal_hist.append(signal)
            eta_lat_hist.append(eta_lat)
            eta_ang_hist.append(eta_ang)
            eta_clean_hist.append(eta_clean)
            dist_perc_hist.append(dist_perc)
            angle_perc_hist.append(angle_perc)
            joint_hist.append(list(position_dict.values()))
            return signal

        # --- Initial measurement ---
        initial_signal = measure_and_record(best_positions_mapped)
        best_eval = initial_signal
        best_positions = best_positions_mapped.copy()
        self.get_logger().info(f"Start HC: initial signal = {best_eval:.6f}")

        if best_eval >= stop_threshold:
            stop_reason = "threshold_reached"
            self.get_logger().info(f"✅ Initial signal already ≥ {stop_threshold*100}%")
        else:
            sweep_idx = 0
            while eval_count < max_iterations:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    stop_reason = "cancelled"
                    raise BreakOptimization("Cancelled during Hill Climb sweep")

                improvement_found = False
                threshold_reached = False

                if coordinate_wise:
                    for joint in joint_names:
                        if eval_count >= max_iterations:
                            stop_reason = "max_measurements"
                            threshold_reached = True
                            break

                        # Positive step
                        candidate = best_positions.copy()
                        candidate[joint] = np.clip(best_positions[joint] + step_size, 0.0, 1.0)
                        signal = measure_and_record(candidate)
                        if signal >= stop_threshold:
                            stop_reason = "threshold_reached"
                            self.get_logger().info(f"✅ Reached {stop_threshold*100}% (signal={signal:.6f})")
                            threshold_reached = True
                            break
                        if (signal > best_eval) if maximize else (signal < best_eval):
                            best_positions, best_eval = candidate, signal
                            improvement_found = True
                            continue

                        # Negative step
                        candidate = best_positions.copy()
                        candidate[joint] = np.clip(best_positions[joint] - step_size, 0.0, 1.0)
                        signal = measure_and_record(candidate)
                        if signal >= stop_threshold:
                            stop_reason = "threshold_reached"
                            self.get_logger().info(f"✅ Reached {stop_threshold*100}% (signal={signal:.6f})")
                            threshold_reached = True
                            break
                        if (signal > best_eval) if maximize else (signal < best_eval):
                            best_positions, best_eval = candidate, signal
                            improvement_found = True

                    if threshold_reached:
                        break

                    if not improvement_found:
                        if adaptive_step:
                            step_size = max(step_size * step_decrease_factor, min_step)
                    else:
                        if adaptive_step:
                            step_size = min(step_size * step_increase_factor, max_step)

                else:
                    # Simultaneous perturbation
                    if eval_count >= max_iterations:
                        break
                    candidate = {j: np.clip(best_positions[j] + step_size, 0.0, 1.0) for j in joint_names}
                    signal = measure_and_record(candidate)
                    if signal >= stop_threshold:
                        stop_reason = "threshold_reached"
                        self.get_logger().info(f"✅ Reached {stop_threshold*100}%")
                        break
                    if (signal > best_eval) if maximize else (signal < best_eval):
                        best_positions, best_eval = candidate, signal
                        improvement_found = True
                    else:
                        if eval_count >= max_iterations:
                            break
                        candidate = {j: np.clip(best_positions[j] - step_size, 0.0, 1.0) for j in joint_names}
                        signal = measure_and_record(candidate)
                        if signal >= stop_threshold:
                            stop_reason = "threshold_reached"
                            self.get_logger().info(f"✅ Reached {stop_threshold*100}%")
                            break
                        if (signal > best_eval) if maximize else (signal < best_eval):
                            best_positions, best_eval = candidate, signal
                            improvement_found = True

                    if not improvement_found and adaptive_step:
                        step_size = max(step_size * step_decrease_factor, min_step)
                    elif improvement_found and adaptive_step:
                        step_size = min(step_size * step_increase_factor, max_step)

                if step_size < min_step and not threshold_reached:
                    stop_reason = "step_too_small"
                    self.get_logger().info(f"Step size {step_size:.5f} below minimum")
                    break

                alignment_controller_mgr.set_joint_states_from_mapped(best_positions)
                sweep_idx += 1
                if sweep_idx % 10 == 0:
                    self.get_logger().info(f"Sweep {sweep_idx}: best={best_eval:.6f}, evals={eval_count}/{max_iterations}")

        # --- Finalise ---
        alignment_controller_mgr.set_joint_states_from_mapped(best_positions)
        self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=movement_time)

        final_signal = best_eval
        final_dist_perc = dist_perc_hist[-1] if dist_perc_hist else 0.0
        final_angle_perc = angle_perc_hist[-1] if angle_perc_hist else 0.0

        if run_folder is not None:
            self.get_logger().info(f"📁 Saving results to {run_folder}")

            n = len(signal_hist)
            if len(eta_lat_hist) < n:
                eta_lat_hist.extend([0.0] * (n - len(eta_lat_hist)))
            if len(eta_ang_hist) < n:
                eta_ang_hist.extend([0.0] * (n - len(eta_ang_hist)))
            if len(eta_clean_hist) < n:
                eta_clean_hist.extend([0.0] * (n - len(eta_clean_hist)))
            if len(dist_perc_hist) < n:
                dist_perc_hist.extend([0.0] * (n - len(dist_perc_hist)))
            if len(angle_perc_hist) < n:
                angle_perc_hist.extend([0.0] * (n - len(angle_perc_hist)))
            if len(joint_hist) < n:
                last_joint = joint_hist[-1] if joint_hist else list(best_positions.values())
                joint_hist.extend([last_joint] * (n - len(joint_hist)))

            end_time = time.time()

            try:
                self.save_run_results(
                    run_folder, signal_hist, eta_lat_hist, eta_ang_hist, eta_clean_hist,
                    dist_perc_hist, angle_perc_hist, joint_hist, joint_names,
                    self.current_algorithm_name, metadata, start_time, end_time,
                    final_signal, final_dist_perc, final_angle_perc, len(signal_hist),  # <-- now len(signal_hist)
                    params={
                        'max_iterations': max_iterations,
                        'step_size': step_size,
                        'stop_threshold': stop_threshold,
                        'maximize': maximize,
                        'coordinate_wise': coordinate_wise,
                        'adaptive_step': adaptive_step,
                        'step_increase_factor': step_increase_factor,
                        'step_decrease_factor': step_decrease_factor,
                        'min_step': min_step,
                        'max_step': max_step,
                        'stop_reason': stop_reason,
                        'total_evaluations': eval_count,
                        'algorithm': 'hill_climb'
                    }
                )
                self.get_logger().info("✅ Results saved successfully.")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to save results: {e}", exc_info=True)
        else:
            self.get_logger().warn("No run_folder provided, skipping plot and save.")

        return OptimizationResult(best_positions, eval_count, final_signal)

    def nelder_mead_optimization(self,
                                alignment_controller_mgr,
                                goal_handle,
                                maxiter: int = 200,
                                xatol: float = 1e-5,
                                fatol: float = 1e-5,
                                disp: bool = True,
                                stop_threshold: float = 0.99,
                                metrics_reader: MetricsReader = None,
                                metadata: dict = None,
                                run_folder: Path = None,
                                start_time: float = None,
                                **kwargs) -> OptimizationResult:

        self.current_algorithm_name = "Nelder-Mead"

        x0_dict = alignment_controller_mgr.get_all_joint_states_as_mapped()
        x0_joint_names = list(x0_dict.keys())
        x0_joint_values = np.array(list(x0_dict.values()), dtype=float)

        # ---- HARDCORED PRE-FLIGHT CHECK ----
        need_fallback = False
        for j in x0_joint_names:
            if j not in self._current_joint_state_positions:
                continue
            if abs(self._current_joint_state_positions[j]) > 1e-4:
                need_fallback = True
                self.get_logger().warn(f"Joint {j} physical position {self._current_joint_state_positions[j]:.6f} not at center (0.0)")
                break

        if need_fallback:
            self.get_logger().warn("⚠️ Joints physically not at center. Forcing each joint to 0.5 with physical verification...")
            current_norm = alignment_controller_mgr.get_all_joint_states_as_mapped()
            for j in x0_joint_names:
                current_norm[j] = 0.5
                alignment_controller_mgr.set_joint_states_from_mapped(current_norm)
                self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=0.05)
                time.sleep(0.05)
                if j in self._current_joint_state_positions and abs(self._current_joint_state_positions[j]) > 1e-4:
                    self.get_logger().error(f"❌ Joint {j} still not at center after forced move! Physical: {self._current_joint_state_positions[j]:.6f}")
                    self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=0.2)
                    time.sleep(0.1)
            # Re‑fetch after fallback
            x0_dict = alignment_controller_mgr.get_all_joint_states_as_mapped()
            x0_joint_names = list(x0_dict.keys())
            x0_joint_values = np.array(list(x0_dict.values()), dtype=float)

        # History containers
        signal_hist = []
        eta_lat_hist = []
        eta_ang_hist = []
        eta_clean_hist = []
        dist_perc_hist = []
        angle_perc_hist = []
        joint_hist = []
        eval_count = 0
        stop_reason = "max_measurements"
        final_best_x = x0_joint_values.copy()
        final_best_f = 0.0

        class BreakOptimization(Exception):
            pass

        class StopEarly(Exception):
            pass

        def measure_signal(x):
            nonlocal eval_count
            eval_count += 1

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                raise BreakOptimization("Cancelled during Nelder-Mead measurement")

            x = np.clip(x, 0, 1)
            alignment_controller_mgr.set_joint_states_from_mapped(dict(zip(x0_joint_names, x)))
            self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=0.01)
            time.sleep(0.005)

            joint_hist.append(list(alignment_controller_mgr.get_all_joint_states_as_mapped().values()))

            if metrics_reader is not None:
                metrics = metrics_reader.get_metrics()
                signal = metrics['signal']
                dist_perc, angle_perc = self.compute_dynamic_percentages(
                    metrics['distance'], metrics['angle_deviation'], x0_joint_names)
                eta_lat = metrics['eta_lateral']
                eta_ang = metrics['eta_angular']
                eta_clean = metrics['eta_clean']
            else:
                signal = self.optimization_value_manager.get_eval_value(wait_for_update_sec=5, check_for_new_values=True)
                dist_perc = angle_perc = eta_lat = eta_ang = eta_clean = 0.0

            signal_hist.append(signal)
            eta_lat_hist.append(eta_lat)
            eta_ang_hist.append(eta_ang)
            eta_clean_hist.append(eta_clean)
            dist_perc_hist.append(dist_perc)
            angle_perc_hist.append(angle_perc)

            self.get_logger().debug(f"NM eval #{eval_count}: signal={signal:.6f}")

            if signal >= stop_threshold:
                self.get_logger().info(f"✅ Reached {stop_threshold*100}% at eval #{eval_count}")
                raise StopEarly("Threshold reached.")

            return signal

        def objective(x):
            return -measure_signal(x)

        try:
            initial_signal = measure_signal(x0_joint_values)

            n = len(x0_joint_values)
            simplex_step = 0.05
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0_joint_values.copy()
            for i in range(n):
                vertex = x0_joint_values.copy()
                vertex[i] += simplex_step
                vertex[i] = np.clip(vertex[i], 0.0, 1.0)
                simplex[i + 1] = vertex

            f_values = np.array([objective(vertex) for vertex in simplex], dtype=float)

            if eval_count >= maxiter:
                stop_reason = "max_measurements"
                self.get_logger().info(f"Reached max measurements ({maxiter}) during simplex init.")
                raise BreakOptimization()

            best_x = simplex[0].copy()
            best_f = f_values[0]

            iteration = 0
            while eval_count < maxiter:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    raise BreakOptimization("Cancelled during Nelder-Mead iteration")

                # Sort simplex
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]

                best = simplex[0]
                second_worst = simplex[-2]
                worst = simplex[-1]
                f_best = f_values[0]
                f_second_worst = f_values[-2]
                f_worst = f_values[-1]

                simplex_size = np.max(np.linalg.norm(simplex[1:] - best, axis=1))
                function_spread = np.max(np.abs(f_values[1:] - f_best))
                if simplex_size <= xatol and function_spread <= fatol:
                    stop_reason = "convergence"
                    self.get_logger().info(f"Converged: size={simplex_size:.6f}, spread={function_spread:.6f}")
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # 1. Reflection
                reflected = centroid + 1.0 * (centroid - worst)
                reflected = np.clip(reflected, 0.0, 1.0)
                f_reflected = objective(reflected)
                if eval_count >= maxiter:
                    stop_reason = "max_measurements"
                    break

                if f_best <= f_reflected < f_second_worst:
                    simplex[-1] = reflected
                    f_values[-1] = f_reflected
                    iteration += 1
                    continue

                # 2. Expansion
                if f_reflected < f_best:
                    expanded = centroid + 2.0 * (reflected - centroid)
                    expanded = np.clip(expanded, 0.0, 1.0)
                    f_expanded = objective(expanded)
                    if eval_count >= maxiter:
                        stop_reason = "max_measurements"
                        break
                    if f_expanded < f_reflected:
                        simplex[-1] = expanded
                        f_values[-1] = f_expanded
                    else:
                        simplex[-1] = reflected
                        f_values[-1] = f_reflected
                    iteration += 1
                    continue

                # 3. Contraction
                if f_second_worst <= f_reflected < f_worst:
                    contracted = centroid + 0.5 * (reflected - centroid)
                    contracted = np.clip(contracted, 0.0, 1.0)
                    f_contracted = objective(contracted)
                    if eval_count >= maxiter:
                        stop_reason = "max_measurements"
                        break
                    if f_contracted <= f_reflected:
                        simplex[-1] = contracted
                        f_values[-1] = f_contracted
                    else:
                        # outside contraction failed, fall through to shrink
                        pass
                else:
                    contracted = centroid + 0.5 * (worst - centroid)
                    contracted = np.clip(contracted, 0.0, 1.0)
                    f_contracted = objective(contracted)
                    if eval_count >= maxiter:
                        stop_reason = "max_measurements"
                        break
                    if f_contracted < f_worst:
                        simplex[-1] = contracted
                        f_values[-1] = f_contracted
                        iteration += 1
                        continue

                # 4. Shrink
                best_vertex = simplex[0].copy()
                for i in range(1, len(simplex)):
                    simplex[i] = best_vertex + 0.5 * (simplex[i] - best_vertex)
                    simplex[i] = np.clip(simplex[i], 0.0, 1.0)
                    f_values[i] = objective(simplex[i])
                    if eval_count >= maxiter:
                        stop_reason = "max_measurements"
                        break
                if eval_count >= maxiter:
                    break

                iteration += 1
                if iteration % 10 == 0:
                    self.get_logger().debug(f"NM iter {iteration}: best={-best_f:.6f}, evals={eval_count}/{maxiter}")

            # Final sort
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            final_best_x = np.clip(simplex[0], 0.0, 1.0)
            final_best_f = f_values[0]

        except StopEarly:
            stop_reason = "threshold_reached"
            if joint_hist:
                final_best_x = np.clip(joint_hist[-1], 0, 1)
            else:
                final_best_x = x0_joint_values
            final_best_f = -signal_hist[-1] if signal_hist else 0.0

        except BreakOptimization:
            if joint_hist:
                final_best_x = np.clip(joint_hist[-1], 0, 1)
            else:
                final_best_x = x0_joint_values
            final_best_f = -signal_hist[-1] if signal_hist else 0.0

        except Exception as e:
            self.get_logger().error(f"Optimization failed: {e}", exc_info=True)
            final_best_x = x0_joint_values
            final_best_f = -signal_hist[-1] if signal_hist else 0.0
            stop_reason = "error"

        best_result = dict(zip(x0_joint_names, final_best_x))
        final_signal = -final_best_f if final_best_f is not None else (signal_hist[-1] if signal_hist else 0.0)

        if run_folder is not None:
            self.get_logger().info(f"📁 Saving results to {run_folder}")

            if not signal_hist:
                self.get_logger().warn("No signal history recorded; cannot generate plot.")
                return OptimizationResult(best_result, eval_count, final_signal)

            n = len(signal_hist)
            if len(eta_lat_hist) < n:
                eta_lat_hist.extend([0.0] * (n - len(eta_lat_hist)))
            if len(eta_ang_hist) < n:
                eta_ang_hist.extend([0.0] * (n - len(eta_ang_hist)))
            if len(eta_clean_hist) < n:
                eta_clean_hist.extend([0.0] * (n - len(eta_clean_hist)))
            if len(dist_perc_hist) < n:
                dist_perc_hist.extend([0.0] * (n - len(dist_perc_hist)))
            if len(angle_perc_hist) < n:
                angle_perc_hist.extend([0.0] * (n - len(angle_perc_hist)))
            if len(joint_hist) < n:
                last_joint = joint_hist[-1] if joint_hist else [0.5] * len(x0_joint_names)
                joint_hist.extend([last_joint] * (n - len(joint_hist)))

            end_time = time.time()
            final_dist_perc = dist_perc_hist[-1] if dist_perc_hist else 0.0
            final_angle_perc = angle_perc_hist[-1] if angle_perc_hist else 0.0

            try:
                self.save_run_results(
                    run_folder, signal_hist, eta_lat_hist, eta_ang_hist, eta_clean_hist,
                    dist_perc_hist, angle_perc_hist, joint_hist, x0_joint_names,
                    self.current_algorithm_name, metadata, start_time, end_time,
                    final_signal, final_dist_perc, final_angle_perc, len(signal_hist),  # <-- now len(signal_hist)
                    params={
                        'maxiter': maxiter,
                        'xatol': xatol,
                        'fatol': fatol,
                        'disp': disp,
                        'stop_threshold': stop_threshold,
                        'stop_reason': stop_reason,
                        'total_evaluations': eval_count,
                        'algorithm': 'classical_nelder_mead',
                        'alpha': 1.0,
                        'gamma': 2.0,
                        'rho': 0.5,
                        'sigma': 0.5,
                        'simplex_step': 0.05
                    }
                )
                self.get_logger().info("✅ Results saved successfully.")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to save results: {e}", exc_info=True)
        else:
            self.get_logger().warn("No run_folder provided, skipping plot and save.")

        return OptimizationResult(best_result, eval_count, final_signal)

    def random_search_optimization(self,
                                    alignment_controller_mgr,
                                    goal_handle,
                                    max_iterations: int = 200,
                                    movement_time: float = 0.1,
                                    wait_for_settle: float = 0.15,
                                    use_global_search: bool = True,
                                    perturbation_scale: float = 0.3,
                                    stop_threshold: float = 0.99,
                                    metrics_reader: MetricsReader = None,
                                    metadata: dict = None,
                                    run_folder: Path = None,
                                    start_time: float = None,
                                    **kwargs) -> OptimizationResult:

        self.current_algorithm_name = "Random Search (Pure)"
        self.get_logger().info("=" * 80)
        self.get_logger().info("🎲 RANDOM SEARCH (Pure baseline)")
        self.get_logger().info("=" * 80)

        x0_dict = alignment_controller_mgr.get_all_joint_states_as_mapped()
        joint_names = list(x0_dict.keys())
        n = len(joint_names)

        # ---- HARDCORED PRE-FLIGHT CHECK: verify physical joints are at center ----
        need_fallback = False
        for j in joint_names:
            if j not in self._current_joint_state_positions:
                self.get_logger().warn(f"Joint {j} not in joint_states, assuming at center.")
                continue
            if abs(self._current_joint_state_positions[j]) > 1e-4:
                need_fallback = True
                self.get_logger().warn(f"Joint {j} physical position {self._current_joint_state_positions[j]:.6f} not at center (0.0)")
                break

        if need_fallback:
            self.get_logger().warn("⚠️ Joints physically not at center. Forcing each joint to 0.5 with physical verification...")
            current_norm = alignment_controller_mgr.get_all_joint_states_as_mapped()
            for j in joint_names:
                current_norm[j] = 0.5
                alignment_controller_mgr.set_joint_states_from_mapped(current_norm)
                self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=0.05)
                time.sleep(0.05)
                if j in self._current_joint_state_positions:
                    if abs(self._current_joint_state_positions[j]) > 1e-4:
                        self.get_logger().error(f"❌ Joint {j} still not at center after forced move! Physical: {self._current_joint_state_positions[j]:.6f}")
                        self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=0.2)
                        time.sleep(0.1)
                        if j in self._current_joint_state_positions and abs(self._current_joint_state_positions[j]) > 1e-4:
                            self.get_logger().error(f"❌ Joint {j} still off-center after retry. Continuing with caution.")
            # Re‑fetch after fallback
            x0_dict = alignment_controller_mgr.get_all_joint_states_as_mapped()
            joint_names = list(x0_dict.keys())
            n = len(joint_names)

        signal_hist = []
        eta_lat_hist = []
        eta_ang_hist = []
        eta_clean_hist = []
        dist_perc_hist = []
        angle_perc_hist = []
        joint_hist = []
        eval_count = 0
        stop_reason = "max_iterations"

        def measure_signal(x):
            nonlocal eval_count
            eval_count += 1

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                raise BreakOptimization("Cancelled during Random Search measurement")

            x = np.clip(x, 0.0, 1.0)
            alignment_controller_mgr.set_joint_states_from_mapped(dict(zip(joint_names, x)))
            self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=movement_time)
            time.sleep(wait_for_settle)

            if metrics_reader is not None:
                metrics = metrics_reader.get_metrics()
                signal = metrics['signal']
                dist_perc, angle_perc = self.compute_dynamic_percentages(
                    metrics['distance'], metrics['angle_deviation'], joint_names)
                signal_hist.append(signal)
                eta_lat_hist.append(metrics['eta_lateral'])
                eta_ang_hist.append(metrics['eta_angular'])
                eta_clean_hist.append(metrics['eta_clean'])
                dist_perc_hist.append(dist_perc)
                angle_perc_hist.append(angle_perc)
                joint_hist.append(list(alignment_controller_mgr.get_all_joint_states_as_mapped().values()))
            else:
                signal = self.optimization_value_manager.get_eval_value(
                    wait_for_update_sec=3.0, check_for_new_values=True)
                signal_hist.append(signal)
                eta_lat_hist.append(0.0)
                eta_ang_hist.append(0.0)
                eta_clean_hist.append(0.0)
                dist_perc_hist.append(0.0)
                angle_perc_hist.append(0.0)
                joint_hist.append(list(x))

            return signal

        x_best = np.array([x0_dict[j] for j in joint_names], dtype=float)
        best_signal = measure_signal(x_best)
        self.get_logger().info(f"Initial signal: {best_signal:.6f}")

        for it in range(max_iterations - 1):
            if goal_handle.is_cancel_requested:
                stop_reason = "cancelled"
                raise BreakOptimization("Cancelled during Random Search loop")

            if best_signal >= stop_threshold:
                stop_reason = "threshold_reached"
                self.get_logger().info(f"✅ Reached {stop_threshold*100}% signal, stopping early.")
                break

            if use_global_search:
                x_candidate = np.random.uniform(0, 1, n)
            else:
                perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, n)
                x_candidate = np.clip(x_best + perturbation, 0.0, 1.0)

            candidate_signal = measure_signal(x_candidate)

            if candidate_signal > best_signal:
                self.get_logger().debug(f"Iter {it+1}: {best_signal:.6f} -> {candidate_signal:.6f}")
                x_best = x_candidate.copy()
                best_signal = candidate_signal

            if (it + 1) % 20 == 0:
                self.get_logger().info(f"Iter {it+1}/{max_iterations-1}: best = {best_signal:.6f}")

        alignment_controller_mgr.set_joint_states_from_mapped(dict(zip(joint_names, x_best)))
        self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=movement_time)

        final_signal = best_signal

        self.get_logger().info("=" * 80)
        self.get_logger().info("✅ RANDOM SEARCH COMPLETE!")
        self.get_logger().info(f"   Best signal: {final_signal:.6f}")
        self.get_logger().info(f"   Evaluations: {eval_count}")
        self.get_logger().info(f"   Stop reason: {stop_reason}")
        self.get_logger().info("=" * 80)

        best_result = dict(zip(joint_names, x_best))

        if run_folder is not None:
            self.get_logger().info(f"📁 Saving results to {run_folder}")

            if not signal_hist:
                self.get_logger().warn("No signal history recorded; cannot generate plot.")
                return OptimizationResult(best_result, eval_count, final_signal)

            n_hist = len(signal_hist)
            if len(eta_lat_hist) < n_hist:
                eta_lat_hist.extend([0.0] * (n_hist - len(eta_lat_hist)))
            if len(eta_ang_hist) < n_hist:
                eta_ang_hist.extend([0.0] * (n_hist - len(eta_ang_hist)))
            if len(eta_clean_hist) < n_hist:
                eta_clean_hist.extend([0.0] * (n_hist - len(eta_clean_hist)))
            if len(dist_perc_hist) < n_hist:
                dist_perc_hist.extend([0.0] * (n_hist - len(dist_perc_hist)))
            if len(angle_perc_hist) < n_hist:
                angle_perc_hist.extend([0.0] * (n_hist - len(angle_perc_hist)))
            if len(joint_hist) < n_hist:
                last_joint = joint_hist[-1] if joint_hist else list(x_best)
                joint_hist.extend([last_joint] * (n_hist - len(joint_hist)))

            end_time = time.time()
            final_dist_perc = dist_perc_hist[-1] if dist_perc_hist else 0.0
            final_angle_perc = angle_perc_hist[-1] if angle_perc_hist else 0.0

            try:
                self.save_run_results(
                    run_folder, signal_hist, eta_lat_hist, eta_ang_hist, eta_clean_hist,
                    dist_perc_hist, angle_perc_hist, joint_hist, joint_names,
                    self.current_algorithm_name, metadata, start_time, end_time,
                    final_signal, final_dist_perc, final_angle_perc, len(signal_hist)-1,
                    params={
                        'max_iterations': max_iterations,
                        'use_global_search': use_global_search,
                        'perturbation_scale': perturbation_scale,
                        'stop_threshold': stop_threshold,
                        'stop_reason': stop_reason,
                        'total_evaluations': eval_count,
                        'algorithm': 'random_search_pure'
                    }
                )
                self.get_logger().info("✅ Results saved successfully.")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to save results: {e}", exc_info=True)
        else:
            self.get_logger().warn("No run_folder provided, skipping plot and save.")

        return OptimizationResult(best_result, eval_count, final_signal)

    # ============================================================
    # SAVING (only CSV + metadata, no plots)
    # ============================================================
    def save_run_results(self, run_folder, signal_hist, eta_lat_hist, eta_ang_hist, eta_clean_hist,
                         dist_perc_hist, angle_perc_hist, joint_hist, joint_names,
                         algo_name, metadata, start_time, end_time, final_signal,
                         final_dist_perc, final_angle_perc, iterations, params=None):
        meta_copy = metadata.copy()
        if params:
            meta_copy['algorithm_params'] = params

        summary_path = run_folder / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Optimisation Summary\n")
            f.write("==================\n")
            f.write(f"Algorithm: {algo_name}\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Iterations: {iterations}\n")
            f.write(f"Time: {(end_time - start_time):.2f} s\n")
            f.write(f"\nFinal Results:\n")
            f.write(f"  Signal: {final_signal:.6f}\n")
            f.write(f"  Distance %: {final_dist_perc:.2f}%\n")
            f.write(f"  Angle %: {final_angle_perc:.2f}%\n")
            f.write(f"  Eta_lateral: {eta_lat_hist[-1] if eta_lat_hist else 0:.6f}\n")
            f.write(f"  Eta_angular: {eta_ang_hist[-1] if eta_ang_hist else 0:.6f}\n")
            f.write(f"  Eta_clean: {eta_clean_hist[-1] if eta_clean_hist else 0:.6f}\n")
            f.write(f"\nInitial misalignment:\n")
            f.write(f"  Seed: {metadata.get('seed_number', 'N/A')}\n")
            f.write(f"  Target power: {metadata.get('power_percent', 'N/A')}%\n")
            f.write(f"  Initial signal: {metadata.get('INITIAL_POWER_PERCENT', 0):.2f}%\n")
            f.write(f"  Distance: {metadata.get('initial_distance_um', 0):.2f} µm\n")
            f.write(f"  Angle: {metadata.get('initial_angle_deg', 0):.2f}°\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"  W0: {metadata.get('W0', 0):.2e} m\n")
            f.write(f"  LAMBDA: {metadata.get('LAMBDA', 0):.2e} m\n")
            f.write(f"  SEARCH_FACTOR: {metadata.get('SEARCH_FACTOR', 0)}\n")
            f.write(f"  MODE: {metadata.get('MODE', 'N/A')}\n")
            f.write(f"  Enabled joints: {[j for j, en in metadata.get('ENABLED_JOINTS', {}).items() if en]}\n")
            if params:
                f.write(f"\nAlgorithm parameters:\n")
                for k, v in params.items():
                    f.write(f"  {k}: {v}\n")

        csv_path = run_folder / "history.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['iteration', 'signal', 'eta_lateral', 'eta_angular', 'eta_clean',
                      'dist_perc', 'angle_perc']
            for j in joint_names:
                header.append(j)
            writer.writerow(header)
            n = len(signal_hist)
            for i in range(n):
                row = [i, signal_hist[i],
                       eta_lat_hist[i] if i < len(eta_lat_hist) else 0,
                       eta_ang_hist[i] if i < len(eta_ang_hist) else 0,
                       eta_clean_hist[i] if i < len(eta_clean_hist) else 0,
                       dist_perc_hist[i] if i < len(dist_perc_hist) else 0,
                       angle_perc_hist[i] if i < len(angle_perc_hist) else 0]
                if i < len(joint_hist):
                    row.extend(joint_hist[i])
                else:
                    row.extend([0] * len(joint_names))
                writer.writerow(row)

        metadata_path = run_folder / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(meta_copy, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))

        self.get_logger().info(f"✅ Results saved to {run_folder}")

    # ============================================================
    # TRAJECTORY METHODS (unchanged)
    # ============================================================
    def move_joints_to_current_stat_sync(self, alignment_controller_mgr, move_time: float = 1.0):
        controllers = alignment_controller_mgr.get_controllers()
        for ctrl in controllers:
            joint_values_dict = ctrl.get_joint_states()
            ok = self.send_trajectory_from_dict_sync(
                controller_name=ctrl.controller_name,
                joint_state_dict=joint_values_dict,
                duration=move_time,
            )
            if not ok:
                self.get_logger().error(f"Failed to move controller '{ctrl.controller_name}'.")

    def send_trajectory_from_dict_sync(self, controller_name: str, joint_state_dict: dict[str, float], duration: float = 1.0) -> bool:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self.send_trajectory_from_dict(controller_name, joint_state_dict, duration),
                loop
            )
            return future.result()
        else:
            return asyncio.run(self.send_trajectory_from_dict(controller_name, joint_state_dict, duration))

    async def send_trajectory_from_dict(self, controller_name: str, joint_state_dict: dict[str, float], duration: float = 1.0) -> bool:
        if controller_name not in self._action_clients:
            self._action_clients[controller_name] = ActionClient(
                self, FollowJointTrajectory, f'/{controller_name}/follow_joint_trajectory',
                callback_group=self.cb_reentrant
            )
        client = self._action_clients[controller_name]
        if not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f"Action server not available: {controller_name}")
            return False
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = list(joint_state_dict.keys())
        point = JointTrajectoryPoint()
        point.positions = list(joint_state_dict.values())
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal_msg.trajectory.points.append(point)
        goal_handle_future = client.send_goal_async(goal_msg)
        goal_handle = await goal_handle_future
        if not goal_handle.accepted:
            self.get_logger().error(f"Trajectory goal for {controller_name} was rejected.")
            return False
        result_future = goal_handle.get_result_async()
        result = await result_future
        if result.status != 4:
            self.get_logger().error(f"Trajectory for {controller_name} failed with status {result.status}")
            return False
        return True

    def get_current_joint_state(self, joint_name: str) -> float:
        current = self._current_joint_state_positions.get(joint_name, None)
        if current is None:
            raise ValueError(f"Joint '{joint_name}' state not available.")
        return current

    def joint_state_callback(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            self._current_joint_state_positions[name] = position

# ============================================================
# MAIN
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = ActiveAlignmentServer()
    executor = MultiThreadedExecutor(num_threads=4)
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()