#!/usr/bin/env python3
import csv
import random
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float32
from active_alignment_interfaces.action import ActiveAlignment, ActiveAlign  # <-- your action interface
from active_alignment_interfaces.msg import AlignTopic, JointAlign, JointAlignmentResult  # <-- your action interface
from controller_manager_msgs.srv import ListControllers
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
import time
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from builtin_interfaces.msg import Duration as MsgDuration
from rclpy.action import ActionClient
from active_alignment_skills.py_modules.JointLimitFetcher import JointLimitFetcher
from active_alignment_skills.py_modules.AlignmentControllerManager import AlignmentControllerManager, ControllerJoint, AlignmentController
from active_alignment_skills.py_modules.OptimizationValueManager import OptimizationValueManager

import numpy as np
from scipy.optimize import minimize
import asyncio
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from urdf_parser_py.urdf import URDF, Joint as URDFJoint
import datetime
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from urdf_parser_py.urdf import URDF, Joint as URDFJoint
import matplotlib.pyplot as plt
import os
from functools import partial

class BreakOptimization(Exception):
    def __init__(self, message="Optimization process was interrupted."):
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

class ActiveAlignmentServer(Node):
    def __init__(self):
        super().__init__('active_alignment_server')
        # --- Callback groups ---
        self.cb_reentrant = ReentrantCallbackGroup()
        self.cb_subscriptions = MutuallyExclusiveCallbackGroup()

        self._joint_limit_fetcher = JointLimitFetcher(self)

        # Persistent ActionClients
        self._action_clients: dict[str, ActionClient] = {}

        # self._action_server = ActionServer(
        #     self,
        #     ActiveAlign,
        #     'exec_active_alignment',
        #     execute_callback=self.execute_callback,
        #     goal_callback=self.goal_callback,
        #     cancel_callback=self.cancel_callback
        # )
        
        self._action_server_hill_climb = ActionServer(
            self,
            ActiveAlign,
            f'{self.get_name()}/exec_active_alignment_hill_climb',
            #execute_callback=partial(self.execute_callback, self.hill_climb_optimization),
            execute_callback=self.make_execute_callback(self.hill_climb_optimization),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        self._action_server_nelder = ActionServer(
            self,
            ActiveAlign,
            f'{self.get_name()}/exec_active_alignment_nelder_mead',
            #execute_callback=partial(self.execute_callback, self.nelder_mead_optimization),
            execute_callback=self.make_execute_callback(self.nelder_mead_optimization),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.cli = self.create_client(ListControllers, '/controller_manager/list_controllers')
        self._current_joint_state_positions = {}

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.optimization_value_manager = OptimizationValueManager(self)

        self.get_logger().info("Active Alignment Server is ready.")

    def make_execute_callback(self, optimization_func):
        async def _callback(goal_handle):
            return await self.execute_callback(goal_handle, optimization_func)
        return _callback


    def goal_callback(self, goal_request: ActiveAlign.Goal):
        self.get_logger().info(f"Received goal: {str(goal_request)}")
        # Accept all goals
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Cancel request received")
        return CancelResponse.ACCEPT

    async def get_controller_joints(self, controller_name: str) -> list[str]:
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Controller manager service not available")
            return []

        req = ListControllers.Request()
        future = self.cli.call_async(req)
        await future
        resp: ListControllers.Response = future.result()

        for ctrl in resp.controller:
            if ctrl.name == controller_name:
                # Use getattr in case field is missing
                interfaces = ctrl.claimed_interfaces
                clean_joint_names = [name.replace('/position', '') for name in interfaces]
                return clean_joint_names
        return []

    async def get_controller_type(self, controller_name: str) -> str:
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Controller manager service not available")
            return ""

        req = ListControllers.Request()
        future = self.cli.call_async(req)
        await future
        resp: ListControllers.Response = future.result()

        for ctrl in resp.controller:
            if ctrl.name == controller_name:
                # Use getattr in case field is missing
                controller_type = ctrl.type
                return controller_type
        return ""

    async def execute_callback(self, goal_handle, optimization_method):
        try:
            goal: ActiveAlign.Goal = goal_handle.request
            alignment_topics = goal.alignment_topics
            
            if len(alignment_topics) == 0:
                self.get_logger().error("No alignment topics provided in goal.")
                result = ActiveAlign.Result()
                result.success = False
                goal_handle.abort()
                return result
        
            alignment_controller_mgr = await self.init_controller_manager(goal)
            
            self.optimization_value_manager.init(alignment_topics)

            # test = await self.hill_climb_optimization(alignment_controller_mgr,
            #                                            goal_handle=goal_handle,
            #                                            max_iterations=300, 
            #                                            step_size=0.05)

            # optim_result = self.nelder_mead_optimization(alignment_controller_mgr,
            #                                             goal_handle=goal_handle)
            optim_result = optimization_method(alignment_controller_mgr,
                                                goal_handle=goal_handle)

            self.optimization_value_manager.clear()

            result = ActiveAlign.Result()
            optim_result.to_result_msg(result)
            result.success = True
            goal_handle.succeed()

        except BreakOptimization:
            self.get_logger().info("Optimization was interrupted.")
            result = ActiveAlign.Result()
            result.success = False
            goal_handle.abort()
            return result
        
        except Exception as e:
            self.get_logger().error(f"Exception in execute_callback: {e}")
            result = ActiveAlign.Result()
            result.success = False
            goal_handle.abort()

        finally:
            self.optimization_value_manager.clear()
            return result







    async def init_controller_manager(self, alignment_goal: ActiveAlign.Goal) -> AlignmentControllerManager:

        alignment_controller_mgr = AlignmentControllerManager()

        active_joints = alignment_goal.active_joints
        
        if len(active_joints) == 0:
            raise ValueError("No active joints provided in goal.")

        involved_controllers = set()

        for joint_align in active_joints:
            joint_align: JointAlign
            _joint_name = joint_align.joint_name
            _joint_info = await self._joint_limit_fetcher.get_joint_info(_joint_name)
            _controller_for_joint = _joint_info["controller_name"]

            if not _joint_info["controller_type"] == "joint_trajectory_controller/JointTrajectoryController":
                raise ValueError(f"Unsupported controller type for joint '{_joint_name}': {_joint_info['controller_type']}")
            
            _current_joint_value = self.get_current_joint_state(_joint_name)

            _c_joint = ControllerJoint(is_active_joint = True,
                                        joint_align_msg = joint_align, 
                                        initial_joint_value =_current_joint_value)
            
            _c_joint.initialize_joint_limits(upper_limit=_joint_info["upper"], lower_limit=_joint_info["lower"])
            involved_controllers.add(_controller_for_joint)
            alignment_controller_mgr.add_joint_to_controller(controller_name=_controller_for_joint, joint=_c_joint)
        
        # add the static joints to the controllers
        for ctrl in involved_controllers:
            _joints = await self._joint_limit_fetcher.get_joints_for_controller(ctrl)

            for j in _joints:
                if alignment_controller_mgr.is_joint_in_any_controller(j):
                    continue  # already added as active joint
                _msg = JointAlign()
                _msg.joint_name = j
                _current_joint_value = self.get_current_joint_state(j)
                _c_joint = ControllerJoint(is_active_joint = False,
                                            initial_joint_value = _current_joint_value,
                                            joint_align_msg = _msg)
                alignment_controller_mgr.add_joint_to_controller(controller_name=ctrl, joint=_c_joint)

        debug_info = alignment_controller_mgr.get_debug_info()  
        self.get_logger().info(f"Alignment Controller Manager Debug Info: {debug_info}")
        return alignment_controller_mgr


    async def move_joints_to_current_stat(self, 
                                          alignment_controller_mgr: AlignmentControllerManager,
                                          move_time : float = 1.0):
        """
        Move all controllers to their current joint states specified in the alignment controller manager.
        """
        controllers: list[AlignmentController] = alignment_controller_mgr.get_controllers()

        for ctrl in controllers:
            ctrl:AlignmentController
            joint_values_dict = ctrl.get_joint_states()

            await self.send_trajectory_from_dict(controller_name=ctrl.controller_name, 
                                                 joint_state_dict=joint_values_dict,
                                                 duration=move_time)
            
            #self.get_logger().info(f"Moved controller '{ctrl.controller_name}' to current joint states.")
            



    def nelder_mead_optimization(self, alignment_controller_mgr: AlignmentControllerManager,
                                       goal_handle
                                        ) -> OptimizationResult:        
        """
        Optimize joint positions using the Nelder-Mead algorithm to maximize the eval value.
        """

        pkg_share = get_package_share_directory('active_alignment_skills')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        x0_dict = alignment_controller_mgr.get_all_joint_states_as_mapped()
        x0_joint_names = list(x0_dict.keys())
        x0_joint_values = list(x0_dict.values())
        eval_history = []

        def measure_signal(x):

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                raise BreakOptimization()
        
            x = np.clip(x, 0, 1)
            alignment_controller_mgr.set_joint_states_from_mapped(dict(zip(x0_joint_names, x)))
            self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=0.1)
            time.sleep(0.2)  # wait for eval to update

            best_eval = self.optimization_value_manager.get_eval_value(wait_for_update_sec=5,
                                                                       check_for_new_values=True)
            
            return best_eval

        # ------------------------------------------------------------
        # Synchronous wrapper (used inside SciPy optimizer threads)
        def objective(x):
            signal = measure_signal(x)

            # Log results for analysis
            log_path = f"{pkg_share}/robot_optimization_log_{timestamp}.csv"
            eval_history.append(signal)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(list(x) + [signal])

            return signal  # we negate to make it a minimization problem


        # ------------------------------------------------------------
        # Run optimizer in thread so ROS stays responsive
        # ------------------------------------------------------------

        result = minimize(
            objective,
            x0_joint_values,
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 1e-5, "fatol": 1e-5, "disp": True},
        )

        self.plot_optimization_history(eval_history, "active_alignment_optimization")

        best_x = np.clip(result.x, 0, 1)
        best_result = dict(zip(x0_joint_names, best_x))
        self.get_logger().info(f"Optimization finished. Best result: {best_result}")
        return OptimizationResult(best_result, result.nit, result.fun)



    def hill_climb_optimization(self, 
            alignment_controller_mgr: AlignmentControllerManager,
            goal_handle, 
            max_iterations: int = 20, 
            step_size: float = 0.1,
            ) -> list[float]:
        """
        Optimize joint positions to minimize eval value using a simple stochastic hill-climb algorithm.
        """

        adaptive_step_size = False
        movement_time = 0.1
        #best_positions = alignment_controller_mgr.get_all_joint_states()
        best_positions_mapped = alignment_controller_mgr.get_all_joint_states_as_mapped()

        self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=movement_time)

        # Wait for initial valid eval value
        best_eval = self.optimization_value_manager.get_eval_value(wait_for_update_sec=5,
                                                                   check_for_new_values=True)
        
        eval_history = [best_eval]  # store history

        no_improve_count = 0

        for it in range(max_iterations):
            
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled, exiting optimization loop.")
                goal_handle.canceled()
                return alignment_controller_mgr.get_all_joint_states()

            # get dict of all normalized joint positions
            normalized_joints = alignment_controller_mgr.get_all_joint_states_as_mapped()

            # Create candidate by random perturbation
            candidate_positions_mapped = {
                j: np.clip(v + np.random.uniform(-step_size, step_size), 0.0, 1.0)
                for j, v in normalized_joints.items()
            }
            
            alignment_controller_mgr.set_joint_states_from_mapped(candidate_positions_mapped)

            self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=movement_time)

            time.sleep(0.5)

            current_eval = self.optimization_value_manager.get_eval_value(wait_for_update_sec=5,
                                                                          check_for_new_values=True)

            eval_history.append(current_eval)

            self.get_logger().info(f"Iteration {it+1}/{max_iterations}: eval={current_eval}, step_size={step_size:.5f}")

                # Compare and update best result
            if current_eval < best_eval:
                best_eval = current_eval
                best_positions_mapped = candidate_positions_mapped.copy()
                self.get_logger().info(f"✅ Improvement found: eval={best_eval:.6f}")
                if adaptive_step_size:
                    step_size = min(step_size * 1.2, 0.3)  # be more exploratory
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 5 and adaptive_step_size:
                    step_size = max(step_size * 0.5, 0.01)  # fine-tune around best
                self.get_logger().info("No improvement, reverting to previous best positions.")
                alignment_controller_mgr.set_joint_states_from_mapped(best_positions_mapped)
                #await self.move_joints_to_current_stat(alignment_controller_mgr)

        # self.get_logger().info(f"Optimization done. Best eval value: {best_eval}")

        # # Plot history
        self.plot_optimization_history(eval_history, "active_alignment_optimization")
        self.move_joints_to_current_stat_sync(alignment_controller_mgr, move_time=movement_time)

        best_positions = alignment_controller_mgr.get_all_joint_states()

        result = OptimizationResult(best_positions, len(eval_history)-1, best_eval)

        return result



    def send_trajectory_from_dict_sync(self, controller_name: str, joint_state_dict: dict[str, float], duration: float = 1.0) -> bool:
        """
        Synchronous wrapper around async send_trajectory_from_dict().
        Can be called safely from normal (non-async) code.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Submit coroutine to the running loop (ROS async context)
            future = asyncio.run_coroutine_threadsafe(
                self.send_trajectory_from_dict(controller_name, joint_state_dict, duration),
                loop
            )
            return future.result()
        else:
            # No event loop running (e.g. pure sync mode) -> create one temporarily
            return asyncio.run(self.send_trajectory_from_dict(controller_name, joint_state_dict, duration))


    def move_joints_to_current_stat_sync(self, alignment_controller_mgr, move_time: float = 1.0):
        """
        Synchronous version of move_joints_to_current_stat().
        Calls send_trajectory_from_dict_sync() for each controller.
        """
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
            else:
                #self.get_logger().info(f"Moved controller '{ctrl.controller_name}' to current joint states.")
                pass


    def plot_optimization_history(self, eval_history: list[float], controller_name: str):
        if not eval_history:
            return

        plt.figure()
        plt.plot(eval_history, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Eval Value")
        plt.title(f"Optimization History: {controller_name}")
        plt.grid(True)

        # Get package share directory
        time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Get package share directory
        pkg_share = get_package_share_directory('active_alignment_skills')
        filename = os.path.join(pkg_share, f"optimization_history_{controller_name}_{time_stamp}.png")

        # Save plot
        plt.savefig(filename)
        plt.close()
        self.get_logger().info(f"Optimization history saved to {filename}")


    # -------------------- Trajectory sending --------------------
    async def send_trajectory(self, controller_name: str, 
                              joint_names: list[str], 
                              positions: list[float], 
                              duration: float = 1.0):
        if controller_name not in self._action_clients:
            self._action_clients[controller_name] = ActionClient(
                self,
                FollowJointTrajectory,
                f'/{controller_name}/follow_joint_trajectory',
                callback_group=self.cb_reentrant
            )
        client = self._action_clients[controller_name]

        if not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f"Action server not available: {controller_name}")
            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal_msg.trajectory.points.append(point)

        goal_handle_future = client.send_goal_async(goal_msg)
        goal_handle = await goal_handle_future

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        result = await result_future

        if result.status != 4:  # SUCCEEDED
            self.get_logger().error(f"Goal failed with status {result.status}")
            return False

        return True
    


    async def send_trajectory_from_dict(self, 
                              controller_name: str, 
                              joint_state_dict: dict[str, float], 
                              duration: float = 1.0) -> bool:
        """
        Send a trajectory goal to a controller using a joint state dictionary.

        Args:
            controller_name (str): Name of the controller action.
            joint_state_dict (dict[str, float]): Mapping of {joint_name: position}.
            duration (float): Movement duration in seconds.

        Returns:
            bool: True if succeeded, False otherwise.
        """
        # Create or reuse ActionClient
        if controller_name not in self._action_clients:
            self._action_clients[controller_name] = ActionClient(
                self,
                FollowJointTrajectory,
                f'/{controller_name}/follow_joint_trajectory',
                callback_group=self.cb_reentrant
            )
        client = self._action_clients[controller_name]

        # Wait for action server
        if not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f"Action server not available: {controller_name}")
            return False

        # Prepare goal message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = list(joint_state_dict.keys())

        point = JointTrajectoryPoint()
        point.positions = list(joint_state_dict.values())
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)

        goal_msg.trajectory.points.append(point)

        # Send goal
        goal_handle_future = client.send_goal_async(goal_msg)
        goal_handle = await goal_handle_future

        if not goal_handle.accepted:
            self.get_logger().error(f"Trajectory goal for {controller_name} was rejected.")
            return False

        # Wait for result
        result_future = goal_handle.get_result_async()
        result = await result_future

        if result.status != 4:  # 4 == SUCCEEDED
            self.get_logger().error(f"Trajectory for {controller_name} failed with status {result.status}")
            return False

        #self.get_logger().info(f"✅ Trajectory sent successfully to {controller_name}")
        return True

            
    def get_current_joint_state(self,joint_name:str)->float:
        """
        Get the current joint state of the robot.
        
        Args:
            joint_name (str): Name of the joint.
        
        Returns:
            float: Current joint state.
        """
        current = self._current_joint_state_positions.get(joint_name, None)
        if current is None:
            raise ValueError(f"Joint '{joint_name}' state not available.")
        
        return current

    def joint_state_callback(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            self._current_joint_state_positions[name] = position

    def float_to_ros_duration(self, time_float):

        secs = int(time_float)
        nsecs = int((time_float - secs) * 1e9)
        return MsgDuration(sec=secs, nanosec=nsecs)


    def wait_for_joints_reached(self, 
                                joint_names:list[str], 
                                target_joint_values:list[float], 
                                tolerance:list[float],
                                timeout:float = 5.0)->bool:
        
        """
        Wait for the robot to reach the desired joint positions.
        Args:
            joint_names (list[str]): List of joint names.   
            joint_values (list[float]): List of joint values.
            tolerance (list[float]): List of tolerances for each joint.
            timeout (float): Timeout in seconds.
        Returns:

            bool: True if the robot reached the desired joint positions, False otherwise.
        """
        start_time = time.time()
        while True:
            current_joint_positions = [self.get_current_joint_state(name) for name in joint_names]
            if None in current_joint_positions:
                self.get_logger().error("Joint state not available.")
                return False
            
            reached = True
            for i, (current_position, target_position, tol) in enumerate(zip(current_joint_positions, target_joint_values, tolerance)):
                if abs(current_position - target_position) > tol:
                    reached = False
                    break
            
            if reached:
                self.get_logger().info("Robot reached the desired joint positions.")
                time.sleep(0.5)
                return True
            
            if time.time() - start_time > timeout:
                self.get_logger().error("Timeout waiting for joint positions.")
                return False
            
def main(args=None):
    rclpy.init(args=args)
    node = ActiveAlignmentServer()

    # Multi-threaded executor (4 threads)
    executor = MultiThreadedExecutor(num_threads=4)
    rclpy.spin(node, executor=executor)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
