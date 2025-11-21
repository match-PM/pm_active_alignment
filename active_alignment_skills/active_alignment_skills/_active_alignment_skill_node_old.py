#!/usr/bin/env python3
import random
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float32
from active_alignment_interfaces.action import ActiveAlignment  # <-- your action interface

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
import numpy as np
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

class ActiveAlignmentServer(Node):
    def __init__(self):
        super().__init__('active_alignment_server')
        # --- Callback groups ---
        self.cb_reentrant = ReentrantCallbackGroup()
        self.cb_subscriptions = MutuallyExclusiveCallbackGroup()

        self._joint_limit_fetcher = JointLimitFetcher(self)

        # Persistent ActionClients
        self._action_clients: dict[str, ActionClient] = {}

        self._action_server = ActionServer(
            self,
            ActiveAlignment,
            'exec_active_alignment',
            execute_callback=self.execute_callback,
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

        self.eval_value_pub = self.create_publisher(Float32, 'active_alignment/eval_value', 10)
        self.eval_value_timer = self.create_timer(0.1, self.publish_eval_value)


        # Store subscribers and latest values
        self.subscription_list = []
        self.eval_dict = {}
        self.get_logger().info("Active Alignment Server is ready.")

    def publish_eval_value(self):
        valid_values = [v for v in self.eval_dict.values() if v is not None]
        if not valid_values:
            return  # don’t publish yet

        eval_value = sum(valid_values) / len(valid_values)
        msg = Float32()
        msg.data = eval_value
        self.eval_value_pub.publish(msg)
        self.get_logger().info(f"Published eval value: {eval_value}")

    def get_eval_value(self) -> float:
        valid_values = [v for v in self.eval_dict.values() if v is not None]
        if not valid_values:
            return None
        return sum(valid_values) / len(valid_values)

    def goal_callback(self, goal_request: ActiveAlignment.Goal):
        self.get_logger().info(f"Received goal: controllers={goal_request.controller_names}, topics={goal_request.topic_names_float}")
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

    async def execute_callback(self, goal_handle):
        goal: ActiveAlignment.Goal = goal_handle.request
        controllers = goal.controller_names
        topics = goal.topic_names_float

        self.create_subscriptions(topics)

        for ctrl in controllers:
            joints = await self.get_controller_joints(ctrl)
            controller_type = await self.get_controller_type(ctrl)
            self.get_logger().info(f"Optimizing controller '{ctrl}' ({controller_type}) for joints: {joints}")

            if not joints:
                self.get_logger().error(f"No joints found for controller: {ctrl}")
                result = ActiveAlignment.Result()
                result.success = False
                goal_handle.abort()
                return result

            await self._joint_limit_fetcher.get_joint_limits(joints)

            best_positions = await self.optimize_joint_positions(ctrl, joints, max_iterations=100, step_size=0.05)

            # move to best positions
            await self.send_trajectory(ctrl, joints, best_positions, duration=1.0)
            
            self.get_logger().info(f"Best joint positions for '{ctrl}': {best_positions}")

        self.destroy_subscriptions()

        result = ActiveAlignment.Result()
        result.success = True
        goal_handle.succeed()
        return result









    async def optimize_joint_positions(self, controller_name: str, 
                                    joint_names: list[str], 
                                    max_iterations: int = 20, 
                                    step_size: float = 0.1) -> list[float]:
        """
        Optimize joint positions to minimize eval value using a simple stochastic hill-climb algorithm.
        """
        limits = await self._joint_limit_fetcher.get_joint_limits(joint_names)
        if not limits:
            self.get_logger().error("No joint limits found, cannot optimize.")
            return []

        current_positions = []
        for j in joint_names:
            try:
                current_positions.append(self.get_current_joint_state(j))
            except ValueError:
                low, high = limits[j]
                current_positions.append((low + high) / 2.0)

        best_positions = current_positions.copy()

        # Wait for initial valid eval value
        best_eval = None
        for _ in range(50):  # wait up to 5s (0.1s steps)
            best_eval = self.get_eval_value()
            if best_eval is not None:
                break
            await asyncio.sleep(0.1)
        if best_eval is None:
            self.get_logger().error("No valid eval value received, aborting optimization.")
            return best_positions
        
        eval_history = [best_eval]  # ← store history

        for it in range(max_iterations):
            normalized = self._joint_limit_fetcher.map_joints_to_normalized(
                dict(zip(joint_names, best_positions))
            )
            candidate_norm = {
                j: np.clip(v + np.random.uniform(-step_size, step_size), 0.0, 1.0)
                for j, v in normalized.items()
            }
            candidate_positions = list(
                self._joint_limit_fetcher.map_normalized_to_joints(candidate_norm).values()
            )

            await self.send_trajectory(controller_name, joint_names, candidate_positions, duration=0.4)

            time.sleep(0.5)

            # Evaluate
            # Wait for valid eval value
            current_eval = None
            for _ in range(50):  # wait up to 5s
                current_eval = self.get_eval_value()
                if current_eval is not None:
                    break
                await asyncio.sleep(0.1)
            if current_eval is None:
                self.get_logger().warn("Eval value not received, skipping this iteration.")
                continue

            eval_history.append(current_eval)

            self.get_logger().info(f"Iteration {it+1}/{max_iterations}: eval={current_eval}")

            if current_eval < best_eval:
                best_eval = current_eval
                best_positions = candidate_positions
                self.get_logger().info(f"✅ Improvement found: eval={best_eval}")
            else:
                self.get_logger().info("No improvement, keeping previous positions.")

        self.get_logger().info(f"Optimization done. Best eval value: {best_eval}")

        # Plot history
        self.plot_optimization_history(eval_history, controller_name)

        return best_positions





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

    def create_subscriptions(self, topics: list[str]):
        self.eval_dict.clear()
        for topic in topics:
            self.eval_dict[topic] = None
            _sub = self.create_subscription(
                    Float32,
                    topic,
                    lambda msg, t=topic: self.topic_callback(msg, t),
                    10
                )
            self.subscription_list.append(_sub)
            self.get_logger().info(f"Subscribed to topic: {topic}")

    def destroy_subscriptions(self):
        for sub in self.subscription_list:
            self.destroy_subscription(sub)
            self.get_logger().info(f"Destroyed subscription.")
        self.subscription_list = []
        self.eval_dict.clear()

    def topic_callback(self, 
                       msg: Float32, 
                       topic: str):
        self.eval_dict[topic] = msg.data


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
