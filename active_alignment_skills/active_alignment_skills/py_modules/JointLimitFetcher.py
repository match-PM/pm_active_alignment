#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from controller_manager_msgs.srv import ListControllers
from urdf_parser_py.urdf import URDF, Joint as URDFJoint
import numpy as np


class JointLimitFetcher:
    """
    Helper class to fetch, cache, and map:
      - joint limits from the URDF (via robot_description)
      - controller names and types (via controller_manager)
    """

    def __init__(self, node: Node):
        self.node = node
        self.joint_limits: dict[str, tuple[float, float]] = {}
        self.joint_to_controller: dict[str, dict[str, str]] = {}  # {joint: {"name": str, "type": str}}
        self._urdf: URDF | None = None

        # ROS2 service clients
        self.robot_description_client = self.node.create_client(
            GetParameters,
            '/robot_state_publisher/get_parameters'
        )
        self.controller_list_client = self.node.create_client(
            ListControllers,
            '/controller_manager/list_controllers'
        )

    # ------------------------------------------------------------
    #  JOINT LIMIT FETCHING
    # ------------------------------------------------------------
    async def update_joint_limits(self) -> None:
        """Fetch and parse all joint limits from the URDF."""
        while not self.robot_description_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Waiting for /robot_state_publisher/get_parameters...")

        req = GetParameters.Request(names=['robot_description'])
        future = self.robot_description_client.call_async(req)
        await future

        try:
            response = future.result()
        except Exception as e:
            self.node.get_logger().error(f"Failed to call get_parameters: {e}")
            return

        if not response.values or not response.values[0].string_value:
            self.node.get_logger().error("robot_description parameter is empty.")
            return

        urdf_string = response.values[0].string_value
        self._urdf = URDF.from_xml_string(urdf_string)

        limits = {}
        for joint in self._urdf.joints:
            joint: URDFJoint
            if joint.limit:
                lower = getattr(joint.limit, 'lower', float('nan'))
                upper = getattr(joint.limit, 'upper', float('nan'))
                if lower is not float('nan') and upper is not float('nan'):
                    limits[joint.name] = (lower, upper)
                else:
                    continue
            else:
                continue
                #limits[joint.name] = (float('nan'), float('nan'))

        self.joint_limits = limits
        self.node.get_logger().info(f"✅ Parsed limits for {len(self.joint_limits)} joints.")

    # ------------------------------------------------------------
    #  CONTROLLER FETCHING
    # ------------------------------------------------------------
    async def update_joint_to_controller_mapping(self) -> None:
        """Fetch controller list and build joint→controller mapping (name + type)."""
        while not self.controller_list_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Waiting for /controller_manager/list_controllers...")

        req = ListControllers.Request()
        future = self.controller_list_client.call_async(req)
        await future

        try:
            response = future.result()
        except Exception as e:
            self.node.get_logger().error(f"Failed to call list_controllers: {e}")
            return

        mapping = {}
        for ctrl in response.controller:
            controller_name = ctrl.name
            controller_type = ctrl.type
            for iface in ctrl.claimed_interfaces:
                joint_name = iface.replace('/position', '').replace('/velocity', '').replace('/effort', '')
                mapping[joint_name] = {
                    "name": controller_name,
                    "type": controller_type,
                }

        self.joint_to_controller = mapping

    # ------------------------------------------------------------
    #  QUERY FUNCTIONS
    # ------------------------------------------------------------
    async def get_joint_info(self, joint_name: str) -> dict[str, str | float]:
        """
        Return the controller name, controller type, and joint limits for a given joint.

        Example:
        {
            'controller_name': 'arm_controller',
            'controller_type': 'joint_trajectory_controller/JointTrajectoryController',
            'lower': -1.57,
            'upper': 1.57
        }
        """
        if not self.joint_limits:
            await self.update_joint_limits()
        if not self.joint_to_controller:
            await self.update_joint_to_controller_mapping()

        controller_info = self.joint_to_controller.get(joint_name, {"name": "unknown", "type": "unknown"})
        lower, upper = self.joint_limits.get(joint_name, (float('nan'), float('nan')))

        if controller_info["name"] == "unknown":
            raise ValueError(f"Controller info for joint '{joint_name}' is not available. Joint name may be incorrect.")
        
        if np.isnan(lower) or np.isnan(upper):
            raise ValueError(f"Joint limits for joint '{joint_name}' are not available.")

        return {
            "controller_name": controller_info["name"],
            "controller_type": controller_info["type"],
            "lower": lower,
            "upper": upper
        }

    async def get_all_joint_info(self) -> dict[str, dict[str, str | float]]:
        """Return controller name, type, and limits for all known joints."""
        if not self.joint_limits:
            await self.update_joint_limits()
        if not self.joint_to_controller:
            await self.update_joint_to_controller_mapping()

        result = {}
        for name, (low, high) in self.joint_limits.items():
            controller_info = self.joint_to_controller.get(name, {"name": "unknown", "type": "unknown"})
            result[name] = {
                "controller_name": controller_info["name"],
                "controller_type": controller_info["type"],
                "lower": low,
                "upper": high
            }
        return result


    async def get_joints_for_controller(self, controller_name: str, controller_type: str | None = None) -> list[str]:
        """
        Return all joint names that belong to a specific controller.

        Args:
            controller_name (str): The name of the controller (e.g. 'arm_controller').
            controller_type (str | None): Optional type filter
                (e.g. 'joint_trajectory_controller/JointTrajectoryController').

        Returns:
            list[str]: Names of all joints claimed by that controller.
        """
        if not self.joint_to_controller:
            await self.update_joint_to_controller_mapping()

        joints = []
        for joint, info in self.joint_to_controller.items():
            if info["name"] == controller_name:
                if controller_type is None or info["type"] == controller_type:
                    joints.append(joint)

        if not joints:
            self.node.get_logger().warn(
                f"No joints found for controller '{controller_name}'"
                + (f" of type '{controller_type}'." if controller_type else ".")
            )

        return joints
