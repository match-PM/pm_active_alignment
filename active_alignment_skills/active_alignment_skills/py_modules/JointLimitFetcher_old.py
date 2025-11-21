#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from urdf_parser_py.urdf import URDF, Joint as URDFJoint
import numpy as np
import asyncio

class JointLimitFetcher:
    """
    Helper class to fetch, cache, and map joint limits from the robot_description
    parameter via the robot_state_publisher.
    """

    def __init__(self, node: Node):
        """
        Args:
            node (Node): Reference to an existing rclpy Node (for logging and service calls)
        """
        self.node = node
        self.joint_limits: dict[str, tuple[float, float]] = {}
        self._urdf: URDF | None = None

        # Create client for robot description
        self.robot_description_client = self.node.create_client(
            GetParameters,
            '/robot_state_publisher/get_parameters'
        )

    async def update_joint_limits(self) -> None:
        """
        Fetch the full robot_description from the parameter server and parse all joint limits.
        """
        # Wait for service
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
                limits[joint.name] = (lower, upper)
            else:
                limits[joint.name] = (float('nan'), float('nan'))

        self.joint_limits = limits
        self.node.get_logger().info(f"Fetched limits for {len(self.joint_limits)} joints.")

    async def get_joint_limits(self, joint_names: list[str]) -> dict[str, tuple[float, float]]:
        """
        Get the limits for the given joints, updating the cache if needed.
        """
        if not self.joint_limits:
            await self.update_joint_limits()

        result = {}
        for j in joint_names:
            if j in self.joint_limits:
                result[j] = self.joint_limits[j]
            else:
                self.node.get_logger().warn(f"Joint '{j}' not found in URDF.")
        return result

    async def get_all_joint_limits(self) -> dict[str, tuple[float, float]]:
        """
        Get all known joint limits (fetching them if needed).
        """
        if not self.joint_limits:
            await self.update_joint_limits()
        return self.joint_limits

    # ------------------------------------------------------------
    #  NORMALIZATION UTILITIES
    # ------------------------------------------------------------

    async def normalize_joint_value(self, joint_name: str, value: float) -> float:
        """
        Map a joint value to [0, 1] range based on its limits.
        """
        if not self.joint_limits:
            await self.update_joint_limits()

        if joint_name not in self.joint_limits:
            self.node.get_logger().warn(f"Joint '{joint_name}' not found; returning 0.5 as default.")
            return 0.5

        lower, upper = self.joint_limits[joint_name]
        if any(map(lambda x: x != x, (lower, upper))):  # NaN check
            self.node.get_logger().warn(f"Invalid limits for joint '{joint_name}'; returning 0.5.")
            return 0.5

        if upper == lower:
            self.node.get_logger().warn(f"Equal limits for joint '{joint_name}'; returning 0.0.")
            return 0.0

        normalized = (value - lower) / (upper - lower)
        return max(0.0, min(1.0, normalized))

    async def denormalize_joint_value(self, joint_name: str, normalized_value: float) -> float:
        """
        Map a normalized value [0, 1] back to the real joint range.
        """
        if not self.joint_limits:
            await self.update_joint_limits()

        if joint_name not in self.joint_limits:
            self.node.get_logger().warn(f"Joint '{joint_name}' not found; returning 0.0 as default.")
            return 0.0

        lower, upper = self.joint_limits[joint_name]
        if any(map(lambda x: x != x, (lower, upper))):  # NaN check
            self.node.get_logger().warn(f"Invalid limits for joint '{joint_name}'; returning 0.0.")
            return 0.0

        normalized_value = max(0.0, min(1.0, normalized_value))
        return lower + normalized_value * (upper - lower)

    async def normalize_joint_values(self, joint_values: dict[str, float]) -> dict[str, float]:
        """
        Normalize multiple joint values at once.

        Args:
            joint_values (dict[str, float]): {joint_name: joint_value}
        Returns:
            dict[str, float]: {joint_name: normalized_value}
        """
        if not self.joint_limits:
            await self.update_joint_limits()

        result = {}
        for name, value in joint_values.items():
            result[name] = await self.normalize_joint_value(name, value)
        return result

    async def denormalize_joint_values(self, normalized_values: dict[str, float]) -> dict[str, float]:
        """
        Convert normalized joint values [0,1] back to real joint values.

        Args:
            normalized_values (dict[str, float]): {joint_name: normalized_value}
        Returns:
            dict[str, float]: {joint_name: real_value}
        """
        if not self.joint_limits:
            await self.update_joint_limits()

        result = {}
        for name, value in normalized_values.items():
            result[name] = await self.denormalize_joint_value(name, value)
        return result

    def map_joints_to_normalized(self, joint_values: dict[str, float]) -> dict[str, float]:
        """
        Map joint values to normalized [0,1] range based on limits.
        """
        result = {}
        for j, val in joint_values.items():
            low, high = self.joint_limits.get(j, (float('nan'), float('nan')))
            if np.isnan(low) or np.isnan(high) or low == high:
                result[j] = 0.5
            else:
                result[j] = np.clip((val - low) / (high - low), 0.0, 1.0)
        return result

    def map_normalized_to_joints(self, normalized_values: dict[str, float]) -> dict[str, float]:
        """
        Convert normalized [0,1] values back to joint values.
        """
        result = {}
        for j, val in normalized_values.items():
            low, high = self.joint_limits.get(j, (float('nan'), float('nan')))
            if np.isnan(low) or np.isnan(high):
                result[j] = 0.0
            else:
                result[j] = low + val * (high - low)
        return result