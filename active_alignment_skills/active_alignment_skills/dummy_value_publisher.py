#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import math


class FramePair:
    """Holds a pair of TF frames to compute deviations."""
    def __init__(self, name: str, 
                 frame_a: str, 
                 frame_b: str):
        
        self.name = name
        self.frame_a = frame_a
        self.frame_b = frame_b



class MultiFrameDeviationNode(Node):
    def __init__(self):
        super().__init__('multi_frame_deviation_node')

        # Hardcode the frame pairs
        self.frame_pairs: List[FramePair] = [
            FramePair("Test1", "AL_Test_Frames_Align_1", "AL_Test_Frames_Target_1"),
            FramePair("Test2", "AL_Test_Frames_Align_2", "AL_Test_Frames_Target_2"),
            #FramePair("camera_to_base", "camera_link", "base_link")
            # Add more pairs as needed
        ]


        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers for each pair
        self.distance_pubs = {}
        self.angle_pubs = {}

        for pair in self.frame_pairs:
            self.distance_pubs[pair.name] = self.create_publisher(
                Float32, f'{pair.name}/distance',10)
            self.angle_pubs[pair.name] = self.create_publisher(
                Float32, f'{pair.name}/angle', 10)

        # Timer to periodically compute deviations
        self.timer = self.create_timer(0.1, self.compute_all_frame_deviations)  # 10 Hz

        # For throttled warnings
        self._last_warn_times = {pair.name: 0.0 for pair in self.frame_pairs}

    def compute_all_frame_deviations(self):
        for pair in self.frame_pairs:
            try:
                # Lookup transform from frame_a -> frame_b
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    pair.frame_a, pair.frame_b, rclpy.time.Time()
                )

                # Extract translation
                t = transform.transform.translation
                distance = math.sqrt(t.x**2 + t.y**2 + t.z**2)

                distance= 10*distance
                # Extract rotation as quaternion
                q = transform.transform.rotation
                r = R.from_quat([q.x, q.y, q.z, q.w])
                # Angle deviation as magnitude of rotation vector
                angle_deviation = r.magnitude()

                # Publish using the node's publisher dictionaries
                self.distance_pubs[pair.name].publish(Float32(data=distance))
                self.angle_pubs[pair.name].publish(Float32(data=angle_deviation))

                self.get_logger().info(
                    f"{pair.name}: distance={distance/10*1e3:.5f} mm, angle_dev={math.degrees(angle_deviation):.5f} deg"
                )

            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for {pair.name}: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = MultiFrameDeviationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
