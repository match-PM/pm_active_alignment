#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import json
from pathlib import Path

# ============================================================
# 1. PHYSICAL CONSTANTS (same as master editor)
# ============================================================
WAVELENGTH_M = 1.55e-6          # 1550nm
MODE_FIELD_RADIUS_M = 5.2e-6    # w0, SMF-28 @ 1550nm
N_MEDIUM = 1.0                  # air
THETA_EFF = WAVELENGTH_M / (np.pi * N_MEDIUM * MODE_FIELD_RADIUS_M)

# Noise parameters (tweak as needed)
SHOT_NOISE_COEFF = 0.01
DARK_NOISE_STD = 0.002
NO_NOISE = False                # set True for deterministic debug

# ============================================================
# 2. FRAME PAIR DEFINITION
# ============================================================
class FramePair:
    def __init__(self, name: str, frame_a: str, frame_b: str):
        self.name = name
        self.frame_a = frame_a
        self.frame_b = frame_b


class MultiFrameDeviationNode(Node):
    def __init__(self):
        super().__init__('multi_frame_deviation_node')

        self.frame_pairs = [
            FramePair("Test1", "AL_Test_Frames_Align_1", "AL_Test_Frames_Target_1"),
            # add more pairs if needed
        ]

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.signal_pubs = {}
        self.distance_pubs = {}
        self.angle_pubs = {}
        self.eta_lateral_pubs = {}
        self.eta_angular_pubs = {}
        self.eta_clean_pubs = {}
        self.dist_perc_pubs = {}
        self.angle_perc_pubs = {}

        for pair in self.frame_pairs:
            self.signal_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/signal', 10)
            self.distance_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/distance', 10)
            self.angle_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/angle_deviation', 10)
            self.eta_lateral_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/eta_lateral', 10)
            self.eta_angular_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/eta_angular', 10)
            self.eta_clean_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/eta_clean', 10)
            self.dist_perc_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/distance_percentage', 10)
            self.angle_perc_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/angle_percentage', 10)

        self.timer = self.create_timer(0.005, self.compute_all_frame_deviations) 

        # Cache loaded config values
        self._load_config()

    # ------------------------------------------------------------------
    # Helper to locate the 'doc' folder in both source and install trees
    # ------------------------------------------------------------------
    @staticmethod
    def get_doc_path() -> Path:
        """
        Return the absolute path to the 'doc' folder, regardless of whether
        the script is running from the source tree or from the installed location.
        """
        script_path = Path(__file__).resolve()

        # Check if we are running inside the install tree
        if "install" in script_path.parts:
            # Walk up until we find the 'install' folder, then go to workspace root
            for parent in script_path.parents:
                if parent.name == "install":
                    ws_root = parent.parent
                    break
            else:
                # Fallback: assume we are in a standard colcon workspace
                # (usually 4 levels up from the script)
                ws_root = script_path.parents[4]

            # Source directory structure: src/pm_active_alignment/active_alignment_skills/doc
            doc_path = ws_root / "src" / "pm_active_alignment" / "active_alignment_skills" / "doc"
        else:
            # In development: script is in active_alignment_skills/active_alignment_skills/
            # doc is one level up: active_alignment_skills/doc/
            doc_path = script_path.parent.parent / "doc"

        return doc_path

    def _load_config(self):
        doc_path = self.get_doc_path()
        config_path = doc_path / "alignment_config.json"

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.search_factor = config.get('SEARCH_FACTOR', 1.6)
            mode = config.get('MODE', '6D')

            if mode == '6D':
                self.n_trans, self.n_rot = 3, 3
            elif mode == '2D_TRANS':
                self.n_trans, self.n_rot = 2, 0
            elif mode == '2D_ROT':
                self.n_trans, self.n_rot = 0, 2
            elif mode == '3D_TRANS':
                self.n_trans, self.n_rot = 3, 0
            elif mode == '3D_ROT':
                self.n_trans, self.n_rot = 0, 3
            else:
                self.n_trans, self.n_rot = 3, 3

            self.get_logger().info(f"✅ Loaded config from {config_path}: mode={mode}, SF={self.search_factor}")
        except Exception as e:
            self.search_factor = 1.6
            self.n_trans, self.n_rot = 3, 3
            self.get_logger().warn(f"⚠️ Could not load config from {config_path}, using defaults. Error: {e}")

        # Percentages use factor 2 (matches skill node's compute_dynamic_percentages)
        self.max_dist = (2 * math.sqrt(self.n_trans) * self.search_factor * MODE_FIELD_RADIUS_M) if self.n_trans > 0 else 1.0
        self.max_angle = (2 * math.sqrt(self.n_rot) * self.search_factor * THETA_EFF) if self.n_rot > 0 else 1.0

    # ------------------------------------------------------------------
    # MAIN COMPUTATION – matches skill node's compute_dynamic_percentages
    # ------------------------------------------------------------------
    def compute_all_frame_deviations(self):
        for pair in self.frame_pairs:
            try:
                transform = self.tf_buffer.lookup_transform(
                    pair.frame_a, pair.frame_b, rclpy.time.Time()
                )
                t = transform.transform.translation
                distance = math.sqrt(t.x**2 + t.y**2 + t.z**2)
                q = transform.transform.rotation
                r = R.from_quat([q.x, q.y, q.z, q.w])
                angle_deviation = r.magnitude()

                # ---- Coupling efficiency (same as master) ----
                eta_lateral = math.exp(-(distance / MODE_FIELD_RADIUS_M) ** 2)
                eta_angular = math.exp(-(angle_deviation / THETA_EFF) ** 2)
                eta_clean = eta_lateral * eta_angular

                # ---- Percentages (using skill node's formula with factor 2) ----
                distance_percentage = (distance / self.max_dist) * 100 if self.max_dist > 0 else 0.0
                angle_percentage = (angle_deviation / self.max_angle) * 100 if self.max_angle > 0 else 0.0

                # ---- Add noise (optional) ----
                if NO_NOISE:
                    powersignal = float(np.clip(eta_clean, 0.0, 1.0))
                else:
                    shot_noise = np.random.normal(0.0, SHOT_NOISE_COEFF * math.sqrt(max(eta_clean, 0.0)))
                    dark_noise = np.random.normal(0.0, DARK_NOISE_STD)
                    powersignal = float(np.clip(eta_clean + shot_noise + dark_noise, 0.0, 1.0))

                # ---- Publish all metrics ----
                self.signal_pubs[pair.name].publish(Float32(data=powersignal))
                self.distance_pubs[pair.name].publish(Float32(data=distance))
                self.angle_pubs[pair.name].publish(Float32(data=angle_deviation))
                self.eta_lateral_pubs[pair.name].publish(Float32(data=eta_lateral))
                self.eta_angular_pubs[pair.name].publish(Float32(data=eta_angular))
                self.eta_clean_pubs[pair.name].publish(Float32(data=eta_clean))
                self.dist_perc_pubs[pair.name].publish(Float32(data=distance_percentage))
                self.angle_perc_pubs[pair.name].publish(Float32(data=angle_percentage))

                # ---- Log (optional) ----
                self.get_logger().info(
                    f"{pair.name}: dist={distance*1e6:.3f} µm [{distance_percentage:.2f}%], "
                    f"angle={math.degrees(angle_deviation):.3f}° [{angle_percentage:.2f}%], "
                    f"eta_clean={eta_clean:.5f}, signal={powersignal:.5f}"
                )

            except Exception as e:
                # Suppress spam, but you can uncomment for debugging:
                # self.get_logger().warn(f"TF lookup failed for {pair.name}: {e}")
                pass


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