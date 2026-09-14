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
from std_srvs.srv import Trigger

# ============================================================
# 1. PHYSICAL CONSTANTS (same as master test makyr script)
# ============================================================
WAVELENGTH_M = 1.55e-6          # 1550nm
MODE_FIELD_RADIUS_M = 5.2e-6    # w0, SMF-28 @ 1550nm
N_MEDIUM = 1.0                  # air
THETA_EFF = WAVELENGTH_M / (np.pi * N_MEDIUM * MODE_FIELD_RADIUS_M)
Z_R = np.pi * N_MEDIUM * MODE_FIELD_RADIUS_M**2 / WAVELENGTH_M   # Rayleigh range [m]

# ---- SINGLE NOISE MODE SELECTOR (change this to 'none', 'low', or 'high') ----
NOISE_MODE = 'low'   # options: 'none', 'low', 'high'

# ---- Noise coefficients (can still be overridden from config) ----
SHOT_COEFF_LOW  = 0.002
DARK_STD_LOW    = 0.0004
SHOT_COEFF_HIGH = 0.01
DARK_STD_HIGH   = 0.002

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
        self.clear_buffer_srv = self.create_service(
            Trigger, 'clear_tf_buffer', self.handle_clear_buffer
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- Noise mode - set from global constant ----
        self.noise_mode = NOISE_MODE
        if self.noise_mode not in ('none', 'low', 'high'):
            self.get_logger().warn(
                f"⚠️ Unknown NOISE_MODE '{self.noise_mode}', falling back to 'none'."
            )
            self.noise_mode = 'none'

        # Coefficients (can be overridden from config)
        self.shot_coeff_low = SHOT_COEFF_LOW
        self.dark_std_low = DARK_STD_LOW
        self.shot_coeff_high = SHOT_COEFF_HIGH
        self.dark_std_high = DARK_STD_HIGH

        # Publishers
        self.signal_pubs = {}
        self.distance_pubs = {}
        self.angle_pubs = {}
        self.eta_lateral_pubs = {}
        self.eta_angular_pubs = {}
        self.eta_axial_pubs = {}
        self.eta_clean_pubs = {}
        self.dist_perc_pubs = {}
        self.dist_lateral_perc_pubs = {}
        self.dist_axial_perc_pubs = {}
        self.angle_perc_pubs = {}

        for pair in self.frame_pairs:
            self.signal_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/signal', 10)
            self.distance_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/distance', 10)
            self.angle_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/angle_deviation', 10)
            self.eta_lateral_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/eta_lateral', 10)
            self.eta_angular_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/eta_angular', 10)
            self.eta_axial_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/eta_axial', 10)
            self.eta_clean_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/eta_clean', 10)
            self.dist_perc_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/distance_percentage', 10)
            self.dist_lateral_perc_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/distance_lateral_percentage', 10)
            self.dist_axial_perc_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/distance_axial_percentage', 10)
            self.angle_perc_pubs[pair.name] = self.create_publisher(Float32, f'{pair.name}/angle_percentage', 10)

        self.timer = self.create_timer(0.005, self.compute_all_frame_deviations)

        # Load config (limits, enabled joints, etc.)
        self._load_config()

    # ------------------------------------------------------------------
    # Helper to locate the 'doc' folder in both source and install trees
    # ------------------------------------------------------------------
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

    def _load_config(self):
        doc_path = self.get_doc_path()
        config_path = doc_path / "alignment_config.json"

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.search_factor = config.get('SEARCH_FACTOR', 2.0)
            self.mode = config.get('MODE', '6D')
            self.trans_limit = config.get('trans_limit', self.search_factor * MODE_FIELD_RADIUS_M)
            self.rot_limit = config.get('rot_limit', self.search_factor * THETA_EFF)
            self.z_limit = config.get('Z_LIMIT', self.search_factor * Z_R)

            # Override coefficients from config if present (optional)
            self.shot_coeff_low = config.get('SHOT_NOISE_COEFF_LOW', self.shot_coeff_low)
            self.dark_std_low = config.get('DARK_NOISE_STD_LOW', self.dark_std_low)
            self.shot_coeff_high = config.get('SHOT_NOISE_COEFF_HIGH', self.shot_coeff_high)
            self.dark_std_high = config.get('DARK_NOISE_STD_HIGH', self.dark_std_high)

            # Get enabled joints
            enabled_joints = config.get('ENABLED_JOINTS', {})
            self.has_xy_lateral = enabled_joints.get('SP_X_Joint', False) or enabled_joints.get('SP_Y_Joint', False)
            self.has_z = enabled_joints.get('SP_Z_Joint', False)

            # ---- Maximum distances for percentage calculations ----
            # Lateral (XY) maximum: Euclidean norm of X and Y limits
            self.max_lateral_dist = 0.0
            n_xy = 0
            if enabled_joints.get('SP_X_Joint', False):
                self.max_lateral_dist += self.trans_limit**2
                n_xy += 1
            if enabled_joints.get('SP_Y_Joint', False):
                self.max_lateral_dist += self.trans_limit**2
                n_xy += 1
            self.max_lateral_dist = math.sqrt(self.max_lateral_dist) if self.max_lateral_dist > 0 else 1.0

            # Axial (Z) maximum
            self.max_axial_dist = self.z_limit if self.has_z else 1.0

            # Total translation maximum (Euclidean norm of X, Y, Z limits)
            max_total_sq = 0.0
            if self.has_xy_lateral:
                max_total_sq += n_xy * self.trans_limit**2
            if self.has_z:
                max_total_sq += self.z_limit**2
            self.max_total_dist = math.sqrt(max_total_sq) if max_total_sq > 0 else 1.0

            # ---- Maximum angle (tilt only, A and B) ----
            n_rot = sum(1 for j in ['SP_A_Joint', 'SP_B_Joint'] if enabled_joints.get(j, False))
            self.max_angle = math.sqrt(n_rot) * self.rot_limit if n_rot > 0 else 1.0

            # ---- Number of axes for eta calculation ----
            self.n_trans = config.get('n_trans_enabled', 3)
            self.n_rot = config.get('n_rot_enabled', 3)

            self.get_logger().info(f"✅ Loaded config from {config_path}: mode={self.mode}, SF={self.search_factor}")
            self.get_logger().info(f"   XY limit={self.trans_limit*1e6:.2f} µm, Z limit={self.z_limit*1e6:.2f} µm")
            self.get_logger().info(f"   max_lateral={self.max_lateral_dist*1e6:.2f} µm, max_axial={self.max_axial_dist*1e6:.2f} µm")
            self.get_logger().info(f"   max_total={self.max_total_dist*1e6:.2f} µm, max_angle={math.degrees(self.max_angle):.3f}°")
            self.get_logger().info(
                f"   noise_mode={self.noise_mode} "
                f"(low: shot={self.shot_coeff_low}, dark={self.dark_std_low} | "
                f"high: shot={self.shot_coeff_high}, dark={self.dark_std_high})"
            )

        except Exception as e:
            self.search_factor = 2.0
            self.trans_limit = self.search_factor * MODE_FIELD_RADIUS_M
            self.z_limit = self.search_factor * Z_R
            self.rot_limit = self.search_factor * THETA_EFF
            self.has_xy_lateral = True
            self.has_z = True
            self.max_lateral_dist = math.sqrt(2 * self.trans_limit**2)
            self.max_axial_dist = self.z_limit
            self.max_total_dist = math.sqrt(2 * self.trans_limit**2 + self.z_limit**2)
            n_rot = 2
            self.max_angle = math.sqrt(n_rot) * self.rot_limit
            self.n_trans, self.n_rot = 3, 3
            self.get_logger().warn(f"⚠️ Could not load config from {config_path}, using defaults. Error: {e}")
            self.get_logger().warn(f"   noise_mode={self.noise_mode} (from global constant)")

    def handle_clear_buffer(self, request, response):
        self.tf_buffer.clear()
        self.get_logger().info("TF buffer cleared.")
        response.success = True
        response.message = "TF buffer cleared"
        return response

    # ------------------------------------------------------------------
    # MAIN COMPUTATION – 5‑DOF coupling model
    # ------------------------------------------------------------------
    def compute_all_frame_deviations(self):
        for pair in self.frame_pairs:
            try:
                transform = self.tf_buffer.lookup_transform(
                    pair.frame_a, pair.frame_b, rclpy.time.Time()
                )
                t = transform.transform.translation
                q = transform.transform.rotation

                # ---- Extract misalignment components ----
                dx = t.x
                dy = t.y
                dz = t.z
                lateral_dist = math.sqrt(dx**2 + dy**2)   # only x,y
                axial_dist = abs(dz)                       # z

                # Rotation vector from quaternion
                r_rot = R.from_quat([q.x, q.y, q.z, q.w])
                rot_vec = r_rot.as_rotvec()
                theta_tilt = math.sqrt(rot_vec[0]**2 + rot_vec[1]**2)   # ignore rz

                # ---- Coupling efficiency terms ----
                # Lateral (XY)
                if self.n_trans > 0 and self.has_xy_lateral:
                    eta_lateral = math.exp(-(lateral_dist / MODE_FIELD_RADIUS_M) ** 2)
                else:
                    eta_lateral = 1.0

                # Angular (tilt, no Rz)
                if self.n_rot > 0:
                    eta_angular = math.exp(-(theta_tilt / THETA_EFF) ** 2)
                else:
                    eta_angular = 1.0

                # Axial (Z)
                if self.n_trans > 0 and self.has_z:
                    eta_axial = 1.0 / (1.0 + (axial_dist / Z_R) ** 2)
                else:
                    eta_axial = 1.0

                # Total clean efficiency
                eta_clean = eta_lateral * eta_angular * eta_axial

                # ---- Percentages ----
                total_dist = math.sqrt(dx**2 + dy**2 + dz**2)

                # Lateral percentage (based on XY limits)
                lateral_perc = (lateral_dist / self.max_lateral_dist) * 100 if self.max_lateral_dist > 0 else 0.0
                # Axial percentage (based on Z limit)
                axial_perc = (axial_dist / self.max_axial_dist) * 100 if self.max_axial_dist > 0 else 0.0
                # Total percentage (based on full Euclidean limit)
                total_perc = (total_dist / self.max_total_dist) * 100 if self.max_total_dist > 0 else 0.0
                # Angle percentage (based on tilt limits)
                angle_perc = (theta_tilt / self.max_angle) * 100 if self.max_angle > 0 else 0.0

                # ---- Noise injection (mode selected by global constant) ----
                if self.noise_mode == 'none':
                    powersignal = float(np.clip(eta_clean, 0.0, 1.0))
                elif self.noise_mode == 'low':
                    shot_noise = np.random.normal(0.0, self.shot_coeff_low * math.sqrt(max(eta_clean, 0.0)))
                    dark_noise = np.random.normal(0.0, self.dark_std_low)
                    powersignal = float(np.clip(eta_clean + shot_noise + dark_noise, 0.0, 1.0))
                elif self.noise_mode == 'high':
                    shot_noise = np.random.normal(0.0, self.shot_coeff_high * math.sqrt(max(eta_clean, 0.0)))
                    dark_noise = np.random.normal(0.0, self.dark_std_high)
                    powersignal = float(np.clip(eta_clean + shot_noise + dark_noise, 0.0, 1.0))
                else:
                    # Fallback (should never happen)
                    powersignal = float(np.clip(eta_clean, 0.0, 1.0))

                # ---- Publish all metrics ----
                self.signal_pubs[pair.name].publish(Float32(data=powersignal))
                self.distance_pubs[pair.name].publish(Float32(data=total_dist))
                self.angle_pubs[pair.name].publish(Float32(data=theta_tilt))
                self.eta_lateral_pubs[pair.name].publish(Float32(data=eta_lateral))
                self.eta_angular_pubs[pair.name].publish(Float32(data=eta_angular))
                self.eta_axial_pubs[pair.name].publish(Float32(data=eta_axial))
                self.eta_clean_pubs[pair.name].publish(Float32(data=eta_clean))
                self.dist_perc_pubs[pair.name].publish(Float32(data=total_perc))
                self.dist_lateral_perc_pubs[pair.name].publish(Float32(data=lateral_perc))
                self.dist_axial_perc_pubs[pair.name].publish(Float32(data=axial_perc))
                self.angle_perc_pubs[pair.name].publish(Float32(data=angle_perc))

                # ---- Log (optional) ----
                self.get_logger().info(
                    f"{pair.name}: lat={lateral_dist*1e6:.3f}µm [{lateral_perc:.2f}%], "
                    f"ax={axial_dist*1e6:.3f}µm [{axial_perc:.2f}%], "
                    f"tilt={math.degrees(theta_tilt):.3f}° [{angle_perc:.2f}%], "
                    f"eta={eta_clean:.5f}, signal={powersignal:.5f}"
                )

            except Exception as e:
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