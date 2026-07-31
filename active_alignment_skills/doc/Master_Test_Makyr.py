#!/usr/bin/env python3
"""
master_json_editor.py

Central configuration for your alignment experiments.
Directly modifies:
  1. spawn_test_frames.json  – random misalignment
  2. test_active_alignment.rsap.json – joint limits and active joint toggles
  3. ring_spawn_info.json    – reproducible ring positions (if ring mode enabled)
  4. seed_poses.json         – multiple seed poses for batch testing

SPAWN MODE PRIORITY (highest to lowest):
  1. search_debug_6d      – fixed custom offset on ALL 6 axes
  2. search_debug         – fixed custom offset, distributed ONLY on enabled joints
  3. USE_RING_SPAWN       – spawn at a fixed power contour (ring)
  4. translatory_debug    – only rotation, no translation (random)
  5. angular_debug        – only translation, no rotation (random)
  6. debug                – perfect alignment (no misalignment)
  7. (default)            – full random misalignment within capture range

RANDOM only affects the seed used for randomisation – it does NOT override ring or debug modes.
RANDOMIZE_ONLY_ENABLED_JOINTS only applies when in default random mode (priority 7).

Features:
  - Mode selection for common experiments (6D, 2D_TRANS, 2D_ROT, 3D_TRANS, 3D_ROT).
  - Re‑enables joints that were previously removed.
  - Joint limits are set to positive values (symmetry handled elsewhere).
  - Ring spawn: generate one specific pose on a given power contour.
  - Seed generation: generate multiple seed poses per power level for batch testing.

Run this script whenever you change parameters or joint toggles.

!!!!!!! IMPORTANT !!!!!!!
When changing modes, need to restart whole Program because Algorithm subs will not update the joint toggles. Only on restart !
!!!!!!! IMPORTANT !!!!!!!
"""

import json
import random
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

# ============================================================
# 1. CENTRAL CONFIGURATION – EDIT THESE VALUES
# ============================================================

# ---- Physical parameters ----
W0 = 5.2e-6               # Mode field radius [m] (SMF-28 @ 1550nm)
LAMBDA = 1.55e-6          # Wavelength [m]
SEARCH_FACTOR = 2.0       # σ (search range = ±SEARCH_FACTOR * w0 or theta_eff)

# ---- Experiment mode (overrides manual joint toggles) ----
# Options: "6D", "2D_TRANS", "2D_ROT", "3D_TRANS", "3D_ROT"
MODE = "6D"  # Set to None to use manual joint toggles

# ---- Joint toggles (manual override, only used if MODE is None) ----
ENABLED_JOINTS_MANUAL = {
    "SP_X_Joint": True,
    "SP_Y_Joint": True,
    "SP_Z_Joint": True,
    "SP_A_Joint": True,
    "SP_B_Joint": True,
    "SP_C_Joint": True,
}

# ---- Randomisation controls ----
RANDOM = True              # True = random seed, False = fixed seed
SEED = 1622                # Fixed seed (only used if RANDOM is False)
RANDOMIZE_ONLY_ENABLED_JOINTS = True  # If True, only randomise joints that are enabled

# ---- Spawner debug flags (priority order as listed above) ----
debug = False              # Perfect alignment (no misalignment)
translatory_debug = False  # Only rotation, no translation
angular_debug = False      # Only translation, no rotation
search_debug = False       # Fixed custom offset, distributed ONLY on enabled joints
search_debug_6d = False    # Fixed custom offset on ALL 6 axes (highest priority)

# ---- Custom offset for search_debug and search_debug_6d ----
MAGNITUDE_OFSET_UM = 12    # micro meters (total Euclidean distance)
CUSTOM_MAGNITUDE_ANGLE_DEG = 3.0  # degrees (total rotation magnitude)

# ---- RING SPAWN controls (single pose for spawn_test_frames.json) ----
USE_RING_SPAWN = False                 # If True, spawn at a fixed power level
RING_POWER_PERCENT = 5.0               # Target power in %
RING_NUM_SEEDS = 4                     # Number of seeds to generate on the ring
RING_FIXED_SEED_BASE = 1000            # Base seed for ring generation
RING_SELECT_INDEX = 0                  # Which seed to use (0..RING_NUM_SEEDS-1)
RING_SAVE_JSON = True                  # Save ring_spawn_info.json

# ---- SEED GENERATION for multi‑seed batch testing ----
GENERATE_SEED_POSES = True
SEED_POWER_LEVELS = [2.0, 20.0, 60.0]   # power percentages to generate seeds for
SEEDS_PER_LEVEL = 4                     # number of seeds per power level
SEED_BASE = 592                        # starting seed number

# ---- JSON file paths ----
SCRIPT_DIR = Path(__file__).parent
SPAWN_JSON_PATH = SCRIPT_DIR / "spawn_test_frames.json"
ACTION_JSON_PATH = SCRIPT_DIR / "test_active_alignment.rsap.json"
RING_INFO_PATH = SCRIPT_DIR / "ring_spawn_info.json"
SEED_POSES_PATH = SCRIPT_DIR / "seed_poses.json"

# ============================================================
# 2. DERIVED QUANTITIES & HELPERS (do not edit)
# ============================================================
THETA_EFF = LAMBDA / (np.pi * W0)           # Angular tolerance [rad]
THETA_EFF_DEG = np.degrees(THETA_EFF)       # Angular tolerance [deg]
MAGNITUDE_OFSET_M = MAGNITUDE_OFSET_UM * 1e-6 # Convert µm to m

# ---- Search bounds ----
TRANS_LIMIT_M = SEARCH_FACTOR * W0          # [m]
TRANS_LIMIT_MM = TRANS_LIMIT_M * 1000       # [mm]
ROT_LIMIT_RAD = SEARCH_FACTOR * THETA_EFF   # [rad]
ROT_LIMIT_DEG = np.degrees(ROT_LIMIT_RAD)   # [deg]

# ---- Custom offset (computed) ----
CUSTOM_DISTANCE_M = MAGNITUDE_OFSET_M
CUSTOM_ANGLE_RAD = np.radians(CUSTOM_MAGNITUDE_ANGLE_DEG)

# ---- All possible joint names (for re‑enabling) ----
ALL_JOINTS = ["SP_X_Joint", "SP_Y_Joint", "SP_Z_Joint",
              "SP_A_Joint", "SP_B_Joint", "SP_C_Joint"]

TRANSLATION_JOINTS = {"SP_X_Joint", "SP_Y_Joint", "SP_Z_Joint"}
ROTATION_JOINTS = {"SP_A_Joint", "SP_B_Joint", "SP_C_Joint"}

# ---- Determine ENABLED_JOINTS from MODE ----
def get_enabled_joints(mode):
    if mode == "6D":
        return {j: True for j in ALL_JOINTS}
    elif mode == "2D_TRANS":
        return {"SP_X_Joint": True, "SP_Y_Joint": True,
                "SP_Z_Joint": False, "SP_A_Joint": False,
                "SP_B_Joint": False, "SP_C_Joint": False}
    elif mode == "2D_ROT":
        return {"SP_X_Joint": False, "SP_Y_Joint": False,
                "SP_Z_Joint": False, "SP_A_Joint": True,
                "SP_B_Joint": True, "SP_C_Joint": False}
    elif mode == "3D_TRANS":
        return {"SP_X_Joint": True, "SP_Y_Joint": True,
                "SP_Z_Joint": True, "SP_A_Joint": False,
                "SP_B_Joint": False, "SP_C_Joint": False}
    elif mode == "3D_ROT":
        return {"SP_X_Joint": False, "SP_Y_Joint": False,
                "SP_Z_Joint": False, "SP_A_Joint": True,
                "SP_B_Joint": True, "SP_C_Joint": True}
    else:
        raise ValueError(f"Unknown MODE: {mode}")

ENABLED_JOINTS = get_enabled_joints(MODE)

# ---- Joint name → axis mapping for restricted randomisation ----
TRANS_AXIS_MAP = {
    "SP_X_Joint": 0,
    "SP_Y_Joint": 1,
    "SP_Z_Joint": 2,
}
ROT_AXIS_MAP = {
    "SP_A_Joint": 0,
    "SP_B_Joint": 1,
    "SP_C_Joint": 2,
}

# ---- Compute enabled axes counts ----
n_trans_enabled = sum(1 for j in TRANSLATION_JOINTS if ENABLED_JOINTS.get(j, False))
n_rot_enabled = sum(1 for j in ROTATION_JOINTS if ENABLED_JOINTS.get(j, False))

# ============================================================
# 3. RING SPAWN HELPERS (subspace‑aware)
# ============================================================
def random_unit_vector_nd(n):
    """Generate a random unit vector in n-dimensional space."""
    if n == 0:
        return np.array([])
    v = np.random.normal(size=n)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        # fallback: first axis
        v = np.zeros(n)
        v[0] = 1.0
        return v
    return v / norm

def generate_random_full_range_pose(seed, n_trans, n_rot):
    """
    Generate a random pose uniformly across the full capture range,
    used for 0% power (no contour constraint).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Translation: uniform within +/- TRANS_LIMIT_M
    dx_m = np.random.uniform(-TRANS_LIMIT_M, TRANS_LIMIT_M) if n_trans > 0 else 0.0
    dy_m = np.random.uniform(-TRANS_LIMIT_M, TRANS_LIMIT_M) if n_trans > 1 else 0.0
    dz_m = np.random.uniform(-TRANS_LIMIT_M, TRANS_LIMIT_M) if n_trans > 2 else 0.0

    # Rotation: uniform within +/- ROT_LIMIT_RAD
    rx = np.random.uniform(-ROT_LIMIT_RAD, ROT_LIMIT_RAD) if n_rot > 0 else 0.0
    ry = np.random.uniform(-ROT_LIMIT_RAD, ROT_LIMIT_RAD) if n_rot > 1 else 0.0
    rz = np.random.uniform(-ROT_LIMIT_RAD, ROT_LIMIT_RAD) if n_rot > 2 else 0.0

    # Convert rotation to quaternion
    if n_rot > 0 and np.linalg.norm([rx, ry, rz]) > 1e-12:
        rot = R.from_rotvec([rx, ry, rz])
    else:
        rot = R.from_quat([0, 0, 0, 1])
    q = rot.as_quat()
    quat = {"X": float(q[0]), "Y": float(q[1]), "Z": float(q[2]), "W": float(q[3])}

    d_actual = np.linalg.norm([dx_m, dy_m, dz_m])
    theta_actual = np.linalg.norm([rx, ry, rz])

    return dx_m, dy_m, dz_m, quat, d_actual, theta_actual, None

def generate_below_1_percent_pose(seed, n_trans, n_rot, max_attempts=100):
    """
    Generate a random pose that guarantees eta_clean < 0.01 (below 1% power).
    If the current limits are insufficient to reach below 1%, we sample the full range
    and return the lowest-signal pose found, with a warning.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Precompute the required radius for 1% (eta = 0.01)
    # For combined case: d^2/W0^2 + theta^2/theta_eff^2 > -ln(0.01) ≈ 4.605
    target_radius_sq = -np.log(0.01)  # ~4.605

    # If only translation or only rotation, we can simplify.
    # But we'll use a generic search: sample uniformly within the limits and check.
    best_d = 0.0
    best_theta = 0.0
    best_eta = 1.0  # worst
    best_pose = None

    for attempt in range(max_attempts):
        # Sample within allowed limits
        dx_m = np.random.uniform(-TRANS_LIMIT_M, TRANS_LIMIT_M) if n_trans > 0 else 0.0
        dy_m = np.random.uniform(-TRANS_LIMIT_M, TRANS_LIMIT_M) if n_trans > 1 else 0.0
        dz_m = np.random.uniform(-TRANS_LIMIT_M, TRANS_LIMIT_M) if n_trans > 2 else 0.0
        rx = np.random.uniform(-ROT_LIMIT_RAD, ROT_LIMIT_RAD) if n_rot > 0 else 0.0
        ry = np.random.uniform(-ROT_LIMIT_RAD, ROT_LIMIT_RAD) if n_rot > 1 else 0.0
        rz = np.random.uniform(-ROT_LIMIT_RAD, ROT_LIMIT_RAD) if n_rot > 2 else 0.0

        d_actual = np.linalg.norm([dx_m, dy_m, dz_m])
        theta_actual = np.linalg.norm([rx, ry, rz])

        # Compute eta components
        eta_lateral = np.exp(- (d_actual / W0)**2) if n_trans > 0 else 1.0
        eta_angular = np.exp(- (theta_actual / THETA_EFF)**2) if n_rot > 0 else 1.0
        eta_clean = eta_lateral * eta_angular

        # Track the best (lowest) eta found
        if eta_clean < best_eta:
            best_eta = eta_clean
            best_d = d_actual
            best_theta = theta_actual
            # Build quaternion
            if n_rot > 0 and np.linalg.norm([rx, ry, rz]) > 1e-12:
                rot = R.from_rotvec([rx, ry, rz])
            else:
                rot = R.from_quat([0, 0, 0, 1])
            q = rot.as_quat()
            best_quat = {"X": float(q[0]), "Y": float(q[1]), "Z": float(q[2]), "W": float(q[3])}
            best_pose = (dx_m, dy_m, dz_m, best_quat, d_actual, theta_actual, None)

            if eta_clean < 0.01:
                # Found a pose below 1%
                return best_pose

    # If we exit the loop, we never found a below-1% pose.
    # Log a warning and return the best we found.
    print(
        f"Could not generate a pose with eta_clean < 0.01 within {max_attempts} attempts "
        f"using current limits (TRANS_LIMIT_M={TRANS_LIMIT_M:.2e}, ROT_LIMIT_RAD={ROT_LIMIT_RAD:.2e}). "
        f"Best eta_clean was {best_eta:.4f}. Increase SEARCH_FACTOR or widen limits."
    )
    if best_pose is not None:
        return best_pose
    else:
        # Fallback: just return a zero pose (should not happen)
        return 0.0, 0.0, 0.0, {"X": 0.0, "Y": 0.0, "Z": 0.0, "W": 1.0}, 0.0, 0.0, None

def generate_ring_pose(seed, eta_target, n_trans, n_rot):
    """
    Generate a pose on the contour where η = eta_target.
    If eta_target <= 0, generate a random pose that guarantees eta < 0.01 (if possible).
    """
    if eta_target <= 0.0:
        return generate_below_1_percent_pose(seed, n_trans, n_rot)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    r = np.sqrt(-np.log(eta_target))
    a = W0 * r          # translation semi‑axis (m)
    b = THETA_EFF * r   # rotation semi‑axis (rad)

    has_trans = n_trans > 0
    has_rot = n_rot > 0

    if has_trans and has_rot:
        # Both enabled: pick a point on the ellipse
        phi = random.uniform(0, 2 * np.pi)
        d = a * np.cos(phi)
        theta = b * np.sin(phi)
        d = abs(d)
        theta = abs(theta)
    elif has_trans and not has_rot:
        # Only translation: theta = 0, d = a
        d = a
        theta = 0.0
        phi = None
    elif not has_trans and has_rot:
        # Only rotation: d = 0, theta = b
        d = 0.0
        theta = b
        phi = None
    else:
        # No axes enabled – shouldn't happen
        d = 0.0
        theta = 0.0
        phi = None

    # Distribute d and theta among enabled axes
    trans_vec = random_unit_vector_nd(n_trans) * d if n_trans > 0 else np.zeros(3)
    rot_vec = random_unit_vector_nd(n_rot) * theta if n_rot > 0 else np.zeros(3)

    # Map to axis variables
    dx_m = trans_vec[0] if n_trans > 0 else 0.0
    dy_m = trans_vec[1] if n_trans > 1 else 0.0
    dz_m = trans_vec[2] if n_trans > 2 else 0.0

    rx = rot_vec[0] if n_rot > 0 else 0.0
    ry = rot_vec[1] if n_rot > 1 else 0.0
    rz = rot_vec[2] if n_rot > 2 else 0.0

    # Convert rotation vector to quaternion
    if n_rot > 0 and np.linalg.norm([rx, ry, rz]) > 1e-12:
        rot = R.from_rotvec([rx, ry, rz])
    else:
        rot = R.from_quat([0, 0, 0, 1])
    q = rot.as_quat()
    quat = {"X": float(q[0]), "Y": float(q[1]), "Z": float(q[2]), "W": float(q[3])}

    d_actual = np.linalg.norm([dx_m, dy_m, dz_m])
    theta_actual = np.linalg.norm([rx, ry, rz])

    return dx_m, dy_m, dz_m, quat, d_actual, theta_actual, phi

def generate_ring_poses(num_seeds, base_seed, power_percent, n_trans, n_rot):
    eta_target = power_percent / 100.0
    poses = []
    for i in range(num_seeds):
        seed = base_seed + i
        dx_m, dy_m, dz_m, quat, d, theta, phi = generate_ring_pose(
            seed, eta_target, n_trans, n_rot)
        poses.append({
            "seed": seed,
            "dx_m": dx_m,
            "dy_m": dy_m,
            "dz_m": dz_m,
            "quat": quat,
            "d": d,
            "theta": theta,
            "phi": phi,
            "eta_target": eta_target,
            "n_trans_enabled": n_trans,
            "n_rot_enabled": n_rot
        })
    return poses

# ============================================================
# 4. SPAWN JSON (misalignment) – uses selected ring pose or random
# ============================================================
def random_offset_m(limit_m: float) -> float:
    return random.uniform(-limit_m, limit_m)

def random_small_rotation(max_deg: float) -> dict:
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle_deg = random.uniform(0.0, max_deg)
    rot = R.from_rotvec(np.radians(angle_deg) * axis)
    q = rot.as_quat()
    return {"X": float(q[0]), "Y": float(q[1]), "Z": float(q[2]), "W": float(q[3])}

def random_rotation_in_subspace(enabled_rot_joints, max_deg) -> dict:
    active_axes = []
    for joint, enabled in enabled_rot_joints.items():
        if enabled:
            active_axes.append(ROT_AXIS_MAP[joint])
    if not active_axes:
        return {"X": 0.0, "Y": 0.0, "Z": 0.0, "W": 1.0}
    vec = np.zeros(3)
    for ax in active_axes:
        vec[ax] = np.random.normal(0.0, 1.0)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        vec[active_axes[0]] = 1.0
        norm = 1.0
    vec /= norm
    angle_deg = random.uniform(0.0, max_deg)
    rot_vec = vec * np.radians(angle_deg)
    rot = R.from_rotvec(rot_vec)
    q = rot.as_quat()
    return {"X": float(q[0]), "Y": float(q[1]), "Z": float(q[2]), "W": float(q[3])}

def build_spawn_json(seed: int | None = None, ring_pose=None):
    """
    Build spawn JSON. If ring_pose is provided, use it; otherwise use random/debug logic.
    ring_pose: dict with keys dx_m, dy_m, dz_m, quat.
    Priority of debug flags (highest first):
      1. search_debug_6d
      2. search_debug
      3. USE_RING_SPAWN (handled outside this function)
      4. translatory_debug
      5. angular_debug
      6. debug
      7. default random
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    trans_enabled = [TRANS_AXIS_MAP[j] for j in TRANSLATION_JOINTS if ENABLED_JOINTS.get(j, False)]
    rot_enabled = {j: ENABLED_JOINTS.get(j, False) for j in ROTATION_JOINTS}
    n_trans_enabled = len(trans_enabled)

    # ---- Determine misalignment ----
    if ring_pose is not None:
        dx_m = ring_pose["dx_m"]
        dy_m = ring_pose["dy_m"]
        dz_m = ring_pose["dz_m"]
        quat = ring_pose["quat"]
    elif search_debug_6d:
        dx_m = CUSTOM_DISTANCE_M / np.sqrt(3)
        dy_m = CUSTOM_DISTANCE_M / np.sqrt(3)
        dz_m = CUSTOM_DISTANCE_M / np.sqrt(3)
        custom_offset = CUSTOM_ANGLE_RAD / np.sqrt(3)
        rot = R.from_rotvec([custom_offset, custom_offset, custom_offset])
        q = rot.as_quat()
        quat = {"X": float(q[0]), "Y": float(q[1]), "Z": float(q[2]), "W": float(q[3])}
    elif search_debug:
        if n_trans_enabled > 0:
            per_axis_dist = CUSTOM_DISTANCE_M / np.sqrt(n_trans_enabled)
        else:
            per_axis_dist = 0.0
        dx_m = per_axis_dist if 0 in trans_enabled else 0.0
        dy_m = per_axis_dist if 1 in trans_enabled else 0.0
        dz_m = per_axis_dist if 2 in trans_enabled else 0.0
        rot_vec = np.zeros(3)
        for ax in range(3):
            joint_name = list(ROT_AXIS_MAP.keys())[list(ROT_AXIS_MAP.values()).index(ax)]
            if ENABLED_JOINTS.get(joint_name, False):
                rot_vec[ax] = CUSTOM_ANGLE_RAD / np.sqrt(3)
        norm = np.linalg.norm(rot_vec)
        if norm > 1e-12:
            rot_vec = rot_vec / norm * CUSTOM_ANGLE_RAD
        rot = R.from_rotvec(rot_vec)
        q = rot.as_quat()
        quat = {"X": float(q[0]), "Y": float(q[1]), "Z": float(q[2]), "W": float(q[3])}
    elif translatory_debug:
        dx_m, dy_m, dz_m = 0.0, 0.0, 0.0
        if RANDOMIZE_ONLY_ENABLED_JOINTS:
            quat = random_rotation_in_subspace(rot_enabled, ROT_LIMIT_DEG)
        else:
            quat = random_small_rotation(ROT_LIMIT_DEG)
    elif angular_debug:
        if RANDOMIZE_ONLY_ENABLED_JOINTS:
            dx_m = random_offset_m(TRANS_LIMIT_M) if 0 in trans_enabled else 0.0
            dy_m = random_offset_m(TRANS_LIMIT_M) if 1 in trans_enabled else 0.0
            dz_m = random_offset_m(TRANS_LIMIT_M) if 2 in trans_enabled else 0.0
        else:
            dx_m = random_offset_m(TRANS_LIMIT_M)
            dy_m = random_offset_m(TRANS_LIMIT_M)
            dz_m = random_offset_m(TRANS_LIMIT_M)
        quat = {"X": 0.0, "Y": 0.0, "Z": 0.0, "W": 1.0}
    elif debug:
        dx_m, dy_m, dz_m = 0.0, 0.0, 0.0
        quat = {"X": 0.0, "Y": 0.0, "Z": 0.0, "W": 1.0}
    else:
        # Full random within capture range
        if RANDOMIZE_ONLY_ENABLED_JOINTS:
            dx_m = random_offset_m(TRANS_LIMIT_M) if 0 in trans_enabled else 0.0
            dy_m = random_offset_m(TRANS_LIMIT_M) if 1 in trans_enabled else 0.0
            dz_m = random_offset_m(TRANS_LIMIT_M) if 2 in trans_enabled else 0.0
            quat = random_rotation_in_subspace(rot_enabled, ROT_LIMIT_DEG)
        else:
            dx_m = random_offset_m(TRANS_LIMIT_M)
            dy_m = random_offset_m(TRANS_LIMIT_M)
            dz_m = random_offset_m(TRANS_LIMIT_M)
            quat = random_small_rotation(ROT_LIMIT_DEG)

    dx_mm = dx_m * 1000
    dy_mm = dy_m * 1000
    dz_mm = dz_m * 1000
    Z_BASE = 90.0 - 32.62570
    z_final_mm = Z_BASE + dz_mm

    return {
        "document_units": "mm",
        "unique_identifier": "AL_Test_Frames_",
        "frames": [
            {
                "name": "Target_1",
                "parent_frame": "SmarPod_Origin",
                "transformation": {
                    "translation": {"X": dx_mm, "Y": dy_mm, "Z": z_final_mm},
                    "rotation": quat
                },
                "constraints": {}
            },
            {
                "name": "Align_1",
                "parent_frame": "Smarpod_Part_Spawn",
                "transformation": {
                    "translation": {"X": 0.0, "Y": 0.0, "Z": 0.0},
                    "rotation": {"X": 0.0, "Y": 0.0, "Z": 0.0, "W": 1.0}
                },
                "constraints": {}
            }
        ]
    }

def determine_spawn_mode():
    """Return a string describing the active spawn mode based on priority."""
    if search_debug_6d:
        return "search_debug_6d (fixed offset on all 6 axes)"
    if search_debug:
        return "search_debug (fixed offset on enabled joints)"
    if USE_RING_SPAWN:
        return "ring spawn (fixed power contour)"
    if translatory_debug:
        return "translatory_debug (rotation only)"
    if angular_debug:
        return "angular_debug (translation only)"
    if debug:
        return "debug (perfect alignment)"
    return "full random misalignment"

# ============================================================
# 4.5. ETA & PERCENTAGE COMPUTATION
# ============================================================

def compute_eta_and_percentages(dx_m, dy_m, dz_m, quat, n_trans, n_rot):
    d_actual = np.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
    r = R.from_quat([quat['X'], quat['Y'], quat['Z'], quat['W']])
    theta_actual = r.magnitude()
    eta_lateral = np.exp(- (d_actual / W0)**2) if n_trans > 0 else 1.0
    eta_angular = np.exp(- (theta_actual / THETA_EFF)**2) if n_rot > 0 else 1.0
    eta_clean = eta_lateral * eta_angular
    if n_trans > 0:
        max_dist = np.sqrt(n_trans) * TRANS_LIMIT_M
        dist_perc = (d_actual / max_dist) * 100
    else:
        dist_perc = 0.0
    if n_rot > 0:
        max_angle = np.sqrt(n_rot) * ROT_LIMIT_RAD
        angle_perc = (theta_actual / max_angle) * 100
    else:
        angle_perc = 0.0
    return {
        'eta_clean': eta_clean,
        'eta_lateral': eta_lateral,
        'eta_angular': eta_angular,
        'dist_perc': dist_perc,
        'angle_perc': angle_perc,
        'eta_lateral_above_1pct': eta_lateral > 0.01,
        'eta_angular_above_1pct': eta_angular > 0.01,
    }

def write_spawn_json(seed: int | None = None, ring_pose=None):
    os.makedirs(SPAWN_JSON_PATH.parent, exist_ok=True)
    spawn = build_spawn_json(seed=seed, ring_pose=ring_pose)
    with open(SPAWN_JSON_PATH, "w") as f:
        json.dump(spawn, f, indent=2)

    Z_BASE = 90.0 - 32.62570
    trans = spawn['frames'][0]['transformation']['translation']
    q = spawn['frames'][0]['transformation']['rotation']
    r = R.from_quat([q['X'], q['Y'], q['Z'], q['W']])
    rot_vec = r.as_rotvec()
    rot_magnitude = r.magnitude()

    dx_m = trans['X'] / 1000.0
    dy_m = trans['Y'] / 1000.0
    dz_m = (trans['Z'] - Z_BASE) / 1000.0

    eta_info = compute_eta_and_percentages(dx_m, dy_m, dz_m, q, n_trans_enabled, n_rot_enabled)

    print("\n--- Spawn JSON written ---")
    print(f"  dx = {trans['X']*1000:.5f} µm")
    print(f"  dy = {trans['Y']*1000:.5f} µm")
    print(f"  dz = {(trans['Z']-Z_BASE)*1000:.5f} µm")
    print(f"  distance magnitude: {np.sqrt(dx_m**2 + dy_m**2 + dz_m**2)*1e6:.5f} µm")
    print(f"  rotation quaternion: {q}")
    print(f"  rotation magnitude: {np.degrees(rot_magnitude):.5f}°")
    print(f"  rotation rx: {np.degrees(rot_vec[0]):.5f}°")
    print(f"  rotation ry: {np.degrees(rot_vec[1]):.5f}°")
    print(f"  rotation rz: {np.degrees(rot_vec[2]):.5f}°")
    print(f"  Saved to: {SPAWN_JSON_PATH}")
    print("\n--- Initial coupling & range usage ---")
    print(f"  eta_clean   : {eta_info['eta_clean']:.4f}  ({eta_info['eta_clean']*100:.2f}%)")
    print(f"  eta_lateral : {eta_info['eta_lateral']:.4f}  ({eta_info['eta_lateral']*100:.2f}%)  >1%? {'✓' if eta_info['eta_lateral_above_1pct'] else '✗'}")
    print(f"  eta_angular : {eta_info['eta_angular']:.4f}  ({eta_info['eta_angular']*100:.2f}%)  >1%? {'✓' if eta_info['eta_angular_above_1pct'] else '✗'}")
    print(f"  Distance % of max Euclidean range: {eta_info['dist_perc']:.2f}%")
    print(f"  Angle %    of max Euclidean range: {eta_info['angle_perc']:.2f}%")
    return dx_m, dy_m, dz_m, q

# ============================================================
# 5. ACTION JSON (joint limits and toggles)
# ============================================================
def update_action_json():
    if not ACTION_JSON_PATH.exists():
        print(f"⚠️ Action JSON not found: {ACTION_JSON_PATH}")
        print("   Creating an empty action list...")
        data = {"action_list": []}
    else:
        with open(ACTION_JSON_PATH, "r") as f:
            data = json.load(f)

    modified_actions = 0
    removed_joints = 0
    added_joints = 0
    modified_joints = 0

    print("\n--- Action JSON updated ---")
    print(f"Translation limit : ±{TRANS_LIMIT_MM*1000:.2f} µm")
    print(f"Rotation limit    : ±{ROT_LIMIT_DEG:.3f}°")
    print("Joint toggles:")
    for name, enabled in ENABLED_JOINTS.items():
        print(f"  {name:<12} : {'ENABLED' if enabled else 'REMOVED'}")

    for action in data.get("action_list", []):
        request = action.get("request")
        if not isinstance(request, dict):
            continue

        joints = request.get("active_joints")
        if not isinstance(joints, list):
            continue

        new_joints = []
        existing_joint_dict = {j.get("joint_name"): j for j in joints if "joint_name" in j}

        for joint_name in ALL_JOINTS:
            if not ENABLED_JOINTS.get(joint_name, False):
                if joint_name in existing_joint_dict:
                    removed_joints += 1
                    print(f"  Removing {joint_name} from action '{action['name']}'")
                continue

            if joint_name in existing_joint_dict:
                new_joint = existing_joint_dict[joint_name].copy()
            else:
                new_joint = {
                    "joint_name": joint_name,
                    "has_constraint": True,
                    "upper_limit": 0.0,
                    "lower_limit": 0.0
                }
                added_joints += 1
                print(f"  Adding {joint_name} to action '{action['name']}'")

            if joint_name in TRANSLATION_JOINTS:
                new_joint["upper_limit"] = TRANS_LIMIT_M
                new_joint["lower_limit"] = TRANS_LIMIT_M
                modified_joints += 1
            elif joint_name in ROTATION_JOINTS:
                new_joint["upper_limit"] = ROT_LIMIT_RAD
                new_joint["lower_limit"] = ROT_LIMIT_RAD
                modified_joints += 1

            new_joints.append(new_joint)

        request["active_joints"] = new_joints
        modified_actions += 1

    with open(ACTION_JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nModified actions : {modified_actions}")
    print(f"Added joints     : {added_joints}")
    print(f"Modified joints  : {modified_joints}")
    print(f"Removed joints   : {removed_joints}")
    print(f"Saved to         : {ACTION_JSON_PATH}")

# ============================================================
# 6. WRITE CONFIG JSON – now with dynamic initial power
# ============================================================
def write_config_json(seed, initial_power_percent):
    config = {
        'W0': W0,
        'LAMBDA': LAMBDA,
        'SEARCH_FACTOR': SEARCH_FACTOR,
        'THETA_EFF': float(THETA_EFF),
        'MODE': MODE,
        'ENABLED_JOINTS': ENABLED_JOINTS,
        'RANDOM_SEED': seed,
        'MAGNITUDE_OFSET_M': MAGNITUDE_OFSET_M,
        'CUSTOM_MAGNITUDE_ANGLE_DEG': CUSTOM_MAGNITUDE_ANGLE_DEG,
        'INITIAL_POWER_PERCENT': initial_power_percent,
        'trans_limit': TRANS_LIMIT_M,
        'rot_limit': ROT_LIMIT_RAD,
    }
    config_path = SCRIPT_DIR / "alignment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📄 Configuration overwritten at {config_path}")
    print(f"   Initial power percent: {initial_power_percent:.4f}%")

# ============================================================
# 7. WRITE RING INFO JSON
# ============================================================
def write_ring_info(poses, power_percent):
    info = {
        "power_percent": power_percent,
        "eta_target": power_percent / 100.0,
        "num_seeds": len(poses),
        "base_seed": poses[0]["seed"] if poses else None,
        "n_trans_enabled": poses[0]["n_trans_enabled"] if poses else 0,
        "n_rot_enabled": poses[0]["n_rot_enabled"] if poses else 0,
        "poses": poses
    }
    with open(RING_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)
    print(f"📄 Ring info saved to {RING_INFO_PATH}")

# ============================================================
# 8. SEED GENERATION for multi‑seed batch testing
# ============================================================
def generate_seed_poses(power_levels, seeds_per_level, base_seed, n_trans, n_rot):
    all_groups = []
    for power in power_levels:
        group = {"power_percent": power, "seeds": []}
        for i in range(seeds_per_level):
            seed = base_seed + int(power * 100) + i
            dx_m, dy_m, dz_m, quat, d, theta, phi = generate_ring_pose(
                seed, power/100.0, n_trans, n_rot)
            group["seeds"].append({
                "seed": seed,
                "dx_m": dx_m,
                "dy_m": dy_m,
                "dz_m": dz_m,
                "quat": quat,
                "d": d,
                "theta": theta,
                "phi": phi
            })
        all_groups.append(group)
    with open(SEED_POSES_PATH, "w") as f:
        json.dump(all_groups, f, indent=2)
    print(f"✅ Seed poses saved to {SEED_POSES_PATH}")
    return all_groups

# ============================================================
# 9. MAIN
# ============================================================
def main():
    global RING_SELECT_INDEX

    print("=" * 60)
    print("MASTER JSON EDITOR – Direct JSON Modification")
    print("=" * 60)
    print(f"Mode            = {MODE}")
    print(f"W0              = {W0:.2e} m")
    print(f"LAMBDA          = {LAMBDA:.2e} m")
    print(f"SEARCH_FACTOR   = {SEARCH_FACTOR}")
    print(f"THETA_EFF       = {np.degrees(THETA_EFF):.3f}°")

    # Determine active spawn mode
    spawn_mode = determine_spawn_mode()
    print(f"Active spawn    = {spawn_mode}")

    random_seed = random.randint(0, 2**32 - 1) if RANDOM else SEED
    print(f"Random seed     = {f'random ({random_seed})' if RANDOM else f'fixed ({SEED})'}")
    print(f"Randomise only enabled joints = {RANDOMIZE_ONLY_ENABLED_JOINTS}")
    print(f"Enabled translation axes: {n_trans_enabled}")
    print(f"Enabled rotation axes:    {n_rot_enabled}")
    print("\nEnabled joints:")
    for name, enabled in ENABLED_JOINTS.items():
        print(f"  {name:<12} : {'ENABLED' if enabled else 'REMOVED'}")
    print()

    # ---- Ring spawn handling (single pose for spawn_test_frames.json) ----
    ring_pose = None
    if USE_RING_SPAWN:
        print(f"🔘 RING SPAWN MODE: {RING_POWER_PERCENT}% power, {RING_NUM_SEEDS} seeds, base seed {RING_FIXED_SEED_BASE}")
        poses = generate_ring_poses(RING_NUM_SEEDS, RING_FIXED_SEED_BASE, RING_POWER_PERCENT,
                                    n_trans_enabled, n_rot_enabled)
        if RING_SAVE_JSON:
            write_ring_info(poses, RING_POWER_PERCENT)
        if RING_SELECT_INDEX >= len(poses):
            print(f"⚠️ RING_SELECT_INDEX ({RING_SELECT_INDEX}) out of range, using 0")
            RING_SELECT_INDEX = 0
        selected = poses[RING_SELECT_INDEX]
        ring_pose = {
            "dx_m": selected["dx_m"],
            "dy_m": selected["dy_m"],
            "dz_m": selected["dz_m"],
            "quat": selected["quat"]
        }
        print(f"  Selected seed {selected['seed']} for spawn JSON.")
        print(f"  d = {selected['d']*1e6:.2f} µm, theta = {np.degrees(selected['theta']):.3f}°")
        random_seed = selected["seed"]
    else:
        print("🔘 Using random/debug spawn mode.")

    # ---- Generate the spawn JSON and get its offsets ----
    dx_m, dy_m, dz_m, quat = write_spawn_json(seed=random_seed, ring_pose=ring_pose)

    # ---- Compute initial power from the actual offsets ----
    d_actual = np.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
    rot = R.from_quat([quat['X'], quat['Y'], quat['Z'], quat['W']])
    theta_actual = rot.magnitude()

    eta_lateral = np.exp(- (d_actual / W0)**2) if n_trans_enabled > 0 else 1.0
    eta_angular = np.exp(- (theta_actual / THETA_EFF)**2) if n_rot_enabled > 0 else 1.0
    eta_clean = eta_lateral * eta_angular
    initial_power_percent = eta_clean * 100.0

    # ---- Write the configuration JSON ----
    write_config_json(random_seed, initial_power_percent)

    # ---- Update the action JSON ----
    update_action_json()

    # ---- Generate seed poses for multi‑seed batch testing ----
    if GENERATE_SEED_POSES:
        print("\n🔘 Generating seed poses for batch testing...")
        all_seeds = generate_seed_poses(
            power_levels=SEED_POWER_LEVELS,
            seeds_per_level=SEEDS_PER_LEVEL,
            base_seed=SEED_BASE,
            n_trans=n_trans_enabled,
            n_rot=n_rot_enabled
        )
        total = sum(len(g['seeds']) for g in all_seeds)
        print(f"  Generated {total} total seeds across {len(all_seeds)} power levels.")
        print(f"  Power levels: {SEED_POWER_LEVELS} with {SEEDS_PER_LEVEL} seeds each.")
    else:
        print("\n🔘 Seed generation disabled (GENERATE_SEED_POSES=False).")

    print("\n" + "=" * 60)
    print("✅ COMPLETE – All JSON files are now up‑to‑date.")
    print("=" * 60)

if __name__ == "__main__":
    main()