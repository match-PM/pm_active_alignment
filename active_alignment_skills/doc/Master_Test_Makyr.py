#!/usr/bin/env python3
"""
master_json_editor.py – 5‑DOF version (X, Y, Z, A, B)

Central configuration for alignment experiments.
Directly modifies:
  1. spawn_test_frames.json  – random misalignment
  2. test_active_alignment.rsap.json – joint limits and active joint toggles
  3. seed_poses.json         – multiple seed poses for batch testing (random)

Signal model (5‑DOF, ignoring Rz):
  η = exp(-(dx²+dy²)/W0²)
      * exp(-(π n W0 / λ)² * (θx² + θy²))
      * 1/(1 + (dz/z_R)²)

C‑axis is permanently disabled and never used.
"""

import json
import random
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import math

# ============================================================
# 1. CENTRAL CONFIGURATION – EDIT THESE VALUES
# ============================================================

# ---- Physical parameters ----
W0 = 5.2e-6               # Mode field radius [m] (SMF-28 @ 1550nm)
LAMBDA = 1.55e-6          # Wavelength [m]
N_INDEX = 1.0             # Refractive index of medium (air)
SEARCH_FACTOR = 2.0       # s (search range = ±SEARCH_FACTOR * w0 or theta_eff)

# ---- Experiment mode (overrides manual joint toggles) ----
# Options: "5D", "2D_TRANS", "2D_ROT", "3D_TRANS", "3D_ROT"
MODE = "5D"  # Set to None to use manual joint toggles

# ---- Joint toggles (manual override, only used if MODE is None) ----
ENABLED_JOINTS_MANUAL = {
    "SP_X_Joint": True,
    "SP_Y_Joint": True,
    "SP_Z_Joint": True,
    "SP_A_Joint": True,
    "SP_B_Joint": True,
}

# ---- Randomisation controls ----
RANDOM = True
SEED = 1622
RANDOMIZE_ONLY_ENABLED_JOINTS = True

# ---- Spawner debug flags ----
debug = False
translatory_debug = False
angular_debug = False
search_debug = False
search_debug_5d = False

# ---- Custom offset ----
MAGNITUDE_OFSET_UM = 12
CUSTOM_MAGNITUDE_ANGLE_DEG = 3.0

# ---- SEED GENERATION ----
GENERATE_SEED_POSES = True
SEED_POWER_LEVELS = [2.0, 20.0, 60.0]   # target power percentages
SEEDS_PER_LEVEL = 3
SEED_BASE = 31248769

# ---- JSON file paths ----
SCRIPT_DIR = Path(__file__).parent
SPAWN_JSON_PATH = SCRIPT_DIR / "spawn_test_frames.json"
ACTION_JSON_PATH = SCRIPT_DIR / "test_active_alignment.rsap.json"
SEED_POSES_PATH = SCRIPT_DIR / "seed_poses.json"

# ============================================================
# 2. DERIVED QUANTITIES
# ============================================================
THETA_EFF = LAMBDA / (np.pi * W0)
THETA_EFF_DEG = np.degrees(THETA_EFF)
MAGNITUDE_OFSET_M = MAGNITUDE_OFSET_UM * 1e-6
Z_R = np.pi * N_INDEX * W0**2 / LAMBDA
Z_LIMIT_M = SEARCH_FACTOR * Z_R

TRANS_LIMIT_M = SEARCH_FACTOR * W0
ROT_LIMIT_RAD = SEARCH_FACTOR * THETA_EFF
ROT_LIMIT_DEG = np.degrees(ROT_LIMIT_RAD)

CUSTOM_DISTANCE_M = MAGNITUDE_OFSET_M
CUSTOM_ANGLE_RAD = np.radians(CUSTOM_MAGNITUDE_ANGLE_DEG)

ALL_JOINTS = ["SP_X_Joint", "SP_Y_Joint", "SP_Z_Joint", "SP_A_Joint", "SP_B_Joint"]
TRANSLATION_JOINTS = {"SP_X_Joint", "SP_Y_Joint", "SP_Z_Joint"}
ROTATION_JOINTS = {"SP_A_Joint", "SP_B_Joint"}

def get_enabled_joints(mode):
    if mode == "5D":
        return {j: True for j in ALL_JOINTS}
    elif mode == "2D_TRANS":
        return {"SP_X_Joint": True, "SP_Y_Joint": True,
                "SP_Z_Joint": False, "SP_A_Joint": False, "SP_B_Joint": False}
    elif mode == "2D_ROT":
        return {"SP_X_Joint": False, "SP_Y_Joint": False,
                "SP_Z_Joint": False, "SP_A_Joint": True, "SP_B_Joint": True}
    elif mode == "3D_TRANS":
        return {"SP_X_Joint": True, "SP_Y_Joint": True,
                "SP_Z_Joint": True, "SP_A_Joint": False, "SP_B_Joint": False}
    elif mode == "3D_ROT":
        return {"SP_X_Joint": False, "SP_Y_Joint": False,
                "SP_Z_Joint": False, "SP_A_Joint": True, "SP_B_Joint": True}
    else:
        raise ValueError(f"Unknown MODE: {mode}")

ENABLED_JOINTS = get_enabled_joints(MODE)

TRANS_AXIS_MAP = {"SP_X_Joint": 0, "SP_Y_Joint": 1, "SP_Z_Joint": 2}
ROT_AXIS_MAP = {"SP_A_Joint": 0, "SP_B_Joint": 1}

n_trans_enabled = sum(1 for j in TRANSLATION_JOINTS if ENABLED_JOINTS.get(j, False))
n_rot_enabled = sum(1 for j in ROTATION_JOINTS if ENABLED_JOINTS.get(j, False))
n_rot_tilt = n_rot_enabled

# ============================================================
# 3. COUPLING EFFICIENCY
# ============================================================
def compute_eta_and_percentages(dx_m, dy_m, dz_m, quat, n_trans, n_rot, n_rot_tilt):
    d_lateral = np.sqrt(dx_m**2 + dy_m**2)
    d_axial = np.abs(dz_m)
    r = R.from_quat([quat['X'], quat['Y'], quat['Z'], quat['W']])
    rot_vec = r.as_rotvec()
    theta_tilt = np.sqrt(rot_vec[0]**2 + rot_vec[1]**2)

    eta_lateral = np.exp(-(d_lateral / W0)**2) if n_trans > 0 else 1.0
    arg_ang = (np.pi * N_INDEX * W0 * theta_tilt) / LAMBDA
    eta_angular = np.exp(-arg_ang**2) if n_rot > 0 else 1.0
    eta_axial = 1.0 / (1.0 + (d_axial / Z_R)**2) if n_trans > 0 else 1.0
    eta_clean = eta_lateral * eta_angular * eta_axial

    # ---- Correct distance percentage ----
    if n_trans > 0:
        # Count enabled X/Y joints
        n_xy = sum(1 for j in ['SP_X_Joint', 'SP_Y_Joint'] if ENABLED_JOINTS.get(j, False))
        has_z = ENABLED_JOINTS.get('SP_Z_Joint', False)

        max_dist_sq = 0.0
        if n_xy > 0:
            max_dist_sq += n_xy * TRANS_LIMIT_M**2
        if has_z:
            max_dist_sq += Z_LIMIT_M**2

        max_total_dist = np.sqrt(max_dist_sq) if max_dist_sq > 0 else 1.0
        d_total = np.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
        dist_perc = (d_total / max_total_dist) * 100
    else:
        dist_perc = 0.0

    if n_rot_tilt > 0:
        max_angle = np.sqrt(n_rot_tilt) * ROT_LIMIT_RAD
        angle_perc = (theta_tilt / max_angle) * 100
    else:
        angle_perc = 0.0

    return {
        'eta_clean': eta_clean,
        'eta_lateral': eta_lateral,
        'eta_angular': eta_angular,
        'eta_axial': eta_axial,
        'dist_perc': dist_perc,
        'angle_perc': angle_perc,
        'eta_lateral_above_1pct': eta_lateral > 0.01,
        'eta_angular_above_1pct': eta_angular > 0.01,
        'eta_axial_above_1pct': eta_axial > 0.01,
    }

# ============================================================
# 4. SPAWN JSON (no C)
# ============================================================
def random_offset_m(limit_m: float) -> float:
    return random.uniform(-limit_m, limit_m)

def random_small_rotation(max_deg: float) -> dict:
    angle_deg = random.uniform(0.0, max_deg)
    phi = random.uniform(0, 2 * np.pi)
    rx = np.radians(angle_deg) * np.cos(phi)
    ry = np.radians(angle_deg) * np.sin(phi)
    rz = 0.0
    if np.linalg.norm([rx, ry, rz]) > 1e-12:
        rot = R.from_rotvec([rx, ry, rz])
    else:
        rot = R.from_quat([0, 0, 0, 1])
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

def build_spawn_json(seed: int | None = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    trans_enabled = [TRANS_AXIS_MAP[j] for j in TRANSLATION_JOINTS if ENABLED_JOINTS.get(j, False)]
    rot_enabled = {j: ENABLED_JOINTS.get(j, False) for j in ROTATION_JOINTS}
    n_trans_enabled = len(trans_enabled)

    if search_debug_5d:
        per_axis_dist = CUSTOM_DISTANCE_M / np.sqrt(3)
        dx_m = per_axis_dist
        dy_m = per_axis_dist
        dz_m = per_axis_dist
        per_axis_rot = CUSTOM_ANGLE_RAD / np.sqrt(2)
        rot_vec = [per_axis_rot, per_axis_rot, 0.0]
        rot = R.from_rotvec(rot_vec)
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
        n_rot_enabled = sum(1 for j in ROTATION_JOINTS if ENABLED_JOINTS.get(j, False))
        if n_rot_enabled > 0:
            per_axis_rot = CUSTOM_ANGLE_RAD / np.sqrt(n_rot_enabled)
            for ax in range(2):
                joint_name = list(ROT_AXIS_MAP.keys())[ax]
                if ENABLED_JOINTS.get(joint_name, False):
                    rot_vec[ax] = per_axis_rot
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

def write_spawn_json(seed: int | None = None):
    os.makedirs(SPAWN_JSON_PATH.parent, exist_ok=True)
    spawn = build_spawn_json(seed=seed)
    with open(SPAWN_JSON_PATH, "w") as f:
        json.dump(spawn, f, indent=2)

    Z_BASE = 90.0 - 32.62570
    trans = spawn['frames'][0]['transformation']['translation']
    q = spawn['frames'][0]['transformation']['rotation']
    dx_m = trans['X'] / 1000.0
    dy_m = trans['Y'] / 1000.0
    dz_m = (trans['Z'] - Z_BASE) / 1000.0

    eta_info = compute_eta_and_percentages(dx_m, dy_m, dz_m, q,
                                           n_trans_enabled, n_rot_enabled, n_rot_tilt)

    print("\n--- Spawn JSON written ---")
    print(f"  dx = {trans['X']*1000:.5f} µm")
    print(f"  dy = {trans['Y']*1000:.5f} µm")
    print(f"  dz = {(trans['Z']-Z_BASE)*1000:.5f} µm")
    print(f"  lateral: {np.sqrt(dx_m**2+dy_m**2)*1e6:.2f} µm")
    print(f"  axial: {abs(dz_m)*1e6:.2f} µm")
    print(f"  eta_lateral  = {eta_info['eta_lateral']:.6f}")
    print(f"  eta_angular  = {eta_info['eta_angular']:.6f}   ← this is what you care about in 2D_ROT")
    print(f"  eta_axial    = {eta_info['eta_axial']:.6f}")
    print(f"  eta_clean    = {eta_info['eta_clean']:.6f} → {eta_info['eta_clean']*100:.2f}%")
    # extra: show the tilt angle actually used
    r = R.from_quat([q['X'], q['Y'], q['Z'], q['W']])
    rot_vec = r.as_rotvec()
    theta_tilt = np.sqrt(rot_vec[0]**2 + rot_vec[1]**2)
    print(f"  theta_tilt   = {np.degrees(theta_tilt):.4f}°   (THETA_EFF = {np.degrees(THETA_EFF):.4f}°)")

    return dx_m, dy_m, dz_m, q, eta_info['eta_clean']

# ============================================================
# 5. ACTION JSON
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
    print(f"Translation X/Y limit : ±{TRANS_LIMIT_M*1e6:.2f} µm")
    print(f"Translation Z limit    : ±{Z_LIMIT_M*1e6:.2f} µm")
    print(f"Rotation limit         : ±{ROT_LIMIT_DEG:.3f}°")

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
                new_joint = {"joint_name": joint_name, "has_constraint": True,
                             "upper_limit": 0.0, "lower_limit": 0.0}
                added_joints += 1
                print(f"  Adding {joint_name} to action '{action['name']}'")

            if joint_name == "SP_Z_Joint":
                new_joint["upper_limit"] = Z_LIMIT_M
                new_joint["lower_limit"] = Z_LIMIT_M
                modified_joints += 1
            elif joint_name in TRANSLATION_JOINTS:   # X, Y
                new_joint["upper_limit"] = TRANS_LIMIT_M
                new_joint["lower_limit"] = TRANS_LIMIT_M
                modified_joints += 1
            elif joint_name in ROTATION_JOINTS:      # A, B
                new_joint["upper_limit"] = ROT_LIMIT_RAD
                new_joint["lower_limit"] = ROT_LIMIT_RAD
                modified_joints += 1

            new_joints.append(new_joint)

        request["active_joints"] = new_joints
        modified_actions += 1

    for action in data.get("action_list", []):
        request = action.get("request")
        if not isinstance(request, dict):
            continue
        joints = request.get("active_joints")
        if not isinstance(joints, list):
            continue
        new_joints = [j for j in joints if j.get("joint_name") != "SP_C_Joint"]
        if len(new_joints) != len(joints):
            removed_joints += len(joints) - len(new_joints)
            request["active_joints"] = new_joints
            print(f"  Removed SP_C_Joint from action '{action['name']}'")

    with open(ACTION_JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Modified actions: {modified_actions}, Added: {added_joints}, "
          f"Modified: {modified_joints}, Removed: {removed_joints}")

# ============================================================
# 6. WRITE CONFIG JSON
# ============================================================
def write_config_json(seed, initial_power_percent):
    config = {
        'W0': W0,
        'LAMBDA': LAMBDA,
        'N_INDEX': N_INDEX,
        'SEARCH_FACTOR': SEARCH_FACTOR,
        'THETA_EFF': float(THETA_EFF),
        'Z_R': float(Z_R),
        'MODE': MODE,
        'ENABLED_JOINTS': ENABLED_JOINTS,
        'RANDOM_SEED': seed,
        'INITIAL_POWER_PERCENT': initial_power_percent,
        'trans_limit': TRANS_LIMIT_M,
        'rot_limit': ROT_LIMIT_RAD,
        'Z_LIMIT': float(Z_LIMIT_M),
    }
    config_path = SCRIPT_DIR / "alignment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📄 Configuration overwritten at {config_path}")
    print(f"   Initial power percent: {initial_power_percent:.4f}%")

# ============================================================
# 7. SEED GENERATION (targeted eta)
# ============================================================
def generate_pose_for_target_eta(eta_target, n_trans, n_rot, enabled_joints, verbose=True):
    """
    Generate a random pose (dx, dy, dz, quat) that yields exactly eta_target,
    while ensuring that all misalignments are within the search limits.
    If eta_target is lower than the minimum possible eta inside the box (eta_corner),
    the function randomly picks a target between eta_corner and a configurable upper bound
    (LOW_SIGNAL_UPPER_BOUND, default 0.001 = 0.1%) to provide variety.
    If eta_corner is already > LOW_SIGNAL_UPPER_BOUND, it uses eta_corner.
    """
    has_lateral = (enabled_joints.get('SP_X_Joint', False) or
                   enabled_joints.get('SP_Y_Joint', False))
    has_axial = enabled_joints.get('SP_Z_Joint', False)
    has_angular = (enabled_joints.get('SP_A_Joint', False) or
                   enabled_joints.get('SP_B_Joint', False))

    if not (has_lateral or has_axial or has_angular):
        return 0.0, 0.0, 0.0, {"X": 0.0, "Y": 0.0, "Z": 0.0, "W": 1.0}, 0.0, 0.0

    # Compute corner misalignments
    if has_lateral:
        d_lat_corner = math.sqrt(2) * TRANS_LIMIT_M
    else:
        d_lat_corner = 0.0
    if has_axial:
        d_ax_corner = Z_LIMIT_M
    else:
        d_ax_corner = 0.0
    if has_angular:
        theta_tilt_corner = math.sqrt(2) * ROT_LIMIT_RAD
    else:
        theta_tilt_corner = 0.0

    eta_corner = 1.0
    if n_trans > 0:
        eta_corner *= math.exp(-(d_lat_corner / W0)**2)
        eta_corner *= 1.0 / (1.0 + (d_ax_corner / Z_R)**2)
    if n_rot > 0:
        eta_corner *= math.exp(-(theta_tilt_corner / THETA_EFF)**2)

    # Determine the final target eta
    final_eta_target = eta_target
    low_signal_chosen = False

    # If the requested target is below the corner
    LOW_SIGNAL_UPPER_BOUND = 0.001  # 0.1% (change this to your preference)
    if eta_target <= eta_corner * (1 + 1e-12):
        if eta_corner >= LOW_SIGNAL_UPPER_BOUND:
            # Corner is already above the upper bound – use the corner
            final_eta_target = eta_corner
            low_signal_chosen = False  # it's the corner, not a random low signal
        else:
            # Pick a random eta between corner and the upper bound
            final_eta_target = random.uniform(eta_corner, LOW_SIGNAL_UPPER_BOUND)
            low_signal_chosen = True

    # Logging
    # if verbose:
    #     print("\n" + "="*60)
    #     print("SEED GENERATION DEBUG")
    #     print(f"  Requested eta: {eta_target:.6g}  ({eta_target*100:.6f}%)")
    #     print(f"  Corner eta (minimum inside box): {eta_corner:.6g}  ({eta_corner*100:.6f}%)")
    #     if low_signal_chosen:
    #         print(f"  Chose random low signal between {eta_corner:.6g} and {LOW_SIGNAL_UPPER_BOUND:.6g}")
    #     elif final_eta_target != eta_target:
    #         print(f"  Fell back to corner (target below corner)")
    #     print(f"  Final eta used: {final_eta_target:.6g}  ({final_eta_target*100:.6f}%)")

    # Rejection sampling for final_eta_target (now guaranteed to be >= eta_corner)
    components = []
    if has_lateral: components.append('lateral')
    if has_axial: components.append('axial')
    if has_angular: components.append('angular')

    for attempt in range(100):
        if len(components) == 1:
            weights = [1.0]
        else:
            raw = [np.random.gamma(1, 1) for _ in components]
            weights = [r / sum(raw) for r in raw]

        eta_components = {}
        for comp, w in zip(components, weights):
            eta_components[comp] = final_eta_target ** w

        dx, dy, dz = 0.0, 0.0, 0.0
        rx, ry, rz = 0.0, 0.0, 0.0

        if has_lateral:
            eta_lat = eta_components['lateral']
            d_lat = W0 * math.sqrt(-math.log(eta_lat))
            phi = random.uniform(0, 2 * math.pi)
            dx = d_lat * math.cos(phi)
            dy = d_lat * math.sin(phi)
            if not enabled_joints.get('SP_X_Joint', False): dx = 0.0
            if not enabled_joints.get('SP_Y_Joint', False): dy = 0.0
            if abs(dx) > TRANS_LIMIT_M or abs(dy) > TRANS_LIMIT_M:
                continue

        if has_axial:
            eta_ax = eta_components['axial']
            d_axial = Z_R * math.sqrt(1.0 / eta_ax - 1.0)
            dz = d_axial * (1 if random.random() < 0.5 else -1)
            if abs(dz) > Z_LIMIT_M:
                continue

        if has_angular:
            eta_ang = eta_components['angular']
            theta_tilt = THETA_EFF * math.sqrt(-math.log(eta_ang))
            phi_rot = random.uniform(0, 2 * math.pi)
            rx = theta_tilt * math.cos(phi_rot)
            ry = theta_tilt * math.sin(phi_rot)
            if not enabled_joints.get('SP_A_Joint', False): rx = 0.0
            if not enabled_joints.get('SP_B_Joint', False): ry = 0.0
            if abs(rx) > ROT_LIMIT_RAD or abs(ry) > ROT_LIMIT_RAD:
                continue

        rot_vec = [rx, ry, 0.0]
        if np.linalg.norm(rot_vec) > 1e-12:
            rot = R.from_rotvec(rot_vec)
        else:
            rot = R.from_quat([0, 0, 0, 1])
        quat_arr = rot.as_quat()
        quat = {"X": float(quat_arr[0]), "Y": float(quat_arr[1]),
                "Z": float(quat_arr[2]), "W": float(quat_arr[3])}
        d_total = math.sqrt(dx**2 + dy**2 + dz**2)
        theta_total = math.sqrt(rx**2 + ry**2 + rz**2)

        # Log the final pose
        if verbose:
            print(f"  Final offsets:")
            print(f"    dx = {dx*1e6:.3f} µm, dy = {dy*1e6:.3f} µm, dz = {dz*1e6:.3f} µm")
            print(f"    rx = {math.degrees(rx):.4f}°, ry = {math.degrees(ry):.4f}°")
            print(f"    quaternion: {quat}")
            print(f"    Total distance: {d_total*1e6:.3f} µm")
            print(f"    Total tilt: {math.degrees(theta_total):.4f}°")
            print("="*60 + "\n")
        return dx, dy, dz, quat, d_total, theta_total

    # Fallback to corner if all attempts fail
    if verbose:
        print("  ⚠️ Rejection sampling failed after 100 attempts. Falling back to corner pose.")
    # Directly construct the corner pose (random signs)
    sign_x = 1 if random.random() < 0.5 else -1
    sign_y = 1 if random.random() < 0.5 else -1
    sign_z = 1 if random.random() < 0.5 else -1
    dx = sign_x * TRANS_LIMIT_M if has_lateral else 0.0
    dy = sign_y * TRANS_LIMIT_M if has_lateral else 0.0
    dz = sign_z * Z_LIMIT_M if has_axial else 0.0
    sign_a = 1 if random.random() < 0.5 else -1
    sign_b = 1 if random.random() < 0.5 else -1
    rx = sign_a * ROT_LIMIT_RAD if has_angular else 0.0
    ry = sign_b * ROT_LIMIT_RAD if has_angular else 0.0
    rz = 0.0
    rot_vec = [rx, ry, rz]
    if np.linalg.norm(rot_vec) > 1e-12:
        rot = R.from_rotvec(rot_vec)
    else:
        rot = R.from_quat([0, 0, 0, 1])
    quat_arr = rot.as_quat()
    quat = {"X": float(quat_arr[0]), "Y": float(quat_arr[1]),
            "Z": float(quat_arr[2]), "W": float(quat_arr[3])}
    d_total = math.sqrt(dx**2 + dy**2 + dz**2)
    theta_total = math.sqrt(rx**2 + ry**2 + rz**2)

    if verbose:
        print(f"  Corner fallback offsets:")
        print(f"    dx = {dx*1e6:.3f} µm, dy = {dy*1e6:.3f} µm, dz = {dz*1e6:.3f} µm")
        print(f"    rx = {math.degrees(rx):.4f}°, ry = {math.degrees(ry):.4f}°")
        print(f"    quaternion: {quat}")
        print(f"    Total distance: {d_total*1e6:.3f} µm")
        print(f"    Total tilt: {math.degrees(theta_total):.4f}°")
        print("="*60 + "\n")
    return dx, dy, dz, quat, d_total, theta_total

def generate_seed_poses(power_levels, seeds_per_level, base_seed, enabled_joints):
    all_groups = []
    for power in power_levels:
        eta_target = power / 100.0
        group = {"power_percent": power, "seeds": []}
        for i in range(seeds_per_level):
            seed = base_seed + i
            np.random.seed(seed)
            random.seed(seed)
            dx_m, dy_m, dz_m, quat, d, theta = generate_pose_for_target_eta(
                eta_target, n_trans_enabled, n_rot_enabled, enabled_joints)
            group["seeds"].append({
                "seed": seed,
                "dx_m": dx_m,
                "dy_m": dy_m,
                "dz_m": dz_m,
                "quat": quat,
                "d": d,
                "theta": theta,
            })
        all_groups.append(group)
    with open(SEED_POSES_PATH, "w") as f:
        json.dump(all_groups, f, indent=2)
    print(f"✅ Seed poses saved to {SEED_POSES_PATH}")
    return all_groups

# ============================================================
# 8. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("MASTER JSON EDITOR – 5‑DOF (X, Y, Z, A, B) – No C axis")
    print("=" * 60)
    print(f"Mode            = {MODE}")
    print(f"W0              = {W0:.2e} m")
    print(f"LAMBDA          = {LAMBDA:.2e} m")
    print(f"N_INDEX         = {N_INDEX} (air)")
    print(f"SEARCH_FACTOR   = {SEARCH_FACTOR}")
    print(f"THETA_EFF       = {np.degrees(THETA_EFF):.3f}°")
    print(f"Z_R             = {Z_R*1e6:.2f} µm")

    random_seed = random.randint(0, 2**32 - 1) if RANDOM else SEED
    print(f"Random seed     = {f'random ({random_seed})' if RANDOM else f'fixed ({SEED})'}")
    print(f"Enabled translation axes: {n_trans_enabled}")
    print(f"Enabled rotation axes (A+B): {n_rot_enabled}")

    # Write spawn JSON and get its initial power
    dx_m, dy_m, dz_m, quat, spawn_eta = write_spawn_json(seed=random_seed)
    initial_power_percent = spawn_eta * 100.0

    # Write config with the spawn's power
    write_config_json(random_seed, initial_power_percent)

    # Update action JSON (joint limits)
    update_action_json()

    # Generate seed poses for batch (if enabled)
    if GENERATE_SEED_POSES:
        print("\n🔘 Generating random seed poses for batch testing...")
        all_seeds = generate_seed_poses(
            power_levels=SEED_POWER_LEVELS,
            seeds_per_level=SEEDS_PER_LEVEL,
            base_seed=SEED_BASE,
            enabled_joints=ENABLED_JOINTS
        )
        total = sum(len(g['seeds']) for g in all_seeds)
        print(f"  Generated {total} seeds across {len(all_seeds)} power levels.")
        print(f"  Power levels: {SEED_POWER_LEVELS} with {SEEDS_PER_LEVEL} seeds each.")
    else:
        print("\n🔘 Seed generation disabled.")

    print("\n" + "=" * 60)
    print("✅ COMPLETE – All JSON files updated (5‑DOF).")
    print("=" * 60)

if __name__ == "__main__":
    main()