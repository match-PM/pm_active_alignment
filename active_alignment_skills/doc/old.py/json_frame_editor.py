#!/usr/bin/env python3
"""
Erzeugt eine randomisierte spawn_test_frames.json innerhalb des
gueltigen Fangradius (Faktor 3 * w0 / theta_eff aus dem Gauss-
Kopplungsmodell), damit jeder Testlauf eine andere, aber realistische
Anfangsfehllage hat.

WICHTIG fuer fairen Algorithmenvergleich: Wenn du mehrere Optimierer
(Hill-Climb, Nelder-Mead, ...) miteinander vergleichen willst, MUSS
jeder Algorithmus mit derselben Fehllage starten - sonst vergleichst du
nicht die Algorithmen, sondern verschiedene Probleme. Benutze dafuer
denselben seed pro Vergleichsdurchlauf (siehe main unten).
"""

import json
import random
import numpy as np
from scipy.spatial.transform import Rotation as R

# ============================================================
# 1. PHYSICAL PARAMETERS (all in SI units: meters, radians)
# ============================================================
W0 = 5.2e-6               # Mode field radius [m] (SMF-28 @ 1550nm)
LAMBDA = 1.55e-6          # Wavelength [m]
THETA_EFF = LAMBDA / (np.pi * W0)   # Angular tolerance [rad]
THETA_EFF_DEG = np.degrees(THETA_EFF)  # Angular tolerance [deg]
R_EFF = 2.146 #threshold of eta > 0.01 (1%) for the Gaussian coupling model, derived from Marcuse's formula
# ============================================================
# 2. SEARCH BOUNDS
# ============================================================
SEARCH_FACTOR = 3        # sigma (search range = ±CAPTURE_FACTOR * w0 or theta_eff)

# ---- Translation limit [m] ----
TRANS_LIMIT_M = SEARCH_FACTOR * W0
TRANS_LIMIT_MM = TRANS_LIMIT_M * 1000   # [mm]

# ---- Rotation limit [rad] and [deg] ----
ROT_LIMIT_RAD = SEARCH_FACTOR * THETA_EFF
ROT_LIMIT_DEG = np.degrees(ROT_LIMIT_RAD)

# ============================================================
# 3. DEBUG / OVERRIDE CONTROLS
# ============================================================
debug = False           # if True: perfect alignment (no randomisation)
translatory_debug = False  # if True: no translation randomisation, only rotation
angular_debug = False      # if True: no rotation randomisation, only translation
search_debug = False        # if True: use custom fixed offset (for testing)
RANDOM = True              # if True: use random seed; if False: fixed seed = 15

# ---- Custom offset for search_debug and testing (in METERS) ----
MAGNITUDE_OFSET_M = 5.2e-6  # Magnitude of translation offset in meters (adjust as needed)
 
CUSTOM_ANGLE_DEG = 0.0  # Magnitude of rotation in degrees (adjust as needed)

#in correct units for calculation
CUSTOM_DISTANCE_M = MAGNITUDE_OFSET_M / np.sqrt(3)    
CUSTOM_ANGLE_RAD = np.radians(CUSTOM_ANGLE_DEG)  # 0.1 rad ≈ 5.73° (adjust as needed)

SPAWN_JSON_PATH = "/home/brotato211/ros2_humble/src/pm_active_alignment/active_alignment_skills/doc/spawn_test_frames.json"

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================
def random_offset_m(limit_m: float) -> float:
    """Uniform offset in [-limit_m, +limit_m] (meters)."""
    return random.uniform(-limit_m, limit_m)

def random_offset_mm(limit_mm: float) -> float:
    """Uniform offset in [-limit_mm, +limit_mm] (millimeters)."""
    return random.uniform(-limit_mm, limit_mm)

def random_small_rotation(max_deg: float) -> dict:
    """
    Random small rotation about a random axis, magnitude uniformly
    distributed in [0, max_deg]. Returns a quaternion dict
    matching the JSON format (X, Y, Z, W).
    """
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle_deg = random.uniform(0.0, max_deg)
    rot = R.from_rotvec(np.radians(angle_deg) * axis)
    q = rot.as_quat()  # scipy: [x, y, z, w]
    return {"X": float(q[0]), "Y": float(q[1]), "Z": float(q[2]), "W": float(q[3])}

# ============================================================
# 5. BUILD SPAWN JSON
# ============================================================
def build_spawn_json(seed: int | None = None) -> dict:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ---- Determine offset values based on debug flags ----
    if debug:
        # Perfect alignment
        dx_m = 0.0
        dy_m = 0.0
        dz_m = 0.0
        quat = {"X": 0.0, "Y": 0.0, "Z": 0.0, "W": 1.0}

    elif translatory_debug:
        # Only rotation, no translation
        dx_m = 0.0
        dy_m = 0.0
        dz_m = 0.0
        quat = random_small_rotation(ROT_LIMIT_DEG)

    elif angular_debug:
        # Only translation, no rotation
        dx_m = random_offset_m(TRANS_LIMIT_M)
        dy_m = random_offset_m(TRANS_LIMIT_M)
        dz_m = random_offset_m(TRANS_LIMIT_M)
        quat = {"X": 0.0, "Y": 0.0, "Z": 0.0, "W": 1.0}

    elif search_debug:
        # Fixed custom offset (for testing specific misalignments)
        dx_m = CUSTOM_DISTANCE_M
        dy_m = CUSTOM_DISTANCE_M
        dz_m = CUSTOM_DISTANCE_M
        # Create a rotation about X, Y, Z axes by CUSTOM_ANGLE_DEG as total magnitude
        custom_offset = CUSTOM_ANGLE_RAD / np.sqrt(3) # the rotation vector magnitude is divided equally among x, y, z for a small rotation measured in degrees of CUSTOM_ANGLE_DEG. This is a simplification for testing purposes.
        rot = R.from_rotvec([custom_offset, custom_offset, custom_offset])
        q = rot.as_quat()
        quat = {"X": float(q[0]), "Y": float(q[1]), "Z": float(q[2]), "W": float(q[3])}

    else:
        # Full random within capture range
        dx_m = random_offset_m(TRANS_LIMIT_M)
        dy_m = random_offset_m(TRANS_LIMIT_M)
        dz_m = random_offset_m(TRANS_LIMIT_M)
        quat = random_small_rotation(ROT_LIMIT_DEG)

    # Convert translation from meters to millimeters (JSON expects mm)
    dx_mm = dx_m * 1000
    dy_mm = dy_m * 1000
    dz_mm = dz_m * 1000

    # The Z offset includes a fixed offset (90.0 - 32.62570) to position the target correctly and based of world coordinates.
    Z_BASE = 90.0 - 32.62570   # mm (adjust this value as needed for your setup)
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

# ============================================================
# 6. WRITE JSON
# ============================================================
def write_spawn_json(seed: int | None = None, path: str = SPAWN_JSON_PATH):
    spawn = build_spawn_json(seed=seed)
    with open(path, "w") as f:
        json.dump(spawn, f, indent=2)
    Z_BASE = 90.0 - 32.62570
    # Print summary
    print(f"Spawn-JSON geschrieben nach {path}")
    trans = spawn['frames'][0]['transformation']['translation']
    print(f"  dx={trans['X']*1000:.5f} µm")
    print(f"  dy={trans['Y']*1000:.5f} µm")
    print(f"  dz={trans['Z']*1000:.5f} µm")
    print(f"  rotation quaternion: {spawn['frames'][0]['transformation']['rotation']}")
    print(f"  distance magnitude: {np.sqrt(trans['X']**2 + trans['Y']**2 + (trans['Z']-Z_BASE)**2)*1000:.5f} µm")
    print(f"  rotation magnitude: {np.degrees(R.from_quat([spawn['frames'][0]['transformation']['rotation'][k] for k in ['X', 'Y', 'Z', 'W']]).magnitude()):.5f}°")
    return spawn

# ============================================================
# 7. MAIN
# ============================================================
if __name__ == "__main__":
    # Generate a random seed for reproducibility
    random_seed = random.randint(0, 2**32 - 1)

    if RANDOM:
        set_seed = random_seed
    else:
        set_seed = 15

    print(f"Using seed: {set_seed}")
    print(f"Translation limit: ±{TRANS_LIMIT_MM*1000:.2f} µm")
    print(f"Rotation limit:    ±{ROT_LIMIT_DEG:.3f}°")
    print(f"Search factor:     {SEARCH_FACTOR}σ")
    print()

    write_spawn_json(seed=set_seed)