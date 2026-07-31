#!/usr/bin/env python3

import json
from pathlib import Path
import numpy as np

# ==========================================================
# Alignment Constraint JSON Editor with Joint Toggles
# ==========================================================
#
# Changes:
#   - upper_limit / lower_limit (if joint is enabled)
#   - Removes joints that are set to False
#
# Usage: Set the ENABLED_JOINTS dictionary below.
#   True  = joint is active and will be in the action
#   False = joint is removed from the action
# ==========================================================

# Name der JSON-Datei im selben Ordner
JSON_FILE = "test_active_alignment.rsap.json"

# -----------------------------
# JOINT TOGGLES
# Select one mode and adjust ENABLED_JOINTS as needed.
# -----------------------------
trans_2d = True
rot_2d = False
six_d = False

if trans_2d:
    ENABLED_JOINTS = {
        "SP_X_Joint": True,
        "SP_Y_Joint": True,
        "SP_Z_Joint": False,  # Disable Z translation for 2D translation
        "SP_A_Joint": False,  # Disable A rotation for 2D translation
        "SP_B_Joint": False,  # Disable B rotation for 2D translation
        "SP_C_Joint": False,  # Disable C rotation for 2D translation
    }

elif rot_2d:
    ENABLED_JOINTS = {
        "SP_X_Joint": False,  # Disable X translation for 2D rotation
        "SP_Y_Joint": False,  # Disable Y translation for 2D rotation
        "SP_Z_Joint": False,  # Disable Z translation for 2D rotation
        "SP_A_Joint": True,
        "SP_B_Joint": True,
        "SP_C_Joint": False,  # Disable C rotation for 2D rotation
    }

elif six_d:
    ENABLED_JOINTS = {
        "SP_X_Joint": True,
        "SP_Y_Joint": True,
        "SP_Z_Joint": True,
        "SP_A_Joint": True,
        "SP_B_Joint": True,
        "SP_C_Joint": True,
    }

else:
    raise ValueError("Select one of trans_2d, rot_2d, or six_d to be True")

# -----------------------------
# LIMIT PARAMETERS
# -----------------------------
SEARCH_FACTOR = 3
W0 = 5.2e-6  # Modenfeldradius in Metern (SMF-28 @ 1550nm)
THETA_EFF = 1.55e-6 / (np.pi * W0)  # Winkeltoleranz [rad]
TRANSLATION_LIMIT = SEARCH_FACTOR * W0 
ROTATION_LIMIT = SEARCH_FACTOR * THETA_EFF 

# -----------------------------
# JOINT GROUPS (for limit assignment)
# -----------------------------
TRANSLATION_JOINTS = {
    "SP_X_Joint",
    "SP_Y_Joint",
    "SP_Z_Joint"
}

ROTATION_JOINTS = {
    "SP_A_Joint",
    "SP_B_Joint",
    "SP_C_Joint"
}


def main():
    json_path = Path(__file__).parent / JSON_FILE

    with open(json_path, "r") as f:
        data = json.load(f)

    modified_actions = 0
    removed_joints = 0
    modified_joints = 0

    print("=" * 60)
    print("Alignment Constraint JSON Editor (with Joint Toggles)")
    print("=" * 60)
    print(f"Translation limit : {TRANSLATION_LIMIT}")
    print(f"Rotation limit    : {ROTATION_LIMIT}")
    print()
    print("Joint toggles:")
    for name, enabled in ENABLED_JOINTS.items():
        print(f"  {name:<12} : {'ENABLED' if enabled else 'REMOVED'}")
    print()

    for action in data.get("action_list", []):
        request = action.get("request")
        if not isinstance(request, dict):
            continue

        joints = request.get("active_joints")
        if not isinstance(joints, list):
            continue

        if len(joints) == 0:
            continue

        modified_actions += 1

        # ---- Filter joints based on toggles ----
        new_joints = []
        for joint in joints:
            name = joint.get("joint_name")

            # If this joint is disabled, skip it (remove from action)
            if name in ENABLED_JOINTS and not ENABLED_JOINTS[name]:
                removed_joints += 1
                print(f"  Removing {name} from action '{action['name']}'")
                continue

            # If joint is enabled, update its limits
            if name in TRANSLATION_JOINTS:
                joint["upper_limit"] = TRANSLATION_LIMIT
                joint["lower_limit"] = -TRANSLATION_LIMIT
                modified_joints += 1
                print(f"  {name:<12} -> upper={TRANSLATION_LIMIT:.10g}, lower={-TRANSLATION_LIMIT:.10g}")

            elif name in ROTATION_JOINTS:
                joint["upper_limit"] = ROTATION_LIMIT
                joint["lower_limit"] = -ROTATION_LIMIT
                modified_joints += 1
                print(f"  {name:<12} -> upper={ROTATION_LIMIT:.10g}, lower={-ROTATION_LIMIT:.10g}")

            # Keep the joint
            new_joints.append(joint)

        # Replace the old joint list with the filtered one
        request["active_joints"] = new_joints
        print()

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print("=" * 60)
    print("Finished.")
    print(f"Modified actions : {modified_actions}")
    print(f"Modified joints  : {modified_joints}")
    print(f"Removed joints   : {removed_joints}")
    print(f"Saved to         : {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()