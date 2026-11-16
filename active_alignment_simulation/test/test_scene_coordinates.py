"""Ensure assembly-manager requests use its world-coordinate contract."""

from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip('assembly_manager_interfaces')
from active_alignment_simulation.helpers.scene_frames import (  # noqa: E402
    SceneFrames, pose_from_values, pose_values)


def test_absolute_pose_request_converts_configured_frame_to_world():
    frames = SceneFrames.__new__(SceneFrames)
    frames.touched = []
    parent_rotation = Rotation.from_euler('z', 90, degrees=True)
    frames.world_pose = Mock(return_value=pose_from_values(
        [10, 20, 0], parent_rotation.as_quat()))
    frames.call = Mock()
    frames.verify_world_pose = Mock()
    baseline = pose_from_values([1, 2, 3], [0, 0, 0, 1])
    seed = {'translation': [0.1, 0, 0], 'quaternion': [0, 0, 0, 1]}
    config = {'world_frame': 'experiment_origin', 'service_timeout': 5.0,
              'target': {'frame_name': 'target'}}
    frames.apply_seed(config, baseline, seed, lambda: False)
    request = frames.call.call_args.args[1]
    position, quaternion = pose_values(request.pose)
    np.testing.assert_allclose(position, [8, 21.1, 3])
    assert (Rotation.from_quat(quaternion).inv() * parent_rotation).magnitude() < 1e-12
    assert request.modify_orientation
    assert frames.verify_world_pose.call_args.args[0] == 'world'
