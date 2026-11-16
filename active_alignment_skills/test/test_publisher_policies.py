"""The original geometric policy and optical policy remain independently editable."""

import math

import pytest
from scipy.spatial.transform import Rotation

from active_alignment_skills.simulation.policies import make_policy
from active_alignment_skills.simulation.publisher_configuration import publisher_defaults


def test_geometric_policy_retains_original_distance_scale_and_full_angle():
    parameters = publisher_defaults()
    parameters['policy'] = 'geometric'
    policy = make_policy(parameters)
    quaternion = Rotation.from_euler('z', 0.2).as_quat()
    values = policy.evaluate([0.003, 0.004, 0.0], quaternion)
    assert values == pytest.approx({'distance': 0.05, 'angle': 0.2})


def test_optical_policy_uses_physical_distance_and_ignores_optical_axis_roll():
    parameters = publisher_defaults()
    policy = make_policy(parameters)
    radius = parameters['mode_field_radius']
    quaternion = Rotation.from_euler('z', 0.2).as_quat()
    values = policy.evaluate([radius, 0.0, 0.0], quaternion)
    assert values['distance'] == radius
    assert values['eta_angular'] == 1.0
    assert values['eta_clean'] == pytest.approx(math.exp(-1))
    assert values['signal'] == values['eta_clean']


def test_invalid_policy_is_rejected():
    parameters = publisher_defaults()
    parameters['policy'] = 'unknown'
    with pytest.raises(ValueError, match='policy'):
        make_policy(parameters)
