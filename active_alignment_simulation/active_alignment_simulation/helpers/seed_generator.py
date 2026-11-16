"""Reproducible target-power poses without overwriting frame-description files."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from .config import JOINT_AXES


def generate_seed(model, enabled_joints, power_percent, seed):
    """Solve along a sampled ray inside the enabled-DOF box.

    Disabled coordinates stay exactly zero. Unlike the working-branch sampler,
    no components are discarded after calculating the target power and no
    impossible request is silently replaced by a different signal level.
    """
    if not 0 < power_percent <= 100:
        raise ValueError('Requested seed power must be in (0, 100]; exact zero is unreachable.')
    rng = np.random.default_rng(seed)
    limits = np.array([model.mode_field_radius, model.mode_field_radius, model.rayleigh_range,
                       model.theta_eff, model.theta_eff]) * model.search_factor
    mask = np.array([enabled_joints.get(name, False) for name in JOINT_AXES])
    limits *= mask
    if np.linalg.norm(limits[3:]) >= math.pi:
        raise ValueError(
            'Angular seed limits must stay within the principal rotation-vector range.')

    def signal(vector):
        quaternion = Rotation.from_rotvec([vector[3], vector[4], 0]).as_quat()
        return model.evaluate(vector[:3], quaternion)['eta_clean']

    target = power_percent / 100
    minimum = signal(limits)
    if target < minimum - 1e-14:
        raise ValueError(
            f'{power_percent}% is outside the enabled search box (minimum {minimum*100:.6g}%).')
    direction = limits.copy()
    for _ in range(256):
        direction = rng.uniform(-1, 1, 5) * limits
        if signal(direction) <= target:
            break
    else:
        direction = limits * rng.choice([-1, 1], 5)
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if signal(direction * middle) > target:
            low = middle
        else:
            high = middle
    vector = direction * ((low + high) / 2)
    if target == 1.0:
        vector[:] = 0
    quaternion = Rotation.from_rotvec([vector[3], vector[4], 0]).as_quat()
    return {'seed': int(seed), 'power_percent': float(power_percent),
            'translation': vector[:3].tolist(), 'quaternion': quaternion.tolist(),
            'expected_signal': signal(vector)}


def generate_seeds(config, model):
    seeds = []
    for group_index, group in enumerate(config.get('seed_groups', [])):
        for index in range(group['num_seeds']):
            # Distinct groups get distinct deterministic random sequences.
            seed = config['seed_base'] + group_index * 100000 + index
            seeds.append(
                generate_seed(
                    model,
                    config['enabled_joints'],
                    group['power_percent'],
                    seed))
    if not seeds:
        raise ValueError('The batch has no enabled seeds.')
    return seeds
