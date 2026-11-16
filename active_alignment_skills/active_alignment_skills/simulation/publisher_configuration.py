"""Configuration helpers kept separate from the publisher's ROS flow."""

from dataclasses import dataclass
import math
from pathlib import Path

import yaml

from .policies.optical import CouplingModel


@dataclass
class FramePair:
    """One measured frame and its corresponding target frame."""

    name: str
    frame_a: str
    frame_b: str


def publisher_defaults():
    """Defaults can be overridden through ROS parameters or the simulation YAML."""
    parameters = {
        'simulation_parameter_file': '',
        'policy': 'optical',
        'publish_rate': 50.0,
        'pair_names': ['Test1', 'Test2'],
        'frames_a': ['AL_Test_Frames_Align_1', 'AL_Test_Frames_Align_2'],
        'frames_b': ['AL_Test_Frames_Target_1', 'AL_Test_Frames_Target_2'],
        'legacy_distance_scale': 10.0,
        'noise_mode': 'none',
        'noise_seed': 0,
        'shot_low': 0.002,
        'dark_low': 0.0004,
        'shot_high': 0.01,
        'dark_high': 0.002,
    }
    parameters.update(CouplingModel().to_parameters())
    return parameters


def load_publisher_overrides(path):
    with Path(path).expanduser().open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('Simulation YAML must contain a mapping.')

    parameters = dict(config.get('model', {}))
    frame_pairs = config.get('frame_pairs')
    if frame_pairs:
        parameters['pair_names'] = []
        parameters['frames_a'] = []
        parameters['frames_b'] = []
        for pair in frame_pairs:
            parameters['pair_names'].append(pair['name'])
            parameters['frames_a'].append(pair['frame_a'])
            parameters['frames_b'].append(pair['frame_b'])
    return parameters


def create_frame_pairs(parameters):
    names = parameters['pair_names']
    frames_a = parameters['frames_a']
    frames_b = parameters['frames_b']
    if not names or len(names) != len(frames_a) or len(names) != len(frames_b):
        raise ValueError('Frame-pair arrays must be nonempty and have equal lengths.')
    if len(set(names)) != len(names) or not all(names + frames_a + frames_b):
        raise ValueError('Frame pairs require unique names and nonempty frame IDs.')
    rate = parameters['publish_rate']
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('publish_rate must be finite and positive.')

    pairs = []
    for name, frame_a, frame_b in zip(names, frames_a, frames_b):
        pairs.append(FramePair(name, frame_a, frame_b))
    return pairs
