"""Simulation YAML validation, task expansion, and parameter resolution."""

from copy import deepcopy
import math
from pathlib import Path

import yaml

from active_alignment_skills.optimization.registry import OPTIMIZERS
from active_alignment_skills.helpers.parameters import (
    flatten, read_parameter_file, validate_parameters)
from active_alignment_skills.simulation.policies.optical import CouplingModel


JOINT_AXES = {'SP_X_Joint': 0, 'SP_Y_Joint': 1, 'SP_Z_Joint': 2,
              'SP_A_Joint': 3, 'SP_B_Joint': 4}
ALIASES = {'Hill Climb': 'hill_climb_manual', 'Adaptive Hill Climb': 'adaptive_hill_climb',
           'Nelder-Mead': 'nelder_mead_manual', 'Py-BOBYQA': 'pybobyqa',
           'Random Search': 'random_search',
           'Simple Adaptive Random Search': 'simple_adaptive_random_search'}


def load_simulation(path):
    path = Path(path).expanduser().resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('Simulation configuration must be a YAML mapping.')
    config = deepcopy(config)
    config['_path'] = str(path)
    config.setdefault('alignment_server', '/active_alignment_server')
    config.setdefault('publisher_node', '/multi_frame_deviation_node')
    config.setdefault('service_timeout', 10.0)
    config.setdefault('run_timeout', 600.0)
    config.setdefault('setup_settle_time', 0.2)
    config.setdefault('continue_on_failure', True)
    config.setdefault('seed_base', 31248769)
    config.setdefault('optimizer_seed_base', 1000)
    config.setdefault('noise_seed_base', 2000)
    config.setdefault('reset_position', 0.5)
    config.setdefault(
        'signal_verification', {
            'topic': '/Test1/eta_clean', 'tolerance': 0.001, 'timeout': 5.0})
    verification = config['signal_verification']
    if (not isinstance(verification, dict) or not verification.get('topic')
            or not math.isfinite(verification.get('tolerance', 0.001))
            or verification.get('tolerance', 0.001) < 0
            or not math.isfinite(verification.get('timeout', 5.0))
            or verification.get('timeout', 5.0) <= 0):
        raise ValueError(
            'signal_verification requires a topic, nonnegative tolerance, and positive timeout.')
    for key in ('service_timeout', 'run_timeout'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    if not math.isfinite(config['setup_settle_time']) or config['setup_settle_time'] < 0:
        raise ValueError('setup_settle_time must be finite and nonnegative.')
    if not isinstance(config['continue_on_failure'], bool):
        raise ValueError('continue_on_failure must be a boolean.')
    for key in ('seed_base', 'optimizer_seed_base', 'noise_seed_base'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer.')
    if not 0 <= config['reset_position'] <= 1:
        raise ValueError('reset_position must lie in [0, 1].')
    enabled = config.get('enabled_joints')
    if enabled is None:
        mode = config.get('mode', '5D')
        modes = {'5D': list(JOINT_AXES), '2D_TRANS': ['SP_X_Joint', 'SP_Y_Joint'],
                 '2D_ROT': ['SP_A_Joint', 'SP_B_Joint'],
                 '3D_TRANS': ['SP_X_Joint', 'SP_Y_Joint', 'SP_Z_Joint']}
        if mode not in modes:
            raise ValueError(
                f'Unsupported mode: {mode}. Use enabled_joints for custom selections.')
        enabled = {name: name in modes[mode] for name in JOINT_AXES}
    if not isinstance(enabled, dict) or not any(enabled.values()):
        raise ValueError('Enable at least one joint.')
    if any(name not in JOINT_AXES or type(value) is not bool for name, value in enabled.items()):
        raise ValueError('enabled_joints must map supported joint names to booleans.')
    config['enabled_joints'] = enabled
    model = make_model(config)

    def axis_limit(axis):
        if axis == 2:
            return model.search_factor * model.rayleigh_range
        if axis < 3:
            return model.search_factor * model.mode_field_radius
        return model.search_factor * model.theta_eff

    config.setdefault('active_joints', [
        {'joint_name': name, 'has_constraint': True,
         'upper_limit': axis_limit(axis), 'lower_limit': -axis_limit(axis)}
        for name, axis in JOINT_AXES.items() if enabled.get(name, False)])
    if {joint['joint_name'] for joint in config['active_joints']} != {
            name for name, value in enabled.items() if value}:
        raise ValueError('active_joints must match enabled_joints for simulation seed generation.')
    if len(config['active_joints']) != sum(enabled.values()):
        raise ValueError('Duplicate active joints are not permitted.')
    for joint in config['active_joints']:
        joint.setdefault('has_constraint', True)
        if not isinstance(joint.get('has_constraint', True), bool):
            raise ValueError('has_constraint must be a boolean.')
        for key in ('upper_limit', 'lower_limit'):
            if not math.isfinite(joint[key]):
                raise ValueError('Joint limits must be finite.')
    initial = config.get('initial_joint_positions', {})
    if initial and (set(initial) != {j['joint_name'] for j in config['active_joints']}
                    or any(not math.isfinite(value) for value in initial.values())):
        raise ValueError(
            'initial_joint_positions must provide every active joint with a finite position.')
    topics = config.get('alignment_topics', [
        {'topic_name': '/Test1/signal', 'minimize_not_maximize': False, 'weight_factor': 1.0}])
    if not topics or len({t['topic_name'] for t in topics}) != len(topics):
        raise ValueError('Provide distinct alignment topics.')
    for topic in topics:
        if not topic['topic_name'] or not isinstance(topic['minimize_not_maximize'], bool):
            raise ValueError('Invalid alignment topic or objective direction.')
        weight = topic.get('weight_factor', 1.0)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Alignment topic weights must be finite and positive.')
    config['alignment_topics'] = topics
    tasks = config.get('tasks', [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Provide at least one batch task.')
    for task in tasks:
        task['algorithm'] = ALIASES.get(task.get('algorithm'), task.get('algorithm'))
        if task['algorithm'] not in OPTIMIZERS:
            raise ValueError(f'Unknown algorithm: {task["algorithm"]}')
        count = task.setdefault('iterations', 1)
        if type(count) is not int or count < 0:
            raise ValueError('Task iterations must be a nonnegative number of complete runs.')
        if not isinstance(task.get('params', {}), dict):
            raise ValueError('Task params must be a YAML mapping.')
        if not isinstance(task.get('general', {}), dict):
            raise ValueError('Task general settings must be a YAML mapping.')
    for group in config.get('seed_groups', []):
        if (not math.isfinite(group['power_percent']) or not 0 <= group['power_percent'] <= 100
                or type(group['num_seeds']) is not int or group['num_seeds'] < 0):
            raise ValueError('Invalid seed group: power_percent in [0,100], num_seeds >= 0.')
    target = config.get('target', {})
    if not target.get('frame_name'):
        raise ValueError('target.frame_name is required.')
    config.setdefault('world_frame', 'world')
    return config


def make_model(config):
    fields = CouplingModel.__dataclass_fields__
    settings = {key: value for key, value in config.get('model', {}).items() if key in fields}
    enabled = config.get('enabled_joints', {})
    settings.update(
        lateral_enabled=any(enabled.get(name, False) for name in ('SP_X_Joint', 'SP_Y_Joint')),
        axial_enabled=enabled.get('SP_Z_Joint', False),
        angular_enabled=any(enabled.get(name, False) for name in ('SP_A_Joint', 'SP_B_Joint')),
        lateral_axes=sum(enabled.get(name, False)
                         for name in ('SP_X_Joint', 'SP_Y_Joint')),
        angular_axes=sum(enabled.get(name, False) for name in ('SP_A_Joint', 'SP_B_Joint')))
    return CouplingModel(**settings)


def task_parameters(config, task, baseline):
    """Start each task from the same snapshot, then apply its explicit overrides."""
    values = dict(baseline)
    base = Path(config['_path']).parent
    for path in (config.get('optimization_parameter_file'), task.get('parameter_file')):
        if path:
            path = Path(path).expanduser()
            values.update(read_parameter_file(path if path.is_absolute() else base / path,
                                              config['alignment_server']))
    values.update(flatten(config.get('optimization_parameters', {})))
    values.update(flatten(task.get('general', {}), 'general'))
    values.update(flatten(task.get('params', {}), task['algorithm']))
    return validate_parameters(values)
