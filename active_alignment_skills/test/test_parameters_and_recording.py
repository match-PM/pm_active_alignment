"""Configuration precedence and trustworthy persisted result records."""

import csv
import json

import pytest

from active_alignment_skills.optimization.base import OptimizationSession, run_optimizer
from active_alignment_skills.optimization.random_search import RandomSearchOptimizer
from active_alignment_skills.helpers.parameters import (
    load_defaults, optimizer_parameters, read_parameter_file,
    resolve_parameter_file, rsap_directory, validate_parameters)
from active_alignment_skills.helpers.run_recorder import RunRecorder


def test_explicit_parameter_file_wins_and_relative_paths_use_rsap_directory(tmp_path):
    fallback = tmp_path / 'optimization_parameter.yaml'
    fallback.write_text('general: {max_evaluations: 40}\n')
    explicit = tmp_path / 'custom.yaml'
    explicit.write_text('general: {max_evaluations: 20}\n')
    rsap = tmp_path / 'program.rsap.json'
    assert rsap_directory(str(rsap)) == tmp_path
    assert resolve_parameter_file('custom.yaml', str(rsap)) == explicit
    assert resolve_parameter_file('', str(rsap)) == fallback
    assert resolve_parameter_file('', '') is None
    with pytest.raises(ValueError, match='Explicit'):
        resolve_parameter_file('missing.yaml', str(rsap))
    fallback.unlink()
    assert resolve_parameter_file('', str(rsap)) is None


@pytest.mark.parametrize('rsap_name', ['', 'program.rsap.json', 'program.rsap'])
def test_missing_rsap_parameter_file_uses_current_defaults(tmp_path, rsap_name):
    assert resolve_parameter_file('', str(tmp_path / rsap_name)) is None


@pytest.mark.parametrize('document', [
    'nelder_mead:\n  xatol: 0.001\n',
    'active_alignment_server:\n  ros__parameters:\n    nelder_mead:\n      xatol: 0.001\n',
    '/active_alignment_server:\n  ros__parameters:\n    nelder_mead.xatol: 0.001\n',
    '/**:\n  ros__parameters:\n    nelder_mead.xatol: 0.001\n',
])
def test_compact_and_ros_yaml_have_same_meaning(tmp_path, document):
    path = tmp_path / 'parameters.yaml'
    path.write_text(document)
    assert read_parameter_file(path) == {'nelder_mead.xatol': 0.001}


@pytest.mark.parametrize('updates', [
    {'nelder_mead.unknown': 1}, {'general.max_evaluations': -1},
    {'general.max_evaluations': True}, {'nelder_mead.xatol': float('nan')},
    {'hill_climb.step_size': 0.0}, {'hill_climb.step_size': 0.9},
    {'simple_adaptive_random_search.window_size': 0},
    {'nelder_mead_manual.gamma': 2.0}, {'pybobyqa.initial_tr_radius': 0.8},
    {'pybobyqa.final_tr_radius': 0.1},
])
def test_invalid_optimizer_configuration_is_rejected(updates):
    with pytest.raises(ValueError):
        validate_parameters({**load_defaults(), **updates})


def test_scientific_notation_yaml_scalar_is_a_numeric_parameter(tmp_path):
    path = tmp_path / 'parameters.yaml'
    path.write_text('nelder_mead: {xatol: 1e-5}\n')
    values = validate_parameters(read_parameter_file(path))
    assert values['nelder_mead.xatol'] == 0.00001


def test_best_history_index_and_partial_run_are_persisted(tmp_path):
    parameters = load_defaults()
    p = optimizer_parameters(parameters, 'random_search')
    p['max_evaluations'] = 5
    recorder = RunRecorder(tmp_path, 'random_search', parameters, {})
    session = OptimizationSession([0.5], p, lambda x: float(x[0]), observe=recorder.record)
    run_optimizer(RandomSearchOptimizer(), session)
    recorder.finish('failed', session, 'Robot failed to restore best position.')
    metadata = json.loads((tmp_path / 'metadata.json').read_text())
    rows = list(csv.DictReader((tmp_path / 'history.csv').open()))
    assert len(rows) == 6
    assert metadata['status'] == 'failed'
    assert float(rows[metadata['best_history_index']]['objective']) == metadata['best_objective']


def test_common_settings_are_defined_once_and_reach_every_optimizer():
    values = load_defaults()
    values['general.measurement_timeout'] = 2.5
    values['general.joint_tolerance'] = 0.0001
    algorithms = set(name.split('.')[0] for name in values) - {'general'}
    for algorithm in algorithms:
        settings = optimizer_parameters(values, algorithm)
        assert settings['measurement_timeout'] == 2.5
        assert settings['joint_tolerance'] == 0.0001
        assert algorithm + '.measurement_timeout' not in values
        assert algorithm + '.joint_tolerance' not in values
