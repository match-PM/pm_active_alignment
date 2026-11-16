"""Per-task settings are complete independent snapshots with explicit precedence."""

from active_alignment_skills.helpers.parameters import load_defaults
from active_alignment_simulation.helpers.config import task_parameters


def test_task_overrides_do_not_leak_into_next_task(tmp_path):
    baseline = load_defaults()
    config = {'_path': str(tmp_path / 'simulation.yaml'),
              'alignment_server': '/active_alignment_server'}
    first = task_parameters(config, {
        'algorithm': 'nelder_mead',
        'general': {'max_evaluations': 3},
        'params': {'simplex_step': 0.1}}, baseline)
    second = task_parameters(config, {'algorithm': 'nelder_mead', 'params': {}}, baseline)
    assert first['general.max_evaluations'] == 3
    assert second['general.max_evaluations'] == baseline['general.max_evaluations']
    assert second['nelder_mead.simplex_step'] == baseline['nelder_mead.simplex_step']


def test_yaml_task_and_inline_parameter_precedence(tmp_path):
    (tmp_path / 'base.yaml').write_text(
        'general: {max_evaluations: 10}\nnelder_mead: {fatol: 0.01}\n')
    (tmp_path / 'task.yaml').write_text(
        'general: {max_evaluations: 20}\nnelder_mead: {fatol: 0.02}\n')
    config = {'_path': str(tmp_path / 'simulation.yaml'),
              'alignment_server': '/active_alignment_server',
              'optimization_parameter_file': 'base.yaml',
              'optimization_parameters': {'nelder_mead': {'fatol': 0.03}}}
    resolved = task_parameters(config, {'algorithm': 'nelder_mead', 'parameter_file': 'task.yaml',
                                        'general': {'max_evaluations': 30}}, load_defaults())
    assert resolved['general.max_evaluations'] == 30
    assert resolved['nelder_mead.fatol'] == 0.03
