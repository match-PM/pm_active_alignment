"""Load compact or ROS parameter YAML and validate all algorithm settings."""

import math
from pathlib import Path

import yaml


def default_parameter_path():
    from ament_index_python.packages import get_package_share_directory
    try:
        return Path(get_package_share_directory('active_alignment_skills')) / \
            'config' / 'optimization_parameter.yaml'
    except LookupError:
        return Path(__file__).resolve().parents[2] / 'config' / 'optimization_parameter.yaml'


def flatten(mapping, prefix=''):
    if not isinstance(mapping, dict):
        raise ValueError('Expected a YAML mapping of parameter names to values.')
    result = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError('Parameter names must be nonempty strings.')
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def read_parameter_file(path, node_name='active_alignment_server'):
    with Path(path).open() as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not document:
        raise ValueError(f'Parameter file must contain a nonempty mapping: {path}')
    if 'ros__parameters' in document:
        return flatten(document['ros__parameters'])
    wrappers = [key for key, value in document.items()
                if isinstance(value, dict) and 'ros__parameters' in value]
    if wrappers:
        result = {}
        full_name = '/' + node_name.strip('/')
        for key in ('/**', node_name.strip('/').split('/')[-1], full_name):
            if key in document:
                result.update(flatten(document[key]['ros__parameters']))
        if not result:
            raise ValueError(f'No parameters for {full_name} in {path}.')
        return result
    return flatten(document)


def load_defaults():
    # Source-tree lookup first: editable installs can otherwise find an older share directory.
    source = Path(__file__).resolve().parents[2] / 'config' / 'optimization_parameter.yaml'
    return read_parameter_file(source if source.is_file() else default_parameter_path())


def validate_parameters(values, schema=None):
    schema = load_defaults() if schema is None else schema
    checked = dict(values)
    for name, value in values.items():
        if name not in schema:
            namespace, _, setting = name.partition('.')
            if namespace != 'general' and 'general.' + setting in schema:
                raise ValueError(f'{name} is shared; use general.{setting} instead.')
            raise ValueError(f'Unknown optimization parameter: {name}')
        expected = type(schema[name])
        if expected is float and isinstance(value, str):
            try:
                value = float(value)
            except ValueError as exc:
                raise ValueError(f'{name} must be a number.') from exc
        if expected is float and type(value) in (int, float):
            value = float(value)
        elif type(value) is not expected:
            raise ValueError(f'{name} must be {expected.__name__}, got {type(value).__name__}.')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{name} must be finite.')
        checked[name] = value
    merged = {**schema, **checked}
    general = namespace_parameters(merged, 'general')
    for key in ('max_evaluations', 'random_seed', 'wait_for_settle'):
        if general[key] < 0:
            raise ValueError(f'general.{key} must be nonnegative.')
    for key in ('movement_time', 'measurement_timeout', 'trajectory_timeout', 'joint_tolerance'):
        if general[key] <= 0:
            raise ValueError(f'general.{key} must be positive.')

    # General settings are declared only once. Each algorithm receives a merged
    # dictionary at execution time, without duplicating ROS parameters.
    algorithms = {name.split('.')[0] for name in schema if not name.startswith('general.')}
    for algorithm in algorithms:
        p = namespace_parameters(merged, algorithm)
        nonnegative = ['npt']
        for key in nonnegative:
            if key in p and p[key] < 0:
                raise ValueError(f'{algorithm}.{key} must be nonnegative.')
        positive = ['max_iterations', 'step_size', 'min_step',
                    'max_step', 'xatol', 'fatol', 'simplex_step', 'initial_tr_radius',
                    'final_tr_radius', 'perturbation_scale', 'sigma', 'min_sigma',
                    'max_sigma', 'window_size', 'adaptation_window']
        for key in positive:
            if key in p and p[key] <= 0:
                raise ValueError(f'{algorithm}.{key} must be positive.')
        if 'step_size' in p and not p['min_step'] <= p['step_size'] <= p['max_step'] <= 1:
            raise ValueError(f'{algorithm}: require min_step <= step_size <= max_step <= 1.')
        if 'step_increase_factor' in p:
            if p['step_increase_factor'] <= 1 or not 0 < p['step_decrease_factor'] < 1:
                raise ValueError(f'{algorithm}: invalid step adaptation factors.')
        if 'simplex_step' in p and p['simplex_step'] > 1:
            raise ValueError(f'{algorithm}.simplex_step must be <= 1.')
        if algorithm == 'nelder_mead_manual':
            if not (p['alpha'] > 0 and p['beta'] > 1 and 0 < p['gamma'] < 1
                    and 0 < p['delta'] < 1):
                raise ValueError('Invalid classical Nelder-Mead coefficients.')
        if algorithm == 'pybobyqa':
            if not 0 < p['final_tr_radius'] < p['initial_tr_radius'] <= 0.5:
                raise ValueError('Py-BOBYQA radii must satisfy 0 < final < initial <= 0.5.')
        if algorithm == 'simple_adaptive_random_search':
            if not (p['min_sigma'] <= p['sigma'] <= p['max_sigma'] <= 1
                    and p['alpha'] > 1 and 0 < p['gamma'] < 1):
                raise ValueError('Invalid adaptive random-search parameters.')
    return checked


def namespace_parameters(values, namespace):
    prefix = namespace + '.'
    return {name[len(prefix):]: value for name, value in values.items() if name.startswith(prefix)}


def optimizer_parameters(values, algorithm):
    """Combine the shared execution settings with one algorithm's search settings."""
    parameters = namespace_parameters(values, 'general')
    parameters.update(namespace_parameters(values, algorithm))
    return parameters


def rsap_directory(rsap_path):
    path = Path(rsap_path).expanduser().resolve()
    if path.is_file() or path.suffix.lower() in ('.json', '.rsap'):
        return path.parent
    return path


def resolve_parameter_file(parameter_file, rsap_path):
    """Return an override file, or None to retain the node's current defaults."""
    directory = rsap_directory(rsap_path) if rsap_path else None
    if parameter_file:
        path = Path(parameter_file).expanduser()
        if not path.is_absolute() and directory is not None:
            path = directory / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f'Explicit parameter file does not exist: {path}')
        return path
    if directory is not None:
        path = directory / 'optimization_parameter.yaml'
        if path.is_file():
            return path
    return None
