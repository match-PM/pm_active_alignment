"""Check power-targeted seeds for every supported axis selection."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from active_alignment_skills.simulation.policies.optical import CouplingModel
from active_alignment_skills.simulation.noise_model import NoiseModel
from active_alignment_simulation.helpers.seed_generator import generate_seed
from active_alignment_simulation.helpers.config import JOINT_AXES


@pytest.mark.parametrize('enabled', [
    ['SP_X_Joint'], ['SP_Y_Joint'], ['SP_Z_Joint'], ['SP_A_Joint'], ['SP_B_Joint'],
    ['SP_X_Joint', 'SP_Z_Joint', 'SP_B_Joint'], list(JOINT_AXES),
])
@pytest.mark.parametrize('power', [20.0, 60.0, 100.0])
def test_disabled_axes_stay_zero_and_target_power_is_exact(enabled, power):
    mask = {name: name in enabled for name in JOINT_AXES}
    model = CouplingModel(lateral_enabled=any(mask.get(n) for n in ('SP_X_Joint', 'SP_Y_Joint')),
                          axial_enabled=mask['SP_Z_Joint'],
                          angular_enabled=any(mask.get(n) for n in ('SP_A_Joint', 'SP_B_Joint')))
    seed = generate_seed(model, mask, power, 42)
    rotvec = Rotation.from_quat(seed['quaternion']).as_rotvec()
    vector = seed['translation'] + list(rotvec[:2])
    for name, axis in JOINT_AXES.items():
        if not mask[name]:
            assert abs(vector[axis]) < 1e-15
    assert seed['expected_signal'] == pytest.approx(power / 100, abs=1e-12)
    assert seed == generate_seed(model, mask, power, 42)


def test_impossible_target_power_is_rejected_instead_of_silently_changed():
    model = CouplingModel(lateral_enabled=False, angular_enabled=False)
    with pytest.raises(ValueError, match='outside'):
        generate_seed(model, {'SP_Z_Joint': True}, 2.0, 1)
    with pytest.raises(ValueError, match='zero'):
        generate_seed(model, {'SP_Z_Joint': True}, 0.0, 1)


def test_noise_is_bounded_reproducible_and_optional():
    first, second = NoiseModel('low', 3), NoiseModel('low', 3)
    np.testing.assert_array_equal([first.sample(0.5) for _ in range(100)],
                                  [second.sample(0.5) for _ in range(100)])
    assert NoiseModel('none').sample(0.4) == 0.4
    assert all(0 <= first.sample(1.0) <= 1 for _ in range(100))
