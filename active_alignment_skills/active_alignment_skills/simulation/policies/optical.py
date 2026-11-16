"""Five-DOF Gaussian coupling model from the experiment branch."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from ..noise_model import NoiseModel


@dataclass
class CouplingModel:
    """SI units throughout; axial roll does not affect the optical signal."""

    mode_field_radius: float = 5.2e-6
    wavelength: float = 1.55e-6
    refractive_index: float = 1.0
    search_factor: float = 2.0
    lateral_enabled: bool = True
    axial_enabled: bool = True
    angular_enabled: bool = True
    lateral_axes: int = 2
    angular_axes: int = 2

    def __post_init__(self):
        for name in ('mode_field_radius', 'wavelength', 'refractive_index', 'search_factor'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        for name in ('lateral_enabled', 'axial_enabled', 'angular_enabled'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean.')
        for name in ('lateral_axes', 'angular_axes'):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 2:
                raise ValueError(f'{name} must be an integer in [0, 2].')

    @property
    def theta_eff(self):
        return self.wavelength / (math.pi * self.refractive_index * self.mode_field_radius)

    @property
    def rayleigh_range(self):
        return math.pi * self.refractive_index * self.mode_field_radius**2 / self.wavelength

    def evaluate(self, translation, quaternion):
        """Calculate clean coupling and geometric diagnostics for one frame pair."""
        translation = np.asarray(translation, dtype=float)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError('Translation must contain three finite values.')
        rotation = Rotation.from_quat(quaternion)
        rotvec = rotation.as_rotvec()
        lateral = float(np.linalg.norm(translation[:2]))
        axial = abs(float(translation[2]))
        tilt = float(np.linalg.norm(rotvec[:2]))
        # Lateral and angular errors reduce the overlap of the optical modes.
        # Disabled degrees of freedom contribute a neutral efficiency of 1.
        eta_lateral = 1.0
        if self.lateral_enabled:
            eta_lateral = math.exp(-(lateral / self.mode_field_radius)**2)

        eta_angular = 1.0
        if self.angular_enabled:
            eta_angular = math.exp(-(tilt / self.theta_eff)**2)

        # Axial defocus follows the working branch's Rayleigh-range model.
        eta_axial = 1.0
        if self.axial_enabled:
            eta_axial = 1 / (1 + (axial / self.rayleigh_range)**2)

        eta_clean = eta_lateral * eta_angular * eta_axial
        lateral_span = max(1, math.sqrt(self.lateral_axes)) * self.mode_field_radius
        total_span = math.sqrt(self.lateral_axes * self.mode_field_radius**2
                               + int(self.axial_enabled) * self.rayleigh_range**2)
        return {
            'distance': float(np.linalg.norm(translation)),
            'angle': float(rotation.magnitude()),
            'angle_deviation': tilt,
            'eta_lateral': eta_lateral,
            'eta_angular': eta_angular,
            'eta_axial': eta_axial,
            'eta_clean': eta_clean,
            'distance_lateral_percentage': 100 * lateral / (self.search_factor * lateral_span),
            'distance_axial_percentage': 100 * axial / (self.search_factor * self.rayleigh_range),
            'distance_percentage': 100 * np.linalg.norm(translation) / (
                self.search_factor * max(1e-30, total_span)),
            'angle_percentage': 100 * tilt / (
                max(1, math.sqrt(self.angular_axes)) * self.search_factor * self.theta_eff),
        }

    def to_parameters(self):
        """ROS parameter names for this model, also used by the batch runner."""
        return {
            'mode_field_radius': self.mode_field_radius,
            'wavelength': self.wavelength,
            'refractive_index': self.refractive_index,
            'search_factor': self.search_factor,
            'lateral_enabled': self.lateral_enabled,
            'axial_enabled': self.axial_enabled,
            'angular_enabled': self.angular_enabled,
            'lateral_axes': self.lateral_axes,
            'angular_axes': self.angular_axes,
        }


class OpticalPolicy:
    """Publish the working branch's optical signal and diagnostics.

    Edit CouplingModel.evaluate above to change the optical equations. Noise is
    applied only to signal, leaving eta_clean available for setup verification.
    """

    TOPICS = (
        'signal', 'distance', 'angle', 'angle_deviation',
        'eta_lateral', 'eta_angular', 'eta_axial', 'eta_clean',
        'distance_percentage', 'distance_lateral_percentage',
        'distance_axial_percentage', 'angle_percentage',
    )

    def __init__(self, parameters):
        self.model = CouplingModel(
            mode_field_radius=parameters['mode_field_radius'],
            wavelength=parameters['wavelength'],
            refractive_index=parameters['refractive_index'],
            search_factor=parameters['search_factor'],
            lateral_enabled=parameters['lateral_enabled'],
            axial_enabled=parameters['axial_enabled'],
            angular_enabled=parameters['angular_enabled'],
            lateral_axes=parameters['lateral_axes'],
            angular_axes=parameters['angular_axes'],
        )
        self.noise = NoiseModel(
            mode=parameters['noise_mode'], seed=parameters['noise_seed'],
            shot_low=parameters['shot_low'], dark_low=parameters['dark_low'],
            shot_high=parameters['shot_high'], dark_high=parameters['dark_high'],
        )

    def evaluate(self, translation, quaternion):
        values = self.model.evaluate(translation, quaternion)
        values['signal'] = self.noise.sample(values['eta_clean'])
        return values
