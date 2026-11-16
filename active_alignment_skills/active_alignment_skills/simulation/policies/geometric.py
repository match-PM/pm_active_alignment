"""Original main-branch policy: distance and full rotation between two frames."""

import math

import numpy as np
from scipy.spatial.transform import Rotation


class GeometricPolicy:
    """Keep the original distance scaling and angle calculation in one place."""

    def __init__(self, parameters):
        self.distance_scale = parameters['legacy_distance_scale']
        if not math.isfinite(self.distance_scale) or self.distance_scale <= 0:
            raise ValueError('legacy_distance_scale must be finite and positive.')

    def evaluate(self, translation, quaternion):
        # This is the original policy: publish distance * 10 and the magnitude
        # of the complete rotation, including roll about the optical axis.
        distance = float(np.linalg.norm(translation)) * self.distance_scale
        angle = float(Rotation.from_quat(quaternion).magnitude())
        return {'distance': distance, 'angle': angle}
