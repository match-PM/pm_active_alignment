"""Reproducible shot and dark measurement noise."""

import math

import numpy as np


class NoiseModel:
    """Noise modes retain the working branch's coefficient defaults."""

    def __init__(self, mode='none', seed=0, shot_low=0.002, dark_low=0.0004,
                 shot_high=0.01, dark_high=0.002):
        if mode not in ('none', 'low', 'high'):
            raise ValueError('Noise mode must be none, low, or high.')
        for value in (shot_low, dark_low, shot_high, dark_high):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Noise coefficients must be finite and nonnegative.')
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.shot = shot_low if mode == 'low' else shot_high
        self.dark = dark_low if mode == 'low' else dark_high

    def sample(self, signal):
        if self.mode == 'none':
            return float(np.clip(signal, 0, 1))
        return float(np.clip(signal + self.rng.normal(0, self.shot * math.sqrt(max(0, signal)))
                             + self.rng.normal(0, self.dark), 0, 1))
