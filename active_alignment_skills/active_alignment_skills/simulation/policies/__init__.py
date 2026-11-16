"""Selectable frame-to-measurement policies for the dummy publisher."""

from .geometric import GeometricPolicy
from .optical import OpticalPolicy


def make_policy(parameters):
    """Add another explicit branch here when implementing a new policy."""
    if parameters['policy'] == 'optical':
        return OpticalPolicy(parameters)
    if parameters['policy'] == 'geometric':
        return GeometricPolicy(parameters)
    raise ValueError('policy must be optical or geometric.')
