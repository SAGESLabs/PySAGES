# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import jax.numpy as np
from jax.numpy import linalg
from .core import TwoPointCV, AxisCV


def barycenter(positions):
    """
    Returns the geometric center, or centroid, of a group of points in space.
    """
    return np.sum(positions, axis=0) / positions.shape[0]


def weighted_barycenter(positions, weights):
    """
    Returns the center of a group of points in space weighted by arbitrary weights.
    """
    n = positions.shape[0]
    R = np.zeros(3)
    # TODO: Replace by `np.sum` and `vmap`
    for i in range(n):
        w, r = weights[i], positions[i]
        R += w * r
    return R


class Component(AxisCV):
    def __init__(self, indices, axis):
        super().__init__(indices, axis)
        self.requires_box_unwrapping = True

    @property
    def function(self):
        return (lambda rs: barycenter(rs)[self.axis])


class Distance(TwoPointCV):
    @property
    def function(self):
        return distance


def distance(r1, r2):
    """
    Returns the distance between two points in space.
    """
    return linalg.norm(r1 - r2)
