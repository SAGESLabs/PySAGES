# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES
"""
Collective Variables that are calcuated from the shape of group of atoms.
"""


import jax.numpy as np
from jax.numpy import linalg
from .core import CollectiveVariable, AxisCV


class RadiusOfGyration(CollectiveVariable):
    """
    Collective Variable that calculates the unweighted radius of gyration as CV.

    Parameters
    ----------
    indices : list[int], list[tuple(int)]
       Must be a list or tuple of atoms (ints or ranges) or groups of
       atoms. A group is specified as a nested list or tuple of atoms.
    group_length: int, optional
       Specify if a fixed group length is expected.
    """

    @property
    def function(self):
        """
        Returns
        -------
        Callable
           Function to calculate the radius of gyration. See `pysages.collective_variables.shape.radius_of_gyration` for details.
        """
        return radius_of_gyration


def radius_of_gyration(positions):
    """
    Calculate the Radius of gyration for a group of atoms.

    Parameters
    ----------
    positions: DeviceArray
       Array of particle positions used to calculate the radius of gyration.

    Returns
    -------
    DeviceArray
       Radius of Gyration vector
    """
    group_length = positions.shape[0]
    pos = np.zeros((3,))
    # TODO: Replace by `np.sum` and `vmap`  # pylint:disable=fixme
    for particle_pos in positions:
        pos[:] += np.dot(particle_pos, particle_pos)
    return pos / group_length


def weighted_radius_of_gyration(positions, weights):
    """
    Calculate the Radius of gyration for a group of atoms weighted by weights.

    Parameters
    ----------
    positions: DeviceArray
       Array of particle positions used to calculate the radius of gyration.
    weights: DeviceArray
       Array of weights for the positions.

    Returns
    -------
    DeviceArray
       Weighted Radius of Gyration vector
    """
    group_length = positions.shape[0]
    pos = np.zeros((3,))
    # TODO: Replace by `np.sum` and `vmap` # pylint:disable=fixme
    for i in range(group_length):
        weight, particle_pos = weights[i], positions[i]
        pos += weight * np.dot(particle_pos, particle_pos)
    return pos


class PrincipalMoment(AxisCV):
    """
    Calculate the principal moment as collective variable.

    Parameters
    ----------
    indices : list[int], list[tuple(int)]
       Must be a list or tuple of atoms (ints or ranges) or groups of
       atoms. A group is specified as a nested list or tuple of atoms.
    axis: int
       Index of the cartesian coordinate: 0==X, 1==Y, 2==Z
    group_length: int, optional
       Specify if a fixed group length is expected.
    """
    @property
    def function(self):
        """
        Returns
        -------
        Callable
           Function to calculate the Eigenvalue with the specified axis index of the GyrationTensor. See `pysages.collective_variables.shape.principal_moments` and `pysages.collective_variables.shape.gyration_tensor` for details.
        """
        return lambda rs: principal_moments(rs)[self.axis]


def gyration_tensor(positions):
    """
    Calculate the gyration tensor for a collection of points in space.

    Parameters
    ----------
    positions: DeviceArray
       Points in space, that are equally weighted to calculate the gyration tensor.
    Returns
    -------
    DeviceArray
       Gyration Tensor
    """
    group_length = positions.shape[0]
    pos = np.zeros((3, 3))
    for particle_pos in positions:
        pos += np.outer(particle_pos, particle_pos)
    return pos / group_length


def weighted_gyration_tensor(positions, weights):
    """
    Calculate the gyration tensor for a collection of points in space.

    Parameters
    ----------
    positions: DeviceArray
       Points in space, that are weighted by `weight` to calculate the gyration tensor.
    positions: DeviceArray
       Weights for the points in space e.g. particle masses

    Returns
    -------
    DeviceArray
       Gyration Tensor
    """
    group_length = positions.shape[0]
    pos = np.zeros((3, 3))
    for i in range(group_length):
        weight, particle_pos = weights[i], positions[i]
        pos += weight * np.outer(particle_pos, particle_pos)
    return pos


def principal_moments(positions):
    """
    Calculate the principal momements for positions.
    The principal moments are the Eigenvalues of the Gyration tensor.
    See `pysages.collective_variables.shape.gyration_tensor` for details.
    Parameters
    ----------
    positions: DeviceArray
       Points in space, that are equally weighted to calculate the gyration tensor.
    Returns
    -------
    DeviceArray
       Eigenvalues of the gyration tensor
    """
    return linalg.eigvals(gyration_tensor(positions))


class Asphericity(CollectiveVariable):
    """
    Collective Variable that calculates the Asphericity CV.

    Parameters
    ----------
    indices : list[int], list[tuple(int)]
       Must be a list or tuple of atoms (ints or ranges) or groups of
       atoms. A group is specified as a nested list or tuple of atoms.

    group_length: int, optional
       Specify if a fixed group length is expected.
    """
    @property
    def function(self):
        """
        Returns
        -------
        Callable
           Function to calculate the asphericity. See `pysages.collective_variables.shape.asphericity` for details.
        """
        return asphericity


def asphericity(positions):
    """
    Calculate the Asphericity from a group of atoms.

    Asphericity is defined as :math:`\\lambda_3 - (\\lambda_1 + \\lambda_2)/2`, where
    :math:`\\lambda_i` specifies the principal moments of the group of atoms.

    See `pysages.collective_variables.shape.principal_moments` for details.
    Parameters
    ----------
    positions: DeviceArray
       Points in space, that are equally weighted to calculate the principal moments.

    Returns
    -------
    float
       Asphericity
    """
    lambda1, lambda2, lambda3 = principal_moments(positions)
    return lambda3 - (lambda1 + lambda2) / 2


class Acylindricity(CollectiveVariable):
    """
    Collective Variable that calculates the Acylindricity CV.

    Parameters
    ----------
    indices : list[int], list[tuple(int)]
       Must be a list or tuple of atoms (ints or ranges) or groups of
       atoms. A group is specified as a nested list or tuple of atoms.

    group_length: int, optional
       Specify if a fixed group length is expected.
    """
    @property
    def function(self):
        """
        Returns
        -------
        Callable
           Function to calculate the acylindricity. See `pysages.collective_variables.shape.acylindricity` for details.
        """
        return acylindricity


def acylindricity(positions):
    """
    Calculate the Acylindricity from a group of atoms.

    Asphericity is defined as :math:`\\lambda_2 - \\lambda_1`, where :math:`\\lambda_i` specifies the principal moments of the group of atoms.

    See `pysages.collective_variables.shape.principal_moments` for details.
    Parameters
    ----------
    positions: DeviceArray
       Points in space, that are equally weighted to calculate the principal moments from.
    Returns
    -------
    float
       Asphericity
    """
    lambda1, lambda2, _ = principal_moments(positions)
    return lambda2 - lambda1


class ShapeAnisotropy(CollectiveVariable):
    """
    Collective Variable that calculates the Shape Anisotropy CV.

    Parameters
    ----------
    indices : list[int], list[tuple(int)]
       Must be a list or tuple of atoms (ints or ranges) or groups of
       atoms. A group is specified as a nested list or tuple of atoms.

    group_length: int, optional
       Specify if a fixed group length is expected.
    """
    @property
    def function(self):
        """
        Returns
        -------
        Callable
           Function to calculate the shape anisotropy. See `pysages.collective_variables.shape.shape_anisotropy` for details.
        """
        return shape_anisotropy


def shape_anisotropy(positions):
    """
    Calculate the shape anisotropy from a group of atoms.

    Asphericity is defined as :math:`1/2\\frac{3(\\lambda_1^2 + \\lambda_2^2 + \\lambda_3^2)}{(\\lambda_1 + \\lambda_2 + \\lambda_3)^2 - 1}`,
    where :math:`\\lambda_i` specifies the principal moments of the group of atoms.

    See `pysages.collective_variables.shape.principal_moments` for details.
    Parameters
    ----------
    positions: DeviceArray
       Points in space, that are equally weighted to calculate the principal moments from.

    Returns
    -------
    float
       Asphericity
    """
    lambda1, lambda2, lambda3 = principal_moments(positions)
    return (3 * (lambda1**2 + lambda2**2 + lambda3**2) / (lambda1 + lambda2 + lambda3)**2 - 1) / 2
