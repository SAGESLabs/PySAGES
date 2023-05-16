# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective Variables that are calculated from the shape of group of atoms.
"""

import jax.numpy as np
from jax.numpy import linalg

from pysages.colvars.core import AxisCV, CollectiveVariable


class RadiusOfGyration(CollectiveVariable):
    """
    Collective Variable that calculates the unweighted radius of gyration as CV.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
        Must be a list or tuple of atoms (integers or ranges) or groups of atoms.
        A group is specified as a nested list or tuple of atoms.
    group_length: int, optional
        Specify if a fixed group length is expected.
    """

    @property
    def function(self):
        """
        Returns
        -------
        Callable
            See `pysages.colvars.shape.radius_of_gyration` for details.
        """
        return radius_of_gyration


def radius_of_gyration(positions):
    """
    Calculate the radius of gyration for a group of atoms.

    Parameters
    ----------
    positions: jax.Array
        Array of particle positions used to calculate the radius of gyration.

    Returns
    -------
    jax.Array
        Radius of gyration vector
    """
    group_length = positions.shape[0]
    rog = np.zeros((3,))
    # TODO: Replace by `np.sum` and `vmap`  # pylint:disable=fixme
    for r in positions:
        rog += np.dot(r, r)
    return rog / group_length


def weighted_radius_of_gyration(positions, weights):
    """
    Calculate the radius of gyration for a group of atoms weighted by arbitrary weights.

    Parameters
    ----------
    positions: jax.Array
        Array of particle positions used to calculate the radius of gyration.
    weights: jax.Array
        Array of weights for the positions.

    Returns
    -------
    jax.Array
        Weighted radius of gyration vector
    """
    group_length = positions.shape[0]
    rog = np.zeros((3,))
    # TODO: Replace by `np.sum` and `vmap` # pylint:disable=fixme
    for i in range(group_length):
        w, r = weights[i], positions[i]
        rog += w * np.dot(r, r)
    return rog


class PrincipalMoment(AxisCV):
    """
    Calculate the principal moment as collective variable.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
        Must be a list or tuple of atoms (integers or ranges) or groups of atoms.
        A group is specified as a nested list or tuple of atoms.
    axis: int
        Index of the Cartesian coordinate: 0 (X), 1 (Y), 2 (Z)
    group_length: Optional[int]
        Specify if a fixed group length is expected.
    """

    @property
    def function(self):
        """
        Returns
        -------
        Callable
            Function to calculate the eigenvalue with the specified axis index \
            of the gyration tensor. \
            See `pysages.colvars.shape.principal_moments` and \
            `pysages.colvars.shape.gyration_tensor` for details.
        """
        return lambda rs: principal_moments(rs)[self.axis]


def gyration_tensor(positions):
    """
    Calculate the gyration tensor for a collection of points in space.

    Parameters
    ----------
    positions: jax.Array
        Points in space that are equally weighted to calculate the gyration tensor.

    Returns
    -------
    jax.Array
        Gyration tensor
    """
    group_length = positions.shape[0]
    gyr = np.zeros((3, 3))
    for r in positions:
        gyr += np.outer(r, r)
    return gyr / group_length


def weighted_gyration_tensor(positions, weights):
    """
    Calculate the gyration tensor for a collection of points in space weighted by arbitrary weights.

    Parameters
    ----------
    positions: jax.Array
        Points in space that are weighted by `weight` to calculate the gyration tensor.
    weights: jax.Array
        Weights for the points in space e.g. particle masses

    Returns
    -------
    jax.Array
        Gyration tensor
    """
    group_length = positions.shape[0]
    gyr = np.zeros((3, 3))
    for i in range(group_length):
        w, r = weights[i], positions[i]
        gyr += w * np.outer(r, r)
    return gyr


def principal_moments(positions):
    """
    Calculate the principal moments for positions.
    The principal moments are the eigenvalues of the gyration tensor.
    See `pysages.colvars.shape.gyration_tensor` for details.

    Parameters
    ----------
    positions: jax.Array
        Points in space that are equally weighted to calculate the gyration tensor.

    Returns
    -------
    jax.Array
        Eigenvalues of the gyration tensor
    """
    return linalg.eigvalsh(gyration_tensor(positions))


class Asphericity(CollectiveVariable):
    """
    Collective Variable that calculates the Asphericity.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
        Must be a list or tuple of atoms (integers or ranges) or groups of atoms.
        A group is specified as a nested list or tuple of atoms.

    group_length: Optional[int]
        Specify if a fixed group length is expected.
    """

    @property
    def function(self):
        """
        Returns
        -------
        Callable
            See `pysages.colvars.shape.asphericity` for details.
        """
        return asphericity


def asphericity(positions):
    r"""
    Calculate the Asphericity from a group of atoms.
    It is defined as :math:`\lambda_3 - (\lambda_1 + \lambda_2) / 2`, where
    :math:`\lambda_i` specifies the principal moments of the group of atoms.

    See `pysages.colvars.shape.principal_moments` for details.

    Parameters
    ----------
    positions: jax.Array
        Points in space that are equally weighted to calculate the principal moments.

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
    indices: list[int], list[tuple(int)]
        Must be a list or tuple of atoms (integers or ranges) or groups of atoms.
        A group is specified as a nested list or tuple of atoms.

    group_length: Optional[int]
        Specify if a fixed group length is expected.

    axes: str
        Axis combination around which the particles are symmetric.
        Options are: xy, yz, xz
    """

    symmetry_axes = {"xy": (1, 2), "yz": (2, 3), "xz": (1, 3)}

    def __init__(self, indices, axes="xy", group_length=None):
        axes = "".join(sorted(axes.lower()))
        if axes not in self.symmetry_axes:
            error_msg = f"Invalid acylindrity axes specification {axes}."
            error_msg += f" Valid options are: {tuple(self.symmetry_axes.keys())}."
            raise RuntimeError(error_msg)

        super().__init__(indices, group_length)
        self.axes = axes

    @property
    def function(self):
        """
        Returns
        -------
        Callable
            See `pysages.colvars.shape.acylindricity` for details.
        """
        return lambda rs: acylindricity(rs, self.symmetry_axes[self.axes])


def acylindricity(positions, axes):
    r"""
    Calculate the Acylindricity from a group of atoms.
    It is defined as :math:`\lambda_k - \lambda_j`,
    where :math:`\lambda_i` specifies the principal moments of the group of atoms,
    and `j` and `k` are the axes around which the particles are symmetric.

    See `pysages.colvars.shape.principal_moments` for details.

    Parameters
    ----------
    positions: jax.Array
        Points in space that are equally weighted to calculate the principal moments from.

    axes: Tuple[int, int]
        Indices of the axes around which the particles are symmetric.

    Returns
    -------
    float
        Acylindricity
    """
    lambdas = principal_moments(positions)
    lambda1, lambda2 = [lambdas[i] for i in axes]
    return lambda2 - lambda1


class ShapeAnisotropy(CollectiveVariable):
    """
    Collective Variable that calculates the Shape Anisotropy CV.

    Parameters
    ----------
    indices : list[int], list[tuple(int)]
        Must be a list or tuple of atoms (integers or ranges) or groups of
        atoms. A group is specified as a nested list or tuple of atoms.

    group_length: Optional[int]
        Specify if a fixed group length is expected.
    """

    @property
    def function(self):
        """
        Returns
        -------
        Callable
            See `pysages.colvars.shape.shape_anisotropy` for details.
        """
        return shape_anisotropy


def shape_anisotropy(positions):
    r"""
    Calculate the shape anisotropy from a group of atoms, defined as:

    .. math::

        \frac{3}{2} \frac{\lambda_1^2 + \lambda_2^2 + \lambda_3^2}
        {(\lambda_1 + \lambda_2 + \lambda_3)^2} - \frac{1}{2}

    where :math:`\lambda_i` specifies the principal moments of the group of atoms.

    See `pysages.colvars.shape.principal_moments` for details.

    Parameters
    ----------
    positions: jax.Array
        Points in space that are equally weighted to calculate the principal moments from.

    Returns
    -------
    float
        Shape Anisotropy
    """
    lambda1, lambda2, lambda3 = principal_moments(positions)
    return (
        3 * (lambda1**2 + lambda2**2 + lambda3**2) / (lambda1 + lambda2 + lambda3) ** 2 - 1
    ) / 2
