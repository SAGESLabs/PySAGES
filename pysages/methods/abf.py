# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from collections import namedtuple

from jax import scipy

from pysages.grids import build_indexer
from pysages.methods.core import GriddedSamplingMethod, generalize

import jax.numpy as np


# ======= #
#   ABF   #
# ======= #

class ABFState(namedtuple(
    "ABFState",
    (
        "bias",
        "hist",
        "Fsum",
        "F",
        "Wp",
        "Wp_",
    ),
)):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class ABF(GriddedSamplingMethod):
    """
    This is how ABF works.
    """
    snapshot_flags = {"positions", "indices", "momenta"}

    def build(self, snapshot, helpers):
        self.N = np.asarray(self.kwargs.get('N', 200))
        return _abf(self, snapshot, helpers)


def _abf(method, snapshot, helpers):
    cv = method.cv
    grid = method.grid
    N = method.N

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    get_grid_index = build_indexer(grid)

    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        return ABFState(bias, hist, Fsum, F, Wp, Wp_)

    def update(state, data):
        # Compute the collective variable and its jacobian
        ξ, Jξ = cv(data)
        #
        p = data.momenta
        # The following could equivalently be computed as `linalg.pinv(Jξ.T) @ p`
        # (both seem to have the same performance).
        # Another option to benchmark against is
        # Wp = linalg.tensorsolve(Jξ @ Jξ.T, Jξ @ p)
        Wp = scipy.linalg.solve(Jξ @ Jξ.T, Jξ @ p, sym_pos="sym")
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_ξ = get_grid_index(ξ)
        N_ξ = state.hist[I_ξ] + 1
        # Add previous force to remove bias
        F_ξ = state.Fsum[I_ξ] + dWp_dt + state.F
        hist = state.hist.at[I_ξ].set(N_ξ)
        Fsum = state.Fsum.at[I_ξ].set(F_ξ)
        F = F_ξ / np.maximum(N_ξ, N)
        #
        bias = np.reshape(-Jξ.T @ F, state.bias.shape)
        #
        return ABFState(bias, hist, Fsum, F, Wp, state.Wp)

    return snapshot, initialize, generalize(update, helpers)
