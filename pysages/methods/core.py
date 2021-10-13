# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABC, abstractmethod
from typing import Callable, Mapping

from jax import jit
from pysages.backends import ContextWrapper
from pysages.collective_variables.core import build


# ================ #
#   Base Classes   #
# ================ #
class SamplingMethod(ABC):
    def __init__(self, cvs, *args, **kwargs):
        self.cv = build(*cvs)
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        """
        Returns the snapshot, and two functions: `initialize` and `update`.
        `initialize` is intended to allocate any runtime information required
        by `update`, while `update` is intended to be called after each call to
        the wrapped context's `run` method.
        """
        pass

    def run(
        self, context_generator: Callable, timesteps: int, callback: Callable = None,
        context_args: Mapping = dict(), **kwargs
    ):
        """
        Base implementation of running a single simulation/replica with a sampling method.

        Arguments
        ---------
        context_generator: Callable
            User defined function that sets up a simulation context with the backend.
            Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
            and `simtk.openmm.Simulation` for OpenMM. The function gets `context_args`
            unpacked for additional user arguments.

        timesteps: int
            Number of timesteps the simulation is running.

        callback: Optional[Callable]
            Allows for user defined actions into the simulation workflow of the method.
            `kwargs` gets passed to the backend `run` function.
        """
        context = context_generator(**context_args)
        wrapped_context = ContextWrapper(context, self, callback)
        with wrapped_context:
            wrapped_context.run(timesteps, **kwargs)


class GriddedSamplingMethod(SamplingMethod):
    def __init__(self, cvs, grid, *args, **kwargs):
        check_dims(cvs, grid)
        self.cv = build(*cvs)
        self.grid = grid
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


class NNSamplingMethod(SamplingMethod):
    def __init__(self, cvs, grid, topology, *args, **kwargs):
        check_dims(cvs, grid)
        self.cv = build(*cvs)
        self.grid = grid
        self.topology = topology
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


# ========= #
#   Utils   #
# ========= #

def check_dims(cvs, grid):
    if len(cvs) != grid.shape.size:
        raise ValueError("Grid and Collective Variable dimensions must match.")


def generalize(concrete_update, jit_compile = True):
    if jit_compile:
        _jit = jit
    else:
        def _jit(x): return x

    _update = _jit(concrete_update)

    def update(snapshot, state):
        vms = snapshot.vel_mass
        rs = snapshot.positions
        ids = snapshot.ids
        #
        return _update(state, rs, vms, ids)

    return _jit(update)
