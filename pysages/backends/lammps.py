# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# Maintainer: ndtrung

import importlib
from functools import partial

import jax
from jax import jit
from jax import numpy as np
from jax import vmap
from jax.dlpack import from_dlpack as asarray
from lammps import dlext
from lammps.dlext import ExecutionSpace, FixDLExt, LAMMPSView, has_kokkos_cuda_enabled

from pysages.backends import snapshot as pbs
from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.typing import Callable, Optional
from pysages.utils import copy

kDefaultLocation = dlext.kOnHost if not hasattr(ExecutionSpace, "kOnDevice") else dlext.kOnDevice


class Sampler(FixDLExt):
    def __init__(
        self, context, sampling_method, callback: Optional[Callable], location=kDefaultLocation
    ):
        super().__init__(context)

        on_gpu = (location != dlext.kOnHost) & has_kokkos_cuda_enabled(context)
        location = location if on_gpu else dlext.kOnHost

        self.context = context
        self.location = location
        self.view = LAMMPSView(context)

        helpers, restore, bias = build_helpers(context, sampling_method, on_gpu, pbs.restore)
        initial_snapshot = self.take_snapshot()
        _, initialize, method_update = sampling_method.build(initial_snapshot, helpers)

        self.callback = callback
        self.restore = lambda prev_snapshot: restore(self.snapshot, prev_snapshot)
        self.snapshot = initial_snapshot
        self.state = initialize()
        self._update_box = lambda: self.snapshot.box

        def update(timestep):
            self.view.synchronize()
            self.snapshot = self._update_snapshot()
            self.state = method_update(self.snapshot, self.state)
            bias(self.snapshot, self.state)
            if self.callback:
                self.callback(self.snapshot, self.state, timestep)

        self.set_callback(update)

    def _partial_snapshot(self, include_masses: bool = False):
        positions = asarray(dlext.positions(self.view, self.location))
        types = asarray(dlext.types(self.view, self.location))
        velocities = asarray(dlext.velocities(self.view, self.location))
        forces = asarray(dlext.forces(self.view, self.location))
        tags_map = asarray(dlext.tags_map(self.view, self.location))
        imgs = asarray(dlext.images(self.view, self.location))

        masses = None
        if include_masses:
            masses = asarray(dlext.masses(self.view, self.location))
        vel_mass = (velocities, (masses, types))

        return Snapshot(positions, vel_mass, forces, tags_map, imgs, None, None)

    def _update_snapshot(self):
        s = self._partial_snapshot()
        velocities, (_, types) = s.vel_mass
        _, (masses, _) = self.snapshot.vel_mass
        vel_mass = (velocities, (masses, types))
        box = self._update_box()
        dt = self.snapshot.dt

        return Snapshot(s.positions, vel_mass, s.forces, s.ids[1:], s.images, box, dt)

    def take_snapshot(self):
        s = self._partial_snapshot(include_masses=True)
        box = Box(*get_global_box(self.context))
        dt = get_timestep(self.context)

        return Snapshot(
            copy(s.positions), copy(s.vel_mass), copy(s.forces), s.ids[1:], copy(s.images), box, dt
        )


def build_helpers(context, sampling_method, on_gpu, restore_fn):
    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if on_gpu:
        cupy = importlib.import_module("cupy")
        view = cupy.asarray

        def sync_forces():
            cupy.cuda.get_current_stream().synchronize()

    else:
        utils = importlib.import_module(".utils", package="pysages.backends")
        view = utils.view

        def sync_forces():
            pass

    # TODO: check if this can be sped up.  # pylint: disable=W0511
    def bias(snapshot, state):
        """Adds the computed bias to the forces."""
        if state.bias is None:
            return
        forces = view(snapshot.forces)
        biases = view(state.bias.block_until_ready())
        forces[:, :3] += biases
        sync_forces()

    snapshot_methods = build_snapshot_methods(sampling_method, on_gpu)
    flags = sampling_method.snapshot_flags
    restore = partial(restore_fn, view)
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), get_dimension(context))

    return helpers, restore, bias


def build_snapshot_methods(sampling_method, on_gpu):
    if sampling_method.requires_box_unwrapping:
        device = jax.devices("gpu" if on_gpu else "cpu")[0]
        dtype = np.int64 if dlext.kImgBitSize == 64 else np.int32
        offset = dlext.kImgMax

        with jax.default_device(device):
            bits = np.asarray((0, dlext.kImgBits, dlext.kImg2Bits), dtype=dtype)
            mask = np.asarray((dlext.kImgMask, dlext.kImgMask, -1), dtype=dtype)

        def unpack(image):
            return (image >> bits & mask) - offset

        def positions(snapshot):
            L = np.diag(snapshot.box.H)
            return snapshot.positions[:, :3] + L * vmap(unpack)(snapshot.images)

    else:

        def positions(snapshot):
            return snapshot.positions

    @jit
    def indices(snapshot):
        return snapshot.ids

    @jit
    def momenta(snapshot):
        V, (masses, types) = snapshot.vel_mass
        M = masses[types]
        return (M * V).flatten()

    @jit
    def masses(snapshot):
        return snapshot.vel_mass[:, 3:]

    return SnapshotMethods(jit(positions), indices, momenta, masses)


def get_dimension(context):
    return context.extract_setting("dimension")


def get_global_box(context):
    boxlo, boxhi, xy, yz, xz, *_ = context.extract_box()
    Lx = boxhi[0] - boxlo[0]
    Ly = boxhi[1] - boxlo[1]
    Lz = boxhi[2] - boxlo[2]
    origin = boxlo
    H = ((Lx, xy * Ly, xz * Lz), (0.0, Ly, yz * Lz), (0.0, 0.0, Lz))
    return H, origin


def get_timestep(context):
    return context.extract_global("dt")


def bind(sampling_context: SamplingContext, callback: Optional[Callable], **kwargs):
    context = sampling_context.context
    sampling_method = sampling_context.method
    sampler = Sampler(context, sampling_method, callback)
    sampling_context.view = sampler.view
    sampling_context.run = lambda n, **kwargs: context.command(f"run {n}")

    # We want to support backends that also are context managers as long
    # as the simulation is kept alive after exiting the context.
    # Unfortunately, the default implementation of `lammps.__exit__` closes
    # the lammps instance, so we need to overwrite it.
    context.__exit__ = lambda *args: None

    return sampler
