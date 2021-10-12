# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES


import importlib
import jax
import pysages.backends.common as common

from functools import partial
from hoomd.dlext import (
    AccessLocation, AccessMode, HalfStepHook, SystemView,
    net_forces, positions_types, rtags, velocities_masses,
)
from jax.dlpack import from_dlpack as asarray
from pysages.backends.common import HelperMethods
from pysages.backends.snapshot import Box, Snapshot
from warnings import warn

from .core import ContextWrapper


# TODO: Figure out a way to automatically tie the lifetime of Sampler
# objects to the contexts they bind to
CONTEXTS_SAMPLERS = {}


class ContextWrapperHOOMD(ContextWrapper):
    def __init__(self, context):
        self.sysview = SystemView(context.system_definition)
        super().__init__(context)
        self.synchronize = self.sysview.synchronize


class Sampler(HalfStepHook):
    def __init__(self, method_bundle, bias):
        super().__init__()
        #
        snapshot, initialize, update = method_bundle
        self.snapshot = snapshot
        self.state = initialize()
        self._update = update
        self.bias = bias
    #
    def update(self, timestep):
        self.state = self._update(self.snapshot, self.state)
        self.bias(self.snapshot, self.state)


if hasattr(AccessLocation, "OnDevice"):
    def default_location():
        return AccessLocation.OnDevice
else:
    def default_location():
        return AccessLocation.OnHost


def is_on_gpu(context):
    return context.on_gpu()


def take_snapshot(wrapped_context, location = default_location()):
    #
    context = wrapped_context.context
    sysview = wrapped_context.sysview
    #
    positions = asarray(positions_types(sysview, location, AccessMode.Read))
    vel_mass = asarray(velocities_masses(sysview, location, AccessMode.Read))
    forces = asarray(net_forces(sysview, location, AccessMode.ReadWrite))
    ids = asarray(rtags(sysview, location, AccessMode.Read))
    #
    box = sysview.particle_data().getGlobalBox()
    L  = box.getL()
    xy = box.getTiltFactorXY()
    xz = box.getTiltFactorXZ()
    yz = box.getTiltFactorYZ()
    lo = box.getLo()
    H = (
        (L.x, xy * L.y, xz * L.z),
        (0.0,      L.y, yz * L.z),
        (0.0,      0.0,      L.z)
    )
    origin = (lo.x, lo.y, lo.z)
    dt = context.integrator.dt
    #
    return Snapshot(positions, vel_mass, forces, ids, Box(H, origin), dt)


def build_helpers(context):
    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if is_on_gpu(context):
        cupy = importlib.import_module("cupy")
        view = cupy.asarray
        #
        def sync_forces():
            cupy.cuda.get_current_stream().synchronize()
    else:
        utils = importlib.import_module(".utils", package = "pysages.backends")
        view = utils.view
        #
        def sync_forces(): pass
    #
    def indices(ids):
        return ids
    #
    def momenta(vel_mass):
        M = vel_mass[:, 3:]
        V = vel_mass[:, :3]
        return jax.numpy.multiply(M, V).flatten()
    #
    def bias(snapshot, state, sync_backend):
        """Adds the computed bias to the forces."""
        # TODO: check if this can be JIT compiled with numba.
        # Forces may be computed asynchronously on the GPU, so we need to
        # synchronize them before applying the bias.
        sync_backend()
        forces = view(snapshot.forces)
        biases = view(state.bias.block_until_ready())
        forces[:, :3] += biases
        sync_forces()
    #
    restore = partial(common.restore, view)
    #
    return HelperMethods(jax.jit(indices), jax.jit(momenta), restore), bias


def bind(context, sampling_method, **kwargs):
    #
    helpers, bias = build_helpers(context)
    #
    wrapped_context = ContextWrapperHOOMD(context)
    snapshot = take_snapshot(wrapped_context)
    method_bundle = sampling_method(snapshot, helpers)
    sync_and_bias = partial(bias, sync_backend = wrapped_context.synchronize)
    #
    sampler = Sampler(method_bundle, sync_and_bias)
    context.integrator.cpp_integrator.setHalfStepHook(sampler)
    #
    CONTEXTS_SAMPLERS[context] = sampler
    #
    return sampler


def detach(context):
    """
    If pysages was bound to this context, this removes the corresponding
    `Sampler` object.
    """
    if context in CONTEXTS_SAMPLERS:
        context.integrator.cpp_integrator.removeHalfStepHook()
        del CONTEXTS_SAMPLERS[context]
    else:
        warn("This context has no sampler bound to it.")
