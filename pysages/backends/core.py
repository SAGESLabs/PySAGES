# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import importlib
import jax
import warnings

from typing import Callable
from jax import numpy as np


# Set default floating point type for arrays in `jax` to `jax.f64`
jax.config.update("jax_enable_x64", True)


# Records the backend selected with `set_backend`
_CURRENT_BACKEND = None


class ContextWrapper:
    def __init__(self, view, context):
        self.view = view(context)
        self.context = context
        self.synchronize = self.view.synchronize


def current_backend():
    if _CURRENT_BACKEND is not None:
        return _CURRENT_BACKEND
    warnings.warn("No backend has been set")


def supported_backends():
    return ("hoomd", "openmm")


def set_backend(name):
    """To see a list of possible backends run `supported_backends()`."""
    #
    global _CURRENT_BACKEND
    #
    if name in supported_backends():
        _CURRENT_BACKEND = importlib.import_module('.' + name, package="pysages.backends")
    else:
        raise ValueError("Invalid backend")
    #
    return _CURRENT_BACKEND


def bind(context, sampling_method, callback: Callable = None, **kwargs):
    """Couples the sampling method to the simulation."""
    #
    if type(context).__module__.startswith("hoomd"):
        set_backend("hoomd")
    elif type(context).__module__.startswith("simtk.openmm"):
        set_backend("openmm")
    #
    check_backend_initialization()
    #
    return _CURRENT_BACKEND.bind(context, sampling_method, callback, **kwargs)


def check_backend_initialization():
    if _CURRENT_BACKEND is None:
        raise RuntimeError("No backend has been set")
