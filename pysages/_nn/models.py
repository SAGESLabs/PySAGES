# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from dataclasses import dataclass
from itertools import chain
from jax.nn.initializers import variance_scaling
from typing import Callable

from .utils import JaxArray, rng_key

import jax.experimental.stax as stax
import jax.numpy as np


@dataclass
class MLP:
    """
    Contains the parameters of a multilayer-perceptron network with inner layers
    defined by `topology`, and activation function `σ` (defaults to `stax.Tanh`).
    """
    parameters: JaxArray
    apply:      Callable

    def __init__(self, indim, outdim, topology, activation = stax.Tanh, seed = 0):
        σ = activation
        # Concatenate inner layers and activation functions
        layers = list(chain.from_iterable((stax.Dense(i), σ) for i in topology))
        # Add output layer
        layers = [stax.Flatten] + layers + [stax.Dense(outdim)]
        # Build initialization and application functions for the network
        init, apply = stax.serial(*layers)
        # Randomly initialize network parameters with seed
        _, parameters = init(rng_key(seed), (-1, indim))
        #
        self.parameters = parameters
        self.apply = apply


@dataclass
class Siren:
    """
    Contains the parameters of a Siren network with inner layers defined by `topology`.
    """
    parameters: JaxArray
    apply:      Callable

    def __init__(self, indim, outdim, topology, omega = 1.0, seed = 0):
        input_weights = variance_scaling(1.0 / (3 * indim), "fan_in", "uniform")
        inner_weights = variance_scaling(2.0 / omega**2, "fan_in", "uniform")
        # Sine activation function
        σ = stax.elementwise(lambda x: np.sin(omega * x))
        # Build layers
        input_layer = [stax.Flatten, stax.Dense(topology[0], input_weights)]
        inner_layers = list(chain.from_iterable(
            (σ, stax.Dense(i, inner_weights)) for i in (*topology[1:], outdim)
        ))
        layers = input_layer + inner_layers
        # Build initialization and application functions for the network
        init, apply = stax.serial(*layers)
        # Randomly initialize network parameters with seed
        _, parameters = init(rng_key(seed), (-1, indim))
        #
        self.parameters = parameters
        self.apply = apply
