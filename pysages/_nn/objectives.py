# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from jax import value_and_grad, vmap
from plum import dispatch
from typing import NamedTuple

from .utils import (
    Float,
    number_of_weights,
    pack,
    prod,
    sum_squares,
    unpack,
)

import jax.numpy as np
import numpy as onp


# Losses
class Loss:
    pass


class GradientsLoss(Loss):
    pass


class Sobolev1Loss(Loss):
    pass


class SSE(Loss):
    pass


class GradientsSSE(GradientsLoss):
    pass


class Sobolev1SSE(Sobolev1Loss):
    pass


# Regularizers
class Regularizer:
    pass


class L2Regularization(Regularizer, NamedTuple):
    coeff: Float


@dispatch
def build_cost_function(model, loss: Loss, reg: Regularizer):
    cost = build_cost_function(loss, reg)

    def objective(params, inputs, reference):
        prediction = model.apply(params, inputs).reshape(reference.shape)
        e = np.asarray(prediction - reference, dtype = np.float32).flatten()
        ps, _ = unpack(params)
        return cost(e, ps)

    return objective


@dispatch
def build_cost_function(model, loss: Sobolev1Loss, reg: Regularizer):
    apply = value_and_grad(
        lambda p, x: model.apply(p, x.reshape(1, -1)).sum(), argnums = 1
    )
    cost = build_cost_function(loss, reg)

    def objective(params, inputs, refs):
        reference, refgrads = refs
        prediction, gradients = vmap(lambda x: apply(params, x))(inputs)
        prediction = prediction.reshape(reference.shape)
        gradients = gradients.reshape(refgrads.shape)
        e = np.asarray(prediction - reference, dtype = np.float32).flatten()
        ge = np.asarray(gradients - refgrads, dtype = np.float32).flatten()
        ps, _ = unpack(params)
        return cost((e, ge), ps)

    return objective


@dispatch
def build_cost_function(loss: SSE, reg: L2Regularization):
    r = reg.coeff

    def objective(errors, ps):
        return (sum_squares(errors) + r * sum_squares(ps)) / 2

    return objective


@dispatch
def build_cost_function(loss: Sobolev1SSE, reg: L2Regularization):
    r = reg.coeff

    def objective(errors, ps):
        e, ge = errors
        return (sum_squares(e) + sum_squares(ge) + r * sum_squares(ps)) / 2

    return objective


@dispatch
def build_error_function(model, loss: Loss):
    _, layout = unpack(model.parameters)

    def error(ps, inputs, reference):
        params = pack(ps, layout)
        prediction = model.apply(params, inputs).reshape(reference.shape)
        return np.asarray(prediction - reference, dtype = np.float32).flatten()

    return error


@dispatch
def build_error_function(model, loss: Sobolev1Loss):
    apply = value_and_grad(
        lambda p, x: model.apply(p, x.reshape(1, -1)).sum(), argnums = 1
    )
    _, layout = unpack(model.parameters)

    def error(ps, inputs, refs):
        params = pack(ps, layout)
        reference, refgrads = refs
        #prediction, gradients = apply(params, inputs)
        #gradients = grad_apply(params, inputs).reshape(refgrads.shape)
        prediction, gradients = vmap(lambda x: apply(params, x))(inputs)
        prediction = prediction.reshape(reference.shape)
        gradients = gradients.reshape(refgrads.shape)
        e = np.asarray(prediction - reference, dtype = np.float32).flatten()
        ge = np.asarray(gradients - refgrads, dtype = np.float32).flatten()
        return (e, ge)

    return error


def build_split_cost_function(model, loss, reg):
    error = build_error_function(model, loss)
    objective = build_cost_function(loss, reg)
    return error, objective


@dispatch
def build_damped_hessian(loss: Loss, reg: L2Regularization):
    r = reg.coeff

    def dhessian(J, mu):
        H = J.T @ J
        I = np.diag_indices_from(H)
        return H.at[I].add(r + mu)

    return dhessian


@dispatch
def build_damped_hessian(loss: Sobolev1Loss, reg: L2Regularization):
    r = reg.coeff

    def dhessian(jacs, mu):
        J, gJ = jacs
        H = J.T @ J + gJ.T @ gJ
        I = np.diag_indices_from(H)
        return H.at[I].add(r + mu)

    return dhessian


@dispatch
def build_jac_err_prod(loss: Loss, reg: L2Regularization):
    r = reg.coeff

    def jep(J, e, ps):
        return J.T @ e + r * ps

    return jep


@dispatch
def build_jac_err_prod(loss: Sobolev1Loss, reg: L2Regularization):
    r = reg.coeff

    def jep(jacs, errors, ps):
        J, gJ = jacs
        e, ge = errors
        return J.T @ e + gJ.T @ ge + r * ps

    return jep


def estimate_l2_coefficient(topology, grid):
    # Grid dimensionality
    d = grid.shape.size
    # Polynomial degree estimate
    k = onp.ceil(onp.sqrt(grid.shape.sum())) + 1
    # Number of weights for reasonably-sized single-hidden-layer NN
    n = number_of_weights((d, k, d))
    # Number of weights for the chosen topology
    p = number_of_weights((d, *topology, d))
    # If we have too few parameters the regularization term should be
    # negligible, for too many parameters the hyperparameter value
    # `len(topology)**2 / prod(grid.shape)` seems to work fine irrespectively
    # of the number of weights. Hence, we use a sigmoid to estimate the
    # regularization coeffiecient.
    return len(topology)**2 / prod(grid.shape) / (1 + onp.exp((n - p) / 2))
