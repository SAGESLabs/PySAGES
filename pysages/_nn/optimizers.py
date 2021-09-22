# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from dataclasses import dataclass
from functools import partial
from jax.lax import cond
from jax.numpy.linalg import pinv
from jax.scipy.linalg import solve
from plum import Dispatcher
from typing import Any, Callable, NamedTuple, Tuple, Union

from .objectives import (
    Loss,
    L2Regularization,
    Regularizer,
    SSE,
    build_cost_function,
    build_error_function,
    build_damped_hessian,
    build_jac_err_prod,
    build_split_cost_function,
    sum_squares,
)
from .utils import (
    Bool,
    Float,
    Int,
    JaxArray,
    number_of_weights,
    pack,
    unpack,
)

import jax
import jax.experimental.optimizers as jopt
import jax.numpy as np


# Create a dispatcher for this submodule
dispatch = Dispatcher()


# Optimizers parameters

class AdamParams(NamedTuple):
    step_size: Union[Float, Callable] = 1e-2
    beta_1:    Float = 0.9
    beta_2:    Float = 0.999
    tol:       Float = 1e-8


class LevenbergMarquardtParams(NamedTuple):
    mu_0:    Float = 1e-1
    mu_c:    Float = 10.0
    mu_min:  Float = 1e-8
    mu_max:  Float = 1e8
    rho_c:   Float = 1e-1
    rho_min: Float = 1e-4


# Optimizers state

class WrappedState(NamedTuple):
    data:     Tuple[JaxArray, JaxArray]
    params:   Any
    iters:    Int = 0
    improved: Bool = True


class LevenbergMarquardtState(NamedTuple):
    data:     Tuple[JaxArray, JaxArray]
    params:   JaxArray
    errors:   JaxArray
    cost:     Float
    mu:       Float
    iters:    Int = 0
    improved: Bool = True


class LevenbergMarquardtBRState(NamedTuple):
    data:     Tuple[JaxArray, JaxArray]
    params:   JaxArray
    errors:   JaxArray
    cost:     Float
    mu:       Float
    alpha:    Float = 1e-4
    iters:    Int = 0
    improved: Bool = True


class Optimizer:
    pass


@dataclass
class Adam(Optimizer):
    params:    AdamParams = AdamParams()
    loss:      Loss = SSE()
    reg:       Regularizer = L2Regularization(0.0)
    tol:       Float = 1e-4
    max_iters: Int = 10000


@dataclass
class LevenbergMarquardt(Optimizer):
    params:    LevenbergMarquardtParams = LevenbergMarquardtParams()
    loss:      Loss = SSE()
    reg:       Regularizer = L2Regularization(0.0)
    max_iters: Int = 500


@dataclass
class LevenbergMarquardtBR(Optimizer):
    params:    LevenbergMarquardtParams = LevenbergMarquardtParams()
    alpha:     Float = np.float64(0.0)
    max_iters: Int = 500
    update:    Callable = lambda a, b, c, t: t


@dispatch
def build(optimizer: Adam, model):
    _init, _update, repack = jopt.adam(*optimizer.params)
    objective = build_cost_function(model, optimizer.loss, optimizer.reg)
    gradient = jax.grad(objective)
    max_iters = optimizer.max_iters
    _, layout = unpack(model.parameters)

    def initialize(params, x, y):
        wrapped_params = _init(pack(params, layout))
        return WrappedState((x, y), wrapped_params)

    def keep_iterating(state):
        return state.improved & (state.iters < max_iters)

    def update(state):
        data, params, iters, _ = state
        dp = gradient(repack(params), *data)
        params = _update(iters, dp, params)
        improved = sum_squares(unpack(dp)[0]) > optimizer.tol
        return WrappedState(data, params, iters + 1, improved)

    return initialize, keep_iterating, update


@dispatch
def build(optimizer: LevenbergMarquardt, model):
    error, objective = build_split_cost_function(model, optimizer.loss, optimizer.reg)
    jac_err_prod = build_jac_err_prod(optimizer.loss, optimizer.reg)
    damped_hessian = build_damped_hessian(optimizer.loss, optimizer.reg)
    jacobian = jax.jacobian(error)
    _, c, mu_min, mu_max, rho_c, rho_min = optimizer.params
    max_iters = optimizer.max_iters

    def initialize(params, x, y):
        e = error(params, x, y)
        mu = optimizer.params.mu_0
        return LevenbergMarquardtState((x, y), params, e, np.inf, mu)

    def keep_iterating(state):
        return state.improved & (state.iters < max_iters) & (state.mu < mu_max)

    def update(state):
        data, p_, e_, cost_, mu, iters, _ = state
        x, y = data
        mu = np.float32(mu)
        #
        J = jacobian(p_, x, y)
        H = damped_hessian(J, mu)
        Je = jac_err_prod(J, e_, p_)
        #
        dp = solve(H, Je, sym_pos = True)
        p = p_ - dp
        e = error(p, x, y)
        cost = objective(e, p)
        rho = (cost_ - cost) / (dp.T @ (mu * dp + Je))
        #
        mu = np.where(rho > rho_c, np.maximum(mu / c, mu_min), mu)
        #
        bad_step = (rho < rho_min) | np.any(np.isnan(p))
        mu = np.where(bad_step, np.minimum(c * mu, mu_max), mu)
        p = np.where(bad_step, p_, p)
        e = np.where(bad_step, e_, e)
        cost = np.where(bad_step, cost_, cost)
        improved = (cost_ > cost) | bad_step
        #
        return LevenbergMarquardtState(data, p, e, cost, mu, iters + ~bad_step, improved)

    return initialize, keep_iterating, update


@dispatch
def build(optimizer: LevenbergMarquardtBR, model):
    error = build_error_function(model, SSE())
    jacobian = jax.jacobian(error)
    _, c, mu_min, mu_max, rho_c, rho_min = optimizer.params
    max_iters = optimizer.max_iters
    #
    l = len(model.parameters) / 2 - 1
    k = unpack(model.parameters)[0].size
    update_hyperparams = partial(optimizer.update, l, k, optimizer.alpha)

    def initialize(params, x, y):
        e = error(params, x, y)
        mu = optimizer.params.mu_0
        gamma = np.float64(params.size)
        beta = (x.size / l)**2 * (x.size - gamma) / sum_squares(e)
        beta = np.where(beta < 0, 1.0, beta)
        alpha = gamma / sum_squares(params)
        return LevenbergMarquardtBRState((x, y), params, e, np.inf, mu, alpha / beta)

    def keep_iterating(state):
        return state.improved & (state.iters < max_iters) & (state.mu < mu_max)

    def update(state):
        data, p_, e_, cost_, mu, alpha, iters, _ = state
        x, y = data
        mu = np.float32(mu)
        alpha_ = np.float32(alpha)
        #
        J = jacobian(p_, x, y)
        H = J.T @ J
        Je = J.T @ e_ + alpha_ * p_
        I = np.diag_indices_from(H)
        #
        dp = solve(H.at[I].add(alpha_ + mu), Je, sym_pos = True)
        p = p_ - dp
        e = error(p, x, y)
        cost = (sum_squares(e) + alpha * sum_squares(p)) / 2
        rho = (cost_ - cost) / (dp.T @ (mu * dp + Je))
        #
        mu = np.where(rho > rho_c, np.maximum(mu / c, mu_min), mu)
        #
        bad_step = (rho < rho_min) | np.any(np.isnan(p))
        mu = np.where(bad_step, np.minimum(c * mu, mu_max), mu)
        p = np.where(bad_step, p_, p)
        e = np.where(bad_step, e_, e)
        #
        sse = sum_squares(e)
        ssp = sum_squares(p)
        cost = np.where(bad_step, cost_, cost)
        improved = (cost_ > cost) | bad_step
        #
        bundle = (alpha, H, I, sse, ssp, x.size)
        alpha, *_ = cond(bad_step, lambda t: t, update_hyperparams, bundle)
        cost = (sse + alpha * ssp) / 2
        #
        return LevenbergMarquardtBRState(
            data, p, e, cost, mu, alpha, iters + ~bad_step, improved
        )

    return initialize, keep_iterating, update


def update_hyperparams(nlayers, nparams, alpha_0, bundle):
    l, k = nlayers, nparams
    alpha, H, I, sse, ssp, n = bundle
    gamma = k - alpha * pinv(H.at[I].add(alpha)).trace()
    reset = np.isnan(gamma) | (gamma >= n) | (sse.sum() < 1e-4) | (ssp.sum() < 1e-4)
    beta = np.where(reset, 1.0, (n / l)**2 * (n - gamma) / sse)
    alpha = np.where(reset, alpha_0, gamma / ssp)
    return (alpha / beta, H, I, sse, ssp, n)
