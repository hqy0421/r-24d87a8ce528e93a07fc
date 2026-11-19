# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module provides correction functions for multiple hypothesis testing. To be used
when building a :ref:`calibrator <calibration>` for multivariate regression or object detection.
"""
import numbers
from typing import Callable
from typing import Union

import numpy as np


def bonferroni(alpha: float, nvars: int = 1) -> Union[float, np.ndarray]:
    """Bonferroni correction for multiple comparisons.

    :param float alpha: nominal coverage level.
    :param int nvars: number of output features.


    :returns: corrected coverage level.
    :rtype: float or ndarray.
    """
    # Sanity checks
    if np.any(alpha <= 0) or np.any(alpha >= 1):
        raise ValueError("alpha must be in (0,1)")

    if nvars <= 0:
        raise ValueError("nvars must be a positive integer")

    if nvars == 1:
        return alpha

    return np.ones(nvars) * alpha / nvars


def weighted_bonferroni(alpha: float, weights: np.ndarray) -> np.ndarray:
    """Weighted Bonferroni correction for multiple comparisons.

    :param float alpha: nominal coverage level.
    :param np.ndarray weights: weights associated to each output feature.


    :returns: array of corrected featurewise coverage levels.
    :rtype: np.ndarray.
    """
    # Sanity checks
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0,1)")

    # Positivity check
    positiveness_condition = np.all(weights > 0)
    if not positiveness_condition:
        raise RuntimeError("weights must be positive")

    # Normalization check
    norm_condition = np.isclose(np.sum(weights) - 1, 0, atol=1e-14)
    if not norm_condition:
        error = (
            "weights are not normalized. Sum of weights is"
            + f"{np.sum(weights)}"
        )
        raise RuntimeError(error)

    return alpha * weights


def fixed_allocation(alphas: np.ndarray) -> Callable:
    """Build a correction function that maps any nominal miscoverage level to
    a fixed, precomputed per-output allocation of a Bonferroni budget.

    The returned callable has the same interface as :func:`bonferroni` and can
    be passed as the ``correction`` (resp. ``correction_func``) argument of a
    calibrator's (resp. conformal predictor's) method. It ignores the nominal
    level it receives and always returns :data:`alphas`, so the native
    calibration path is driven by the precomputed per-output levels instead of
    an equal split of the nominal level.

    :param np.ndarray alphas: precomputed per-output miscoverage levels. Must
        be a non-empty one-dimensional array with all coordinates in (0, 1).

    :returns: correction function mapping any nominal level to :data:`alphas`.
    :rtype: Callable

    :raises ValueError: if :data:`alphas` is not a non-empty one-dimensional
        array with all coordinates in (0, 1).
    """
    alphas = np.asarray(alphas, dtype=float)

    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError(
            "alphas must be a non-empty one-dimensional array. "
            + f"Got array of shape {alphas.shape}."
        )

    if not np.all(np.isfinite(alphas)):
        raise ValueError("alphas must contain only finite values.")

    if np.any(alphas <= 0) or np.any(alphas >= 1):
        raise ValueError("All coordinates of alphas must be in (0, 1).")

    def _correction(alpha: float) -> np.ndarray:
        del alpha  # The allocation is precomputed; the nominal level is unused.
        return alphas

    return _correction


def _check_allocation_inputs(residuals, budget, weights):
    """Check the inputs of :func:`optimal_bonferroni_allocation` and return
    them normalized: 2D float residuals, float budget and 1D float weights.

    :raises ValueError: invalid residuals, budget or weights.
    """
    try:
        residuals = np.asarray(residuals, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "residuals must be a 2D array-like of absolute residuals."
        ) from exc

    if residuals.ndim != 2:
        raise ValueError(
            "residuals must be a 2D array of shape "
            + f"(n_calibration, n_outputs); got ndim={residuals.ndim}."
        )

    n, n_outputs = residuals.shape

    if n < 1 or n_outputs < 1:
        raise ValueError(
            "residuals must contain at least one calibration record and one "
            + f"output; got shape {residuals.shape}."
        )

    if not np.all(np.isfinite(residuals)):
        raise ValueError(
            "residuals must contain only finite values (no NaN or infinity)."
        )

    if np.any(residuals < 0):
        raise ValueError("residuals must be non-negative (absolute residuals).")

    if isinstance(budget, bool) or not isinstance(budget, numbers.Real):
        raise ValueError(
            f"budget must be a scalar in the open interval (0, 1); got {budget!r}."
        )
    budget = float(budget)
    if not np.isfinite(budget) or budget <= 0 or budget >= 1:
        raise ValueError(
            f"budget must be in the open interval (0, 1); got {budget}."
        )

    try:
        weights = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "weights must be a 1D array-like of positive numbers."
        ) from exc

    if weights.shape != (n_outputs,):
        raise ValueError(
            "weights must be a 1D array with one positive weight per output "
            + f"({n_outputs} expected); got shape {weights.shape}."
        )

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError(
            "weights must contain only finite, strictly positive values."
        )

    return residuals, budget, weights


def optimal_bonferroni_allocation(
    residuals: np.ndarray, budget: float, weights: np.ndarray
) -> np.ndarray:
    """Optimal nominal Bonferroni allocation of a miscoverage budget across
    the outputs of a multivariate regression, based on the unweighted
    calibration absolute residuals.

    Splitting the budget equally across outputs (the standard Bonferroni
    correction) makes the intervals of expensive or noisy outputs overly wide.
    Given :math:`n` calibration records and :math:`d` outputs, this function
    restricts the per-output miscoverage levels to the exact grid

    .. math::

        \\alpha_j \\in \\{k/(n+1),\\ k = 1, \\dots, n\\},
        \\qquad \\sum_{j=1}^{d} \\alpha_j \\leq B,

    where :math:`B` is the nominal budget, and selects the allocation that
    minimizes the business-weighted sum of the interval widths in the
    original units:

    .. math::

        \\min_{(\\alpha_1, \\dots, \\alpha_d)} \\ \\sum_{j=1}^{d} w_j \\cdot
        2 \\, Q_j(\\alpha_j),

    where :math:`Q_j(\\alpha_j)` is the :math:`(1 - \\alpha_j)`-th empirical
    quantile (inverted CDF, i.e. exact rank) of the :math:`n` raw absolute
    residuals of output :math:`j` augmented with an infinite score, and the
    interval of output :math:`j` is :math:`[\\hat{y}_j - Q_j, \\hat{y}_j +
    Q_j]`. The constant factor 2 does not affect the minimizer.

    The global minimum is computed exactly by dynamic programming over the
    integer budget :math:`\\sum_j k_j` (a greedy allocation would be
    suboptimal for non-concave width curves). Ties between optimal
    allocations are broken deterministically. Duplicated and zero residuals
    are supported; grid points dominated by a smaller :math:`k` with the same
    quantile are discarded.

    .. note::

        The weights only enter the width objective above; they do not
        reweight the calibration scores. The result is a *nominal* Bonferroni
        split of the budget (the per-output levels sum to at most
        :data:`budget`), chosen adaptively from the calibration residuals.
        This function does not claim any coverage guarantee beyond the
        nominal Bonferroni interpretation of the budget.

    :param np.ndarray residuals: absolute residuals computed on the
        calibration set, of shape (n_calibration, n_outputs). All values must
        be finite and non-negative.
    :param float budget: nominal miscoverage budget to distribute across the
        outputs. Must be in the open interval (0, 1). The budget is feasible
        only if it is at least ``n_outputs / (n_calibration + 1)``, the
        smallest total level for which every output gets a finite interval.
    :param np.ndarray weights: positive business weights of the width
        objective, one per output.

    :returns: per-output miscoverage levels of the optimal allocation. Each
        coordinate is of the form ``k / (n_calibration + 1)`` with integer
        ``1 <= k <= n_calibration`` and the coordinates sum to at most
        :data:`budget`.
    :rtype: np.ndarray

    :raises ValueError: if :data:`residuals` is not a 2D array of finite
        non-negative values, if :data:`budget` is not a scalar in (0, 1), if
        :data:`weights` is not a 1D array of finite strictly positive values
        with one entry per output, or if the budget is too small to make all
        output intervals finite (no finite solution).
    """
    # ----------------------------- Input checks -----------------------------
    residuals, budget, weights = _check_allocation_inputs(
        residuals, budget, weights
    )
    n, n_outputs = residuals.shape

    # -------------------------- Feasibility check ---------------------------
    n_grid = n + 1  # n residuals + the appended infinite score
    # Largest integer K such that K/(n+1) <= budget. A small tolerance
    # absorbs the floating-point representation error of budget*(n+1).
    tol = 8.0 * np.finfo(float).eps * n_grid
    max_total_k = int(np.floor(budget * n_grid + tol))

    if max_total_k < n_outputs:
        raise ValueError(
            "No finite solution: the budget is too small to make all output "
            + f"intervals finite. With n={n} calibration records and "
            + f"{n_outputs} outputs, each output needs alpha >= 1/{n_grid}, "
            + f"so the budget must be at least {n_outputs}/{n_grid} "
            + f"(~{n_outputs / n_grid:.6g}); got budget={budget}."
        )

    # -------------------- Per-output exact width curves ---------------------
    k_grid = np.arange(1, n + 1)  # k = 1 .. n
    alpha_grid = k_grid / n_grid
    # Quantile levels computed exactly as in the native calibration path:
    # q = 1 - alpha with alpha = k/(n+1).
    q_levels = 1.0 - alpha_grid

    # Exact-rank quantiles of the raw residuals with an infinite score
    # appended (Tibshirani's lemma). Row k-1 holds the quantile associated
    # to alpha = k/(n+1).
    lemma_residuals = np.concatenate(
        (residuals, np.full((1, n_outputs), np.inf)), axis=0
    )
    quantile_grid = np.empty((n, n_outputs))
    for j in range(n_outputs):
        quantile_grid[:, j] = np.quantile(
            lemma_residuals[:, j], q_levels, method="inverted_cdf"
        )

    # A grid point k is dominated by a smaller k' < k whenever the quantile
    # does not strictly decrease; dominated points can never belong to an
    # optimal allocation and are discarded. This collapses duplicated and
    # zero residuals.
    kept_k = []
    kept_cost = []
    max_useful_k_sum = 0
    for j in range(n_outputs):
        q_j = quantile_grid[:, j]
        is_record = np.concatenate(([True], q_j[1:] < q_j[:-1]))
        k_j = k_grid[is_record]
        kept_k.append(k_j)
        kept_cost.append(weights[j] * q_j[is_record])
        max_useful_k_sum += int(k_j[-1])

    # ------------- Exact global optimization by dynamic programming ---------
    # E[c] = minimal weighted width achievable with the processed outputs and
    # a total grid index exactly equal to c.
    budget_cap = min(max_total_k, max_useful_k_sum)
    E = np.full(budget_cap + 1, np.inf)
    E[0] = 0.0
    choices = []  # choices[j][c] = k allocated to output j at total c
    for j in range(n_outputs):
        E_cur = np.full(budget_cap + 1, np.inf)
        arg_k = np.zeros(budget_cap + 1, dtype=np.int64)
        for k_val, cost in zip(kept_k[j], kept_cost[j]):
            k_int = int(k_val)
            if k_int > budget_cap:
                break  # kept_k[j] is sorted in ascending order
            candidate = E[: budget_cap + 1 - k_int] + cost
            target = E_cur[k_int:]
            better = candidate < target
            target[better] = candidate[better]
            arg_k[k_int:][better] = k_int
        E = E_cur
        choices.append(arg_k)

    # Among all optimal allocations, the first minimizer (smallest total k,
    # i.e. smallest total alpha) is selected; ties inside the DP are broken
    # toward smaller k. The procedure is fully deterministic.
    total_k = int(np.argmin(E))
    if not np.isfinite(E[total_k]):  # Defensive: cannot happen, see checks.
        raise ValueError(
            "No finite solution: the budget is too small to make all output "
            + "intervals finite."
        )

    allocation_k = np.zeros(n_outputs, dtype=np.int64)
    remaining = total_k
    for j in range(n_outputs - 1, -1, -1):
        allocation_k[j] = choices[j][remaining]
        remaining -= int(allocation_k[j])

    return allocation_k / n_grid
