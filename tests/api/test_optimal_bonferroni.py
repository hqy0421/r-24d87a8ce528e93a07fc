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
import itertools

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from deel.puncc.api import nonconformity_scores
from deel.puncc.api import prediction_sets
from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.corrections import bonferroni
from deel.puncc.api.corrections import fixed_allocation
from deel.puncc.api.corrections import optimal_bonferroni_allocation
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.regression import SplitCP


def _reference_quantile_grid(residuals):
    """Exact-rank quantiles of the residuals with an infinite score appended,
    computed per output and per grid level k/(n+1) with scalar calls, exactly
    as the native calibration path does."""
    n, d = residuals.shape
    k_grid = np.arange(1, n + 1)
    q_levels = 1.0 - k_grid / (n + 1)
    quantile_grid = np.empty((n, d))
    for j in range(d):
        column = np.concatenate([residuals[:, j], [np.inf]])
        quantile_grid[:, j] = [
            np.quantile(column, q, method="inverted_cdf") for q in q_levels
        ]
    return quantile_grid


def _weighted_width(quantile_grid, weights, allocation_k):
    """Weighted sum of the interval widths (2 * quantile) of an allocation."""
    return sum(
        weights[j] * 2 * quantile_grid[allocation_k[j] - 1, j]
        for j in range(len(allocation_k))
    )


def _brute_force_optimum(residuals, budget, weights):
    """Exhaustive enumeration of all the allocations on the grid."""
    n, d = residuals.shape
    quantile_grid = _reference_quantile_grid(residuals)
    tol = 8.0 * np.finfo(float).eps * (n + 1)
    max_total_k = int(np.floor(budget * (n + 1) + tol))
    best_value = np.inf
    for allocation_k in itertools.product(range(1, n + 1), repeat=d):
        if sum(allocation_k) > max_total_k:
            continue
        value = _weighted_width(quantile_grid, weights, allocation_k)
        best_value = min(best_value, value)
    return best_value


def _make_calibrator(y_pred_calib, y_calib):
    calibrator = BaseCalibrator(
        nonconf_score_func=nonconformity_scores.absolute_difference,
        pred_set_func=prediction_sets.constant_interval,
    )
    calibrator.fit(y_pred=y_pred_calib, y_true=y_calib)
    return calibrator


@pytest.mark.parametrize(
    "n, d, budget, seed",
    [
        (4, 2, 0.5, 0),
        (5, 3, 0.7, 1),
        (6, 2, 0.4, 2),
        (6, 3, 0.6, 3),
        (7, 4, 0.9, 4),
        (8, 2, 0.3, 5),
    ],
)
def test_allocation_matches_brute_force(n, d, budget, seed):
    # Residuals drawn from a small set of values to generate duplicated
    # order statistics and zero residuals.
    rng = np.random.RandomState(seed)
    residuals = rng.choice([0.0, 0.5, 1.0, 2.0, 5.0], size=(n, d))
    weights = rng.uniform(0.5, 3.0, size=d)

    alphas = optimal_bonferroni_allocation(residuals, budget, weights)

    # Levels are on the grid k/(n+1), k integer in [1, n]
    allocation_k = np.rint(alphas * (n + 1)).astype(int)
    assert np.allclose(alphas, allocation_k / (n + 1), rtol=0, atol=1e-12)
    assert np.all(allocation_k >= 1)
    assert np.all(allocation_k <= n)
    # Budget constraint
    assert np.sum(alphas) <= budget + 1e-9

    # Global optimality against exhaustive enumeration
    quantile_grid = _reference_quantile_grid(residuals)
    dp_value = _weighted_width(quantile_grid, weights, allocation_k)
    brute_value = _brute_force_optimum(residuals, budget, weights)
    assert dp_value <= brute_value + 1e-9
    assert np.isclose(dp_value, brute_value, rtol=1e-9, atol=1e-9)


def test_allocation_is_deterministic_on_ties():
    # Identical outputs and weights: many allocations are tied optimal.
    residuals = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    weights = np.array([1.0, 1.0])

    first = optimal_bonferroni_allocation(residuals, 0.6, weights)
    second = optimal_bonferroni_allocation(residuals, 0.6, weights)

    np.testing.assert_array_equal(first, second)
    # The deterministic allocation is one of the optima
    quantile_grid = _reference_quantile_grid(residuals)
    allocation_k = np.rint(first * 5).astype(int)
    value = _weighted_width(quantile_grid, weights, allocation_k)
    assert np.isclose(value, _brute_force_optimum(residuals, 0.6, weights))


def test_allocation_exact_rank_quantiles_no_approximation():
    # n = 4 calibration records, single output: every grid quantile must be
    # one of the raw residuals (exact rank, no interpolation), and finite
    # because k >= 1 (the appended infinite score is never reached).
    residuals = np.array([[3.0], [1.0], [4.0], [2.0]])
    weights = np.array([1.0])

    alphas = optimal_bonferroni_allocation(residuals, 0.8, weights)
    assert np.allclose(alphas, [4 / 5])

    calibrator = BaseCalibrator(
        nonconf_score_func=lambda y_pred, y_true: residuals,
        pred_set_func=prediction_sets.constant_interval,
    )
    calibrator.fit(y_pred=np.zeros((4, 1)), y_true=np.zeros((4, 1)))
    y_lo, y_hi = calibrator.calibrate(
        alpha=0.8, y_pred=np.zeros((1, 1)), correction=fixed_allocation(alphas)
    )
    # alpha = 4/5 -> q = 1/5 -> first order statistic of the residuals
    assert np.isfinite(y_lo).all() and np.isfinite(y_hi).all()
    assert (y_hi - y_lo)[0, 0] == 2 * 1.0

    # All grid quantiles are exact order statistics of the raw residuals
    quantile_grid = _reference_quantile_grid(residuals)
    for value in quantile_grid[:, 0]:
        assert value in set(residuals[:, 0])


def test_allocation_insufficient_budget_has_no_finite_solution():
    residuals = np.abs(np.random.RandomState(0).randn(9, 3))
    weights = np.ones(3)

    # Minimum budget for all-finite intervals is d/(n+1) = 3/10
    with pytest.raises(ValueError, match="No finite solution"):
        optimal_bonferroni_allocation(residuals, 0.29, weights)

    # Boundary budget is feasible: every output gets k = 1
    alphas = optimal_bonferroni_allocation(residuals, 0.3, weights)
    np.testing.assert_allclose(alphas, np.full(3, 0.1))


@pytest.mark.parametrize(
    "budget", [0.0, 1.0, 1.5, -0.2, np.nan, np.inf, "0.5", [0.1], True]
)
def test_allocation_rejects_invalid_budget(budget):
    residuals = np.abs(np.random.RandomState(0).randn(5, 2))
    with pytest.raises(ValueError, match="budget"):
        optimal_bonferroni_allocation(residuals, budget, np.ones(2))


@pytest.mark.parametrize(
    "weights",
    [
        np.array([0.0, 1.0]),
        np.array([-1.0, 2.0]),
        np.array([1.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 2.0]]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
    ],
)
def test_allocation_rejects_invalid_weights(weights):
    residuals = np.abs(np.random.RandomState(0).randn(5, 2))
    with pytest.raises(ValueError, match="weights"):
        optimal_bonferroni_allocation(residuals, 0.5, weights)


@pytest.mark.parametrize(
    "residuals",
    [
        np.array([1.0, 2.0, 3.0]),  # 1D
        np.zeros((3, 2, 2)),  # 3D
        np.empty((0, 2)),  # no calibration record
        np.empty((3, 0)),  # no output
        np.array([[1.0, np.nan], [2.0, 3.0]]),  # NaN
        np.array([[1.0, np.inf], [2.0, 3.0]]),  # infinite
        np.array([[1.0, -2.0], [2.0, 3.0]]),  # negative
    ],
)
def test_allocation_rejects_invalid_residuals(residuals):
    d = residuals.shape[1] if residuals.ndim == 2 and residuals.shape[1] else 1
    with pytest.raises(ValueError, match="residuals"):
        optimal_bonferroni_allocation(residuals, 0.5, np.ones(d))


def test_allocation_zero_residuals_get_minimal_budget():
    # Output 0 is perfectly predicted (zero residuals): it must receive the
    # minimal level k = 1 and the spare budget must go to output 1.
    residuals = np.array(
        [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0]]
    )
    alphas = optimal_bonferroni_allocation(residuals, 0.8, np.ones(2))
    n = residuals.shape[0]
    allocation_k = np.rint(alphas * (n + 1)).astype(int)
    assert allocation_k[0] == 1
    # K_max = floor(0.8 * 6) = 4; the spare budget goes to output 1
    assert allocation_k[1] == 3

    # All-zero residuals: every allocation is optimal (zero width), the
    # returned one is deterministic and on the grid.
    zero_residuals = np.zeros((5, 2))
    first = optimal_bonferroni_allocation(zero_residuals, 0.5, np.ones(2))
    second = optimal_bonferroni_allocation(zero_residuals, 0.5, np.ones(2))
    np.testing.assert_array_equal(first, second)
    assert np.sum(first) <= 0.5 + 1e-9


def test_allocation_weights_shift_budget():
    # Identical outputs with linearly decreasing width curves: the optimal
    # allocation gives the spare budget to the output with the larger
    # business weight.
    residuals = np.column_stack([np.linspace(1.0, 100.0, 100)] * 2)

    alphas = optimal_bonferroni_allocation(residuals, 0.1, np.array([2.0, 1.0]))
    np.testing.assert_array_equal(np.rint(alphas * 101).astype(int), [9, 1])

    alphas = optimal_bonferroni_allocation(residuals, 0.1, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(np.rint(alphas * 101).astype(int), [1, 9])

    # Heterogeneous scales with equal weights: the noisy output (large
    # residuals) receives more budget than the precise one.
    residuals = np.column_stack(
        [np.linspace(0.001, 0.01, 100), np.linspace(1.0, 100.0, 100)]
    )
    alphas = optimal_bonferroni_allocation(residuals, 0.1, np.ones(2))
    k = np.rint(alphas * 101).astype(int)
    assert k[1] > k[0]


def test_allocation_single_output_2d():
    residuals = np.array([[2.0], [1.0], [4.0], [3.0], [5.0]])
    # K_max = floor(0.9 * 6) = 5 = n -> largest admissible k
    alphas = optimal_bonferroni_allocation(residuals, 0.9, np.ones(1))
    np.testing.assert_allclose(alphas, [5 / 6])
    # K_max = floor(0.4 * 6) = 2
    alphas = optimal_bonferroni_allocation(residuals, 0.4, np.ones(1))
    np.testing.assert_allclose(alphas, [2 / 6])


def test_fixed_allocation_returns_precomputed_levels():
    alphas = np.array([0.05, 0.1, 0.15])
    correction = fixed_allocation(alphas)

    # The nominal level is ignored: the precomputed allocation is returned
    np.testing.assert_array_equal(correction(0.4), alphas)
    np.testing.assert_array_equal(correction(0.01), alphas)


@pytest.mark.parametrize(
    "alphas",
    [
        np.array([]),
        np.array([[0.1, 0.2]]),
        np.array([0.0, 0.2]),
        np.array([0.1, 1.0]),
        np.array([0.1, np.nan]),
        np.array([0.1, np.inf]),
        np.array([-0.1, 0.2]),
    ],
)
def test_fixed_allocation_rejects_invalid_alphas(alphas):
    with pytest.raises(ValueError, match="alphas"):
        fixed_allocation(alphas)


def test_calibrator_allocation_methods_require_fit():
    calibrator = BaseCalibrator(
        nonconf_score_func=nonconformity_scores.absolute_difference,
        pred_set_func=prediction_sets.constant_interval,
    )

    with pytest.raises(RuntimeError):
        calibrator.compute_optimal_bonferroni_allocation(
            budget=0.5, weights=np.ones(2)
        )
    with pytest.raises(RuntimeError):
        calibrator.calibrate_optimal_bonferroni(
            budget=0.5, weights=np.ones(2), y_pred=np.zeros((2, 2))
        )


def test_calibrate_optimal_bonferroni_drives_native_calibrator():
    rng = np.random.RandomState(42)
    n, d = 60, 3
    y_pred_calib = rng.randn(n, d)
    y_calib = y_pred_calib + rng.randn(n, d) * np.array([0.1, 1.0, 3.0])
    y_pred_test = rng.randn(7, d)
    budget = 0.2
    weights = np.array([1.0, 2.0, 3.0])

    calibrator = _make_calibrator(y_pred_calib, y_calib)

    # The selected alphas drive the native calibrator: same intervals as the
    # native calibrate path frozen on the precomputed allocation.
    y_lo, y_hi = calibrator.calibrate_optimal_bonferroni(
        budget=budget, weights=weights, y_pred=y_pred_test
    )
    alphas = calibrator.compute_optimal_bonferroni_allocation(
        budget=budget, weights=weights
    )
    y_lo_ref, y_hi_ref = calibrator.calibrate(
        alpha=budget, y_pred=y_pred_test, correction=fixed_allocation(alphas)
    )
    np.testing.assert_array_equal(y_lo, y_lo_ref)
    np.testing.assert_array_equal(y_hi, y_hi_ref)

    # Widths are the exact-rank quantiles of the raw residuals + infinity
    residuals = calibrator.get_nonconformity_scores()
    quantile_grid = _reference_quantile_grid(residuals)
    allocation_k = np.rint(alphas * (n + 1)).astype(int)
    for j in range(d):
        np.testing.assert_allclose(
            (y_hi - y_lo)[:, j],
            2 * quantile_grid[allocation_k[j] - 1, j],
            rtol=0,
            atol=1e-12,
        )

    assert np.all(np.isfinite(y_lo)) and np.all(np.isfinite(y_hi))
    assert np.all(y_lo <= y_hi)

    # The allocation is not an equal split of the budget
    equal_split = bonferroni(budget, nvars=d)
    assert not np.allclose(alphas, equal_split)


def test_calibrate_optimal_bonferroni_improves_over_equal_split():
    # Heterogeneous output scales: the optimized allocation strictly
    # improves the weighted width w.r.t the equal-split correction.
    residuals = np.column_stack(
        [np.linspace(0.001, 0.01, 100), np.linspace(1.0, 100.0, 100)]
    )
    budget = 0.1
    weights = np.ones(2)

    calibrator = BaseCalibrator(
        nonconf_score_func=lambda y_pred, y_true: residuals,
        pred_set_func=prediction_sets.constant_interval,
    )
    calibrator.fit(y_pred=np.zeros((100, 2)), y_true=np.zeros((100, 2)))
    y_pred_test = np.zeros((3, 2))

    y_lo_opt, y_hi_opt = calibrator.calibrate_optimal_bonferroni(
        budget=budget, weights=weights, y_pred=y_pred_test
    )
    y_lo_eq, y_hi_eq = calibrator.calibrate(
        alpha=budget,
        y_pred=y_pred_test,
        correction=lambda alpha: bonferroni(alpha, nvars=2),
    )

    optimized_width = np.sum(weights * (y_hi_opt - y_lo_opt)[0])
    equal_split_width = np.sum(weights * (y_hi_eq - y_lo_eq)[0])
    assert optimized_width < equal_split_width


def test_split_cp_pipeline_with_fixed_allocation():
    # End-to-end: the allocation computed from the fitted residuals drives
    # the native conformal predictor for new predictions.
    X, y = make_regression(
        n_samples=300, n_features=4, n_targets=2, random_state=0
    )
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.3, random_state=0
    )

    predictor = BasePredictor(
        RandomForestRegressor(n_estimators=20, random_state=0),
        is_trained=False,
    )
    split_cp = SplitCP(predictor)
    split_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)

    residuals = split_cp.get_nonconformity_scores()
    budget = 0.2
    weights = np.array([1.0, 4.0])
    alphas = optimal_bonferroni_allocation(residuals, budget, weights)

    y_pred, y_lo, y_hi = split_cp.conformal_predictor.predict(
        X_test, alpha=budget, correction_func=fixed_allocation(alphas)
    )

    n = residuals.shape[0]
    quantile_grid = _reference_quantile_grid(residuals)
    allocation_k = np.rint(alphas * (n + 1)).astype(int)
    for j in range(2):
        np.testing.assert_allclose(
            (y_hi - y_lo)[:, j],
            2 * quantile_grid[allocation_k[j] - 1, j],
            rtol=0,
            atol=1e-12,
        )

    assert y_pred.shape == y_lo.shape == y_hi.shape
    assert np.all(np.isfinite(y_lo)) and np.all(np.isfinite(y_hi))
    assert np.all(y_lo <= y_hi)
    assert np.sum(alphas) <= budget + 1e-9
