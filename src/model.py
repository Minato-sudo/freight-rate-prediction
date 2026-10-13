"""The rate model: a robust log-linear base plus a gradient-boosted correction.

Why this shape
--------------
The validation window (Nov-Dec 2025) sits entirely *after* the development
window (Jan-Oct 2025), and log-rates drift upward across the training period.
A tree model cannot extrapolate that drift: its predictions flatten at the last
value it saw, which leaves a -3.4% shortfall two months out. A linear model
extrapolates the drift but misses the curvature in distance and weight and the
regional premium carried by the coordinates.

So the base is a trimmed-OLS fit on log(rate) carrying every effect that has to
extrapolate - distance elasticity, both market signals, the time trend and
equipment - and LightGBM then learns what the linear form left behind from
features that all stay inside ranges already observed.

Working in logs is deliberate: rates span 57 to 25,533 dollars, and the error
that matters commercially is proportional, not absolute. It also turns the
multiplicative label corruption into a constant offset, which is what makes the
corrupted rows separable by a single ratio gate.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CORRUPT_HIGH, CORRUPT_LOW, RANDOM_STATE
from .features import GBM_FEATURES, LINEAR_FEATURES


def _design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    matrix = frame[columns].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(frame)), matrix])


def fit_trimmed_ols(
    frame: pd.DataFrame,
    target: np.ndarray,
    columns: list[str],
    max_iter: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Least squares on log-rate, re-fitted with corrupted labels excluded.

    Returns the coefficients and the boolean mask of rows judged clean. The
    gate is multiplicative: a label is kept when it lands within
    [CORRUPT_LOW, CORRUPT_HIGH] times the model's expectation. Those bounds sit
    inside genuinely empty regions of the ratio histogram - no training row
    falls between 0.50 and 0.86, or between 1.21 and 2.19 - so the rule is a
    threshold on an observed gap rather than a tuned hyper-parameter.
    """
    design = _design(frame, columns)
    keep = np.ones(len(frame), dtype=bool)
    coefficients = np.zeros(design.shape[1])

    for _ in range(max_iter):
        coefficients, *_ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
        ratio = np.exp(target - design @ coefficients)
        proposed = (ratio > CORRUPT_LOW) & (ratio < CORRUPT_HIGH)
        if proposed.sum() < 0.5 * len(frame):  # refuse to trim away the data
            break
        if np.array_equal(proposed, keep):
            break
        keep = proposed

    return coefficients, keep


GBM_PARAMS = dict(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
    random_state=RANDOM_STATE,
)


class HybridRateModel:
    """Trimmed log-linear base + LightGBM residual correction."""

    def __init__(self, gbm_params: dict | None = None, use_gbm: bool = True):
        self.gbm_params = {**GBM_PARAMS, **(gbm_params or {})}
        self.use_gbm = use_gbm
        self.smearing = 1.0
        self.gbm = None

    def fit(self, frame: pd.DataFrame, coords: pd.DataFrame | None = None):
        """`coords` is accepted for call-site symmetry and is unused."""
        target = np.log(frame["posted_rate"].to_numpy(dtype=float))

        self.coefficients, self.clean_mask = fit_trimmed_ols(
            frame, target, LINEAR_FEATURES
        )
        self.n_dropped = int((~self.clean_mask).sum())

        base = _design(frame, LINEAR_FEATURES) @ self.coefficients
        clean = frame.loc[self.clean_mask]
        clean_residual = (target - base)[self.clean_mask]

        if self.use_gbm:
            self.gbm = lgb.LGBMRegressor(**self.gbm_params)
            self.gbm.fit(
                clean[GBM_FEATURES],
                clean_residual,
                categorical_feature=["equipment"],
            )
            final_residual = clean_residual - self.gbm.predict(clean[GBM_FEATURES])
        else:
            self.gbm = None
            final_residual = clean_residual

        # Duan smearing: E[rate] = exp(mu) * E[exp(eps)]. Converts the
        # median-unbiased exponentiated fit into a mean-unbiased one.
        self.smearing = float(np.mean(np.exp(final_residual)))
        self.residual_sd = float(np.std(final_residual))
        return self

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = _design(frame, LINEAR_FEATURES) @ self.coefficients
        if self.gbm is not None:
            prediction = prediction + self.gbm.predict(frame[GBM_FEATURES])
        return prediction

    def predict(self, frame: pd.DataFrame, smear: bool = True) -> np.ndarray:
        rate = np.exp(self.predict_log(frame))
        if smear:
            rate = rate * self.smearing
        return rate

    def coefficient_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": ["intercept"] + LINEAR_FEATURES,
             "coefficient": self.coefficients}
        )
