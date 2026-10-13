"""Ablation: what each modelling decision is worth on a forward holdout.

Run with:  python -m src.ablation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import REPORTS
from .features import (
    GBM_FEATURES,
    city_coordinates,
    daily_market_table,
    engineer,
    load_frames,
)
from . import features as feature_module
from . import model as model_module
from .model import HybridRateModel
from .validate import label_corruption_mask, metrics

CUTOFF = "2025-09-01"


def run() -> pd.DataFrame:
    train_raw, validation_raw = load_frames()
    daily = daily_market_table(train_raw, validation_raw)
    coords = city_coordinates(train_raw, validation_raw)
    weight_fill = float(train_raw["weight"].abs().median())
    train = engineer(train_raw, daily, weight_fill=weight_fill)

    fitting = train[train.date < CUTOFF]
    holdout = train[train.date >= CUTOFF]
    corrupt = label_corruption_mask(holdout)
    actual = holdout["posted_rate"].to_numpy(dtype=float)[~corrupt]

    baseline_linear = list(feature_module.LINEAR_FEATURES)
    rows = []

    def record(name, predicted, note):
        result = metrics(actual, predicted[~corrupt])
        rows.append({"variant": name, **{k: round(v, 3) for k, v in result.items()},
                     "note": note})
        print(f"  {name:34s} MAE={result['MAE']:8.2f}  MAPE={result['MAPE']:6.3f}%")

    def with_linear(active):
        model_module.LINEAR_FEATURES = active

    print(f"Forward holdout: train < {CUTOFF} ({len(fitting):,}), "
          f"holdout >= {CUTOFF} ({len(holdout):,}; {(~corrupt).sum():,} clean)\n")

    # 1. full model
    with_linear(baseline_linear)
    full = HybridRateModel().fit(fitting, coords)
    record("full hybrid", full.predict(holdout), "shipped model")

    # 2. linear base only
    with_linear(baseline_linear)
    record("  - no GBM residual stage",
           HybridRateModel(use_gbm=False).fit(fitting, coords).predict(holdout),
           "loses distance/weight curvature")

    # 3. no time trend
    with_linear([c for c in baseline_linear if c != "days_elapsed"])
    record("  - no time trend",
           HybridRateModel().fit(fitting, coords).predict(holdout),
           "cannot extrapolate forward")

    # 4. no geography at all
    with_linear(baseline_linear)
    original_gbm = list(feature_module.GBM_FEATURES)
    geo = {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
           "mid_lat", "mid_lon", "delta_lat", "delta_lon",
           "heads_north", "heads_east", "great_circle", "circuity"}
    model_module.GBM_FEATURES = [c for c in original_gbm if c not in geo]
    feature_module.GBM_FEATURES = model_module.GBM_FEATURES
    record("  - no geography",
           HybridRateModel().fit(fitting, coords).predict(holdout),
           "drops the regional premium")
    model_module.GBM_FEATURES = original_gbm
    feature_module.GBM_FEATURES = original_gbm

    # 5. no daily market aggregates (row-level signals only)
    with_linear([c for c in baseline_linear if not c.endswith("_daily")])
    original_gbm = list(feature_module.GBM_FEATURES)
    model_module.GBM_FEATURES = [c for c in original_gbm if not c.endswith("_daily")]
    feature_module.GBM_FEATURES = model_module.GBM_FEATURES
    record("  - no daily market aggregates",
           HybridRateModel().fit(fitting, coords).predict(holdout),
           "row-level market signals only")
    model_module.GBM_FEATURES = original_gbm
    feature_module.GBM_FEATURES = original_gbm

    # 6. no label cleaning: train on everything
    with_linear(baseline_linear)
    import src.config as config_module
    low, high = config_module.CORRUPT_LOW, config_module.CORRUPT_HIGH
    model_module.CORRUPT_LOW, model_module.CORRUPT_HIGH = 0.0, 1e9
    record("  - no corrupt-label removal",
           HybridRateModel().fit(fitting, coords).predict(holdout),
           "1.4% bad labels left in")
    model_module.CORRUPT_LOW, model_module.CORRUPT_HIGH = low, high

    # 7. plain LightGBM on log-rate, everything, no structure
    with_linear(baseline_linear)
    enriched_fit, enriched_hold = fitting, holdout
    plain_features = list(GBM_FEATURES)
    plain = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        verbose=-1, random_state=42,
    )
    plain.fit(
        enriched_fit[plain_features + ["days_elapsed"]],
        np.log(enriched_fit["posted_rate"].to_numpy(dtype=float)),
        categorical_feature=["equipment"],
    )
    record("plain LightGBM (no linear base)",
           np.exp(plain.predict(enriched_hold[plain_features + ["days_elapsed"]])),
           "trees alone, raw labels")

    with_linear(baseline_linear)
    table = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "ablation.csv", index=False)
    print(f"\nWrote {REPORTS / 'ablation.csv'}")
    return table


if __name__ == "__main__":
    print(run().to_string(index=False))
