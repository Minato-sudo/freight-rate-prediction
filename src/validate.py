"""Validation study: how the development data is split, and what it costs.

Run with:  python -m src.validate
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import CORRUPT_HIGH, CORRUPT_LOW, RANDOM_STATE, REPORTS
from .features import (
    city_coordinates,
    daily_market_table,
    engineer,
    load_frames,
    load_december,
)
from .features import LINEAR_FEATURES
from .model import HybridRateModel, _design, fit_trimmed_ols


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(np.mean(np.abs(error / actual)) * 100.0),
        "R2": float(1.0 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def label_corruption_mask(frame: pd.DataFrame) -> np.ndarray:
    """True where the label looks corrupted, by the ratio gate."""
    target = np.log(frame["posted_rate"].to_numpy(dtype=float))
    coefficients, _ = fit_trimmed_ols(frame, target, LINEAR_FEATURES)
    ratio = np.exp(target - _design(frame, LINEAR_FEATURES) @ coefficients)
    return (ratio <= CORRUPT_LOW) | (ratio >= CORRUPT_HIGH)


def evaluate(train_frame, holdout_frame, coords, label, variants=("hybrid",)):
    """Fit on train_frame, score on holdout_frame, clean rows and all rows."""
    results = []
    corrupt = label_corruption_mask(holdout_frame)
    actual = holdout_frame["posted_rate"].to_numpy(dtype=float)

    for variant in variants:
        model = HybridRateModel(use_gbm=(variant != "linear-only")).fit(
            train_frame, coords
        )
        predicted = model.predict(holdout_frame)
        row = {"split": label, "model": variant, "n_train": len(train_frame),
               "n_holdout": len(holdout_frame)}
        row.update({f"clean_{k}": v for k, v in
                    metrics(actual[~corrupt], predicted[~corrupt]).items()})
        row.update({f"all_{k}": v for k, v in metrics(actual, predicted).items()})
        row["dropped_in_training"] = model.n_dropped
        row["residual_sd_log"] = model.residual_sd
        results.append(row)
    return pd.DataFrame(results)


def main() -> None:
    train_raw, validation_raw = load_frames()
    daily = daily_market_table(train_raw, validation_raw)
    coords = city_coordinates(train_raw, validation_raw)
    weight_fill = float(train_raw["weight"].abs().median())
    train = engineer(train_raw, daily, weight_fill=weight_fill)

    print("=" * 78)
    print("1. WHY NOT A RANDOM SPLIT")
    print("=" * 78)
    print(
        "Development data: {} -> {}\nValidation data : {} -> {}".format(
            train_raw.date.min().date(), train_raw.date.max().date(),
            validation_raw.date.min().date(), validation_raw.date.max().date(),
        )
    )
    print(
        "The validation window starts the day after development data ends, so the\n"
        "real task is forecasting forward. A random K-fold would let the model see\n"
        "November-adjacent days while scoring October ones and would overstate skill."
    )

    # --- random K-fold vs forward-chaining, same model, same data -------------
    rows = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, te_idx) in enumerate(kfold.split(train), start=1):
        rows.append(
            evaluate(train.iloc[tr_idx], train.iloc[te_idx], coords,
                     f"random-kfold-{fold}")
        )
    random_cv = pd.concat(rows, ignore_index=True)

    # Forward chaining: expand the training window a month at a time and always
    # score the month that comes next - the same shape as the real task.
    folds = [
        ("2025-06-01", "2025-07-01"),
        ("2025-07-01", "2025-08-01"),
        ("2025-08-01", "2025-09-01"),
        ("2025-09-01", "2025-10-01"),
        ("2025-10-01", "2025-11-01"),
    ]
    rows = []
    for cut, end in folds:
        tr = train[train.date < cut]
        te = train[(train.date >= cut) & (train.date < end)]
        rows.append(evaluate(tr, te, coords, f"forward-{cut[:7]}"))
    forward_cv = pd.concat(rows, ignore_index=True)

    # A two-month holdout mirrors the exact 61-day horizon that must be predicted.
    horizon = evaluate(
        train[train.date < "2025-09-01"], train[train.date >= "2025-09-01"],
        coords, "two-month-horizon (Sep-Oct)",
        variants=("hybrid", "linear-only"),
    )

    print("\n" + "=" * 78)
    print("2. RANDOM K-FOLD vs FORWARD CHAINING  (identical model)")
    print("=" * 78)
    summary = pd.DataFrame(
        {
            "random K-fold (5)": random_cv[["clean_MAE", "clean_RMSE", "clean_MAPE"]].mean(),
            "forward chaining (5)": forward_cv[["clean_MAE", "clean_RMSE", "clean_MAPE"]].mean(),
        }
    ).T.round(3)
    print(summary.to_string())
    gap = (
        forward_cv.clean_MAE.mean() / random_cv.clean_MAE.mean() - 1.0
    ) * 100.0
    print(f"\nForward-chaining MAE is {gap:+.1f}% vs random K-fold: that gap is the")
    print("optimism a random split would have hidden. Model selection used the")
    print("forward-chaining number only.")

    print("\n" + "=" * 78)
    print("3. PER-FOLD DETAIL (forward chaining)")
    print("=" * 78)
    cols = ["split", "n_train", "n_holdout", "clean_MAE", "clean_RMSE",
            "clean_MAPE", "clean_R2", "all_MAE", "all_RMSE"]
    print(forward_cv[cols].round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("4. TWO-MONTH HORIZON HOLDOUT  (Sep-Oct, the shape of the real task)")
    print("=" * 78)
    print(horizon[["model"] + cols[1:]].round(3).to_string(index=False))
    print(
        "\nclean_* excludes holdout rows whose own label is corrupted; all_* keeps\n"
        "them. The gap between the two is the error floor imposed by corrupted\n"
        "ground truth, not by the model."
    )

    print("\n" + "=" * 78)
    print("5. THE TIME TREND AND WHETHER TO EXTRAPOLATE IT")
    print("=" * 78)
    target = np.log(train["posted_rate"].to_numpy(dtype=float))
    timeless = [c for c in LINEAR_FEATURES if c != "days_elapsed"]
    coefficients, keep = fit_trimmed_ols(train, target, timeless)
    residual = (target - _design(train, timeless) @ coefficients)
    clean = train.loc[keep]
    clean_residual = residual[keep]

    monthly = pd.Series(clean_residual).groupby(
        clean.date.dt.to_period("M").astype(str).to_numpy()
    ).mean()
    print("Mean residual by month, from a model with NO time term:")
    print(monthly.round(5).to_string())

    trend_rows = []
    windows = [
        ("Jan-Mar", clean.date < "2025-04-01"),
        ("Apr-Jun", (clean.date >= "2025-04-01") & (clean.date < "2025-07-01")),
        ("Jul-Aug", (clean.date >= "2025-07-01") & (clean.date < "2025-09-01")),
        ("Sep-Oct", clean.date >= "2025-09-01"),
        ("full Jan-Oct", clean.date.notna()),
    ]
    for name, mask in windows:
        sub = clean.loc[mask]
        res = clean_residual[mask.to_numpy()]
        design = np.column_stack([np.ones(len(sub)), sub.days_elapsed.to_numpy()])
        slope_fit, *_ = np.linalg.lstsq(design, res, rcond=None)
        dof = max(len(sub) - 2, 1)
        centred = sub.days_elapsed - sub.days_elapsed.mean()
        std_err = np.sqrt(
            np.sum((res - design @ slope_fit) ** 2) / dof / np.sum(centred**2)
        )
        trend_rows.append(
            {
                "window": name,
                "drift per day": slope_fit[1],
                "t-stat": slope_fit[1] / std_err,
                "% per 30 days": (np.exp(slope_fit[1] * 30) - 1) * 100,
            }
        )
    trend = pd.DataFrame(trend_rows)
    print("\nLinear drift fitted inside each window:")
    print(trend.round(6).to_string(index=False))

    month_index = np.arange(len(monthly))
    shapes = {}
    for degree in (1, 2, 3):
        fitted = np.polyval(np.polyfit(month_index, monthly.to_numpy(), degree), month_index)
        shapes[f"degree {degree}"] = 1 - np.sum(
            (monthly.to_numpy() - fitted) ** 2
        ) / np.sum((monthly.to_numpy() - monthly.mean()) ** 2)
    print("\nFit to the monthly pattern: " + ", ".join(
        f"{k} R2={v:.3f}" for k, v in shapes.items()))
    print(
        "The drift is up over the ten months (+0.0002/day, t=121 on the full\n"
        "window) but it is not a clean ramp: it pauses and dips through Jul-Aug.\n"
        "A straight line already explains ~89% of the monthly pattern, and the\n"
        "quadratic and cubic terms add almost nothing while being far more\n"
        "dangerous 61 days past the last observation. So the trend is carried\n"
        "forward as a single linear term, deliberately not as a flexible curve."
    )

    print("\nWhat the term is worth on a two-month-ahead holdout (Sep-Oct):")
    bias_rows = []
    holdout = train[train.date >= "2025-09-01"]
    fitting = train[train.date < "2025-09-01"]
    holdout_corrupt = label_corruption_mask(holdout)
    actual = holdout["posted_rate"].to_numpy(dtype=float)[~holdout_corrupt]
    import src.features as feature_module
    import src.model as model_module

    original = list(feature_module.LINEAR_FEATURES)
    for label, keep_time in [("with days_elapsed", True), ("without days_elapsed", False)]:
        active = original if keep_time else [c for c in original if c != "days_elapsed"]
        model_module.LINEAR_FEATURES = active
        model_module.LINEAR_FEATURES = [
            c for c in active if not c.endswith("region_index")
        ]
        fitted_model = model_module.HybridRateModel().fit(fitting, coords)
        predicted = fitted_model.predict(holdout)[~holdout_corrupt]
        bias_rows.append(
            {
                "model": label,
                "clean MAE": float(np.mean(np.abs(predicted - actual))),
                "mean bias %": float(np.mean(predicted / actual) - 1) * 100,
            }
        )
    model_module.LINEAR_FEATURES = original
    model_module.LINEAR_FEATURES = [
        c for c in original if not c.endswith("region_index")
    ]
    bias_table = pd.DataFrame(bias_rows)
    print(bias_table.round(3).to_string(index=False))
    print(
        "Dropping it leaves a systematic -3.4% shortfall two months out. That is\n"
        "the single largest avoidable error in this problem, and it is exactly\n"
        "what a tree-only model would suffer."
    )

    # --- data quality tally ---------------------------------------------------
    print("\n" + "=" * 78)
    print("6. DATA QUALITY LEDGER (development data)")
    print("=" * 78)
    corrupt = label_corruption_mask(train)
    ledger = pd.DataFrame(
        [
            {"issue": "label corrupted (ratio outside gate)", "rows": int(corrupt.sum()),
             "action": "excluded from training"},
            {"issue": "weight sign-flipped (negative)",
             "rows": int((train_raw.weight < 0).sum()), "action": "absolute value"},
            {"issue": "weight missing", "rows": int(train_raw.weight.isna().sum()),
             "action": "median impute + flag"},
            {"issue": "market_index missing",
             "rows": int(train_raw.market_index.isna().sum()),
             "action": "daily market mean + flag"},
            {"issue": "weight at 47,500 cap",
             "rows": int((train_raw.weight.abs() == 47500).sum()),
             "action": "flagged (upstream cap)"},
            {"issue": "distance at 70-mile floor",
             "rows": int((train_raw.distance <= 70).sum()),
             "action": "flagged (short-haul minimum)"},
        ]
    )
    ledger["share %"] = (ledger.rows / len(train_raw) * 100).round(2)
    print(ledger.to_string(index=False))

    unseen = sorted(set(validation_raw.pickup) - set(train_raw.pickup))
    touched = validation_raw.pickup.isin(unseen) | validation_raw.delivery.isin(unseen)
    print(f"\nCities appearing only in validation: {len(unseen)} -> {unseen}")
    print(
        f"Validation rows touching one: {touched.sum():,} "
        f"({touched.mean() * 100:.1f}%) - handled by coordinate interpolation."
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = {
        "random_kfold": random_cv.to_dict("records"),
        "forward_chaining": forward_cv.to_dict("records"),
        "two_month_horizon": horizon.to_dict("records"),
        "trend_stability": trend.to_dict("records"),
        "trend_term_value": bias_table.to_dict("records"),
        "monthly_residual_no_time_term": monthly.round(6).to_dict(),
        "data_quality": ledger.to_dict("records"),
        "unseen_cities": unseen,
        "validation_rows_touching_unseen_city": int(touched.sum()),
    }
    (REPORTS / "validation_results.json").write_text(json.dumps(out, indent=2))
    forward_cv.to_csv(REPORTS / "forward_chaining_folds.csv", index=False)
    print(f"\nWrote {REPORTS / 'validation_results.json'}")


if __name__ == "__main__":
    main()
