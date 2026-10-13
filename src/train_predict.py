"""Train the final model and write both required prediction files.

Run with:  python -m src.train_predict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DECEMBER_CSV,
    PREDICTIONS_CSV,
    REPORTS,
    TEMPLATE_CSV,
)
from .features import (
    GBM_FEATURES,
    city_coordinates,
    daily_market_table,
    engineer,
    load_december,
    load_frames,
)
from .model import HybridRateModel

DECEMBER_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]


def prepare_december(
    december: pd.DataFrame, coords: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    """Give the 31 fixed rows the columns the model expects.

    The file ships without coordinates or market signals, so both are attached
    from sources that need no label: city coordinates are constant per city,
    and the two market signals come from the pooled daily table. Only `date`
    varies across these rows, which is the point of the chart.
    """
    frame = december.copy()
    lookup = coords.set_index("city")
    for side in ("pickup", "delivery"):
        frame[f"{side}_lat"] = frame[side].map(lookup["lat"])
        frame[f"{side}_lon"] = frame[side].map(lookup["lon"])

    missing = frame[["pickup_lat", "delivery_lat"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"no coordinates for: {frame.loc[missing, 'pickup'].tolist()}")

    frame = frame.merge(daily, on="date", how="left")
    if frame[["market_index_daily", "quote_signal_daily"]].isna().any().any():
        raise ValueError("December dates missing from the daily market table")

    # The row-level signals are unobserved for these synthetic rows, so the
    # day's mean is the best available estimate of each.
    frame["market_index"] = frame["market_index_daily"]
    frame["quote_signal"] = frame["quote_signal_daily"]
    return frame


def main() -> None:
    train_raw, validation_raw = load_frames()
    december_raw = load_december()

    daily = daily_market_table(train_raw, validation_raw)
    coords = city_coordinates(train_raw, validation_raw)
    weight_fill = float(train_raw["weight"].abs().median())

    train = engineer(train_raw, daily, weight_fill=weight_fill)
    validation = engineer(validation_raw, daily, weight_fill=weight_fill)

    print(f"Training on {len(train):,} rows ({train.date.min().date()} -> "
          f"{train.date.max().date()})")
    model = HybridRateModel().fit(train, coords)
    print(f"  excluded {model.n_dropped} corrupted labels "
          f"({model.n_dropped / len(train) * 100:.2f}%)")
    print(f"  residual sd on log-rate: {model.residual_sd:.5f}")
    print(f"  smearing factor: {model.smearing:.5f}")
    print("\nBase model coefficients:")
    print(model.coefficient_table().round(6).to_string(index=False))

    importance = pd.DataFrame(
        {"feature": GBM_FEATURES, "gain": model.gbm.booster_.feature_importance("gain")}
    ).sort_values("gain", ascending=False)
    print("\nTop residual-stage features by gain:")
    print(importance.head(12).round(1).to_string(index=False))

    # ---- validation predictions ------------------------------------------
    validation_rate = model.predict(validation)
    template = pd.read_csv(TEMPLATE_CSV)
    predictions = template[["load_id"]].merge(
        pd.DataFrame(
            {"load_id": validation_raw["load_id"], "predicted_rate": validation_rate}
        ),
        on="load_id",
        how="left",
    )
    if predictions["predicted_rate"].isna().any():
        raise ValueError("some template load_ids were not predicted")
    predictions["predicted_rate"] = predictions["predicted_rate"].round(2)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nWrote {PREDICTIONS_CSV}  ({len(predictions):,} rows)")
    print(predictions["predicted_rate"].describe().round(2).to_string())

    # ---- fixed December chart inputs -------------------------------------
    december = prepare_december(december_raw, coords, daily)
    december_features = engineer(
        december.drop(columns=["market_index_daily", "quote_signal_daily",
                               "loads_posted"]),
        daily,
        weight_fill=weight_fill,
    )
    december_rate = model.predict(december_features)

    output = december_raw.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output["predicted_rate"] = np.round(december_rate, 2)
    output = output[DECEMBER_COLUMNS]
    output.to_csv(DECEMBER_CSV, index=False)
    print(f"\nWrote {DECEMBER_CSV}  ({len(output)} rows)")
    print(output[["date", "predicted_rate"]].to_string(index=False))

    REPORTS.mkdir(parents=True, exist_ok=True)
    model.coefficient_table().to_csv(REPORTS / "base_coefficients.csv", index=False)
    importance.to_csv(REPORTS / "gbm_feature_importance.csv", index=False)


if __name__ == "__main__":
    main()
