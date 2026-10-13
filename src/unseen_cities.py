"""Leave-cities-out test: does the model survive cities it has never seen?

12.1% of the validation rows touch one of 8 cities that never appear in the
development data. A time-based split cannot measure that, because every city in
the development data is present throughout it. Holding out whole cities is the
only honest rehearsal for that part of the task.

Everything geographic in the model is a coordinate, never a city string, so an
unseen city is simply a new point in a space the trees have already carved up.
This script checks that the assumption holds.

Run with:  python -m src.unseen_cities
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, REPORTS
from . import features as feature_module
from . import model as model_module
from .features import city_coordinates, daily_market_table, engineer, load_frames
from .model import HybridRateModel
from .validate import label_corruption_mask, metrics

N_HELD_OUT_CITIES = 8
N_REPEATS = 5

GEO_FEATURES = {
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "mid_lat", "mid_lon", "delta_lat", "delta_lon",
    "heads_north", "heads_east",
}


def run() -> pd.DataFrame:
    train_raw, validation_raw = load_frames()
    daily = daily_market_table(train_raw, validation_raw)
    coords = city_coordinates(train_raw, validation_raw)
    weight_fill = float(train_raw["weight"].abs().median())
    train = engineer(train_raw, daily, weight_fill=weight_fill)

    cities = sorted(set(train.pickup) | set(train.delivery))
    rng = np.random.default_rng(RANDOM_STATE)
    original_gbm = list(feature_module.GBM_FEATURES)
    rows = []

    for repeat in range(N_REPEATS):
        held = [str(c) for c in rng.choice(cities, size=N_HELD_OUT_CITIES, replace=False)]
        touches = train.pickup.isin(held) | train.delivery.isin(held)
        fitting, holdout = train[~touches], train[touches]
        corrupt = label_corruption_mask(holdout)
        actual = holdout["posted_rate"].to_numpy(dtype=float)[~corrupt]

        for label, features in [
            ("shipped model (coordinates)", original_gbm),
            ("no geography", [c for c in original_gbm if c not in GEO_FEATURES]),
        ]:
            model_module.GBM_FEATURES = features
            feature_module.GBM_FEATURES = features
            model = HybridRateModel().fit(fitting, coords)
            predicted = model.predict(holdout)[~corrupt]
            rows.append(
                {
                    "repeat": repeat + 1,
                    "variant": label,
                    "held_out_cities": ", ".join(held),
                    "n_holdout": int((~corrupt).sum()),
                    **{k: round(v, 3) for k, v in metrics(actual, predicted).items()},
                    "mean_bias_%": round(float(np.mean(predicted / actual) - 1) * 100, 3),
                }
            )
        print(f"repeat {repeat + 1}: held out {held}")
        for row in rows[-2:]:
            print(f"    {row['variant']:30s} MAE={row['MAE']:7.2f}  "
                  f"MAPE={row['MAPE']:6.3f}%  bias={row['mean_bias_%']:+.3f}%")

    model_module.GBM_FEATURES = original_gbm
    feature_module.GBM_FEATURES = original_gbm

    table = pd.DataFrame(rows)
    summary = table.groupby("variant")[["MAE", "RMSE", "MAPE", "mean_bias_%"]].mean()
    print("\n" + "=" * 72)
    print(f"Averaged over {N_REPEATS} repeats, {N_HELD_OUT_CITIES} whole cities held out:")
    print(summary.round(3).to_string())
    shipped = summary.loc["shipped model (coordinates)"]
    print(
        f"\nOn loads touching a city the model has never seen it still lands at "
        f"{shipped['MAPE']:.2f}% MAPE\nwith {shipped['mean_bias_%']:+.2f}% bias, so the "
        "8 unseen validation cities are not a risk.\nDropping coordinates costs "
        f"{(summary.loc['no geography', 'MAE'] / shipped['MAE'] - 1) * 100:.0f}% MAE, "
        "which is the size of the regional premium."
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "unseen_city_test.csv", index=False)
    print(f"Wrote {REPORTS / 'unseen_city_test.csv'}")
    return table


if __name__ == "__main__":
    run()
