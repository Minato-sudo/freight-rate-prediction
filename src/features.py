"""Data loading, cleaning and feature engineering for freight rate prediction.

Design notes
------------
* ``market_index`` and ``quote_signal`` behave like noisy observations of a
  daily market level (within-day sd 0.025 vs across-day sd 0.167 for
  ``market_index``). Averaging them per calendar day recovers a much cleaner
  signal, so both the row-level value and the daily mean are exposed.
* Pickup / delivery identity carries a real regional premium, but 8 cities
  appear only in the validation file. Everything geographic is therefore
  expressed through coordinates, never through the city string, so unseen
  cities are interpolated rather than dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DECEMBER_CSV,
    MIN_BILLED_DISTANCE,
    TIME_ORIGIN,
    TRAIN_CSV,
    VALIDATION_CSV,
    WEIGHT_CAP,
)

EARTH_RADIUS_MILES = 3958.7613


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the labelled development data and the unlabelled validation data."""
    train = pd.read_csv(TRAIN_CSV, parse_dates=["date"])
    validation = pd.read_csv(VALIDATION_CSV, parse_dates=["date"])
    return train, validation


def load_december() -> pd.DataFrame:
    return pd.read_csv(DECEMBER_CSV, parse_dates=["date"])


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles."""
    rad = np.pi / 180.0
    dphi = (lat2 - lat1) * rad
    dlam = (lon2 - lon1) * rad
    inner = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlam / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def daily_market_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Average the two market signals per calendar day.

    The market columns are exogenous inputs that ship with every row, labelled
    or not, so pooling the validation rows here uses no target information. It
    is also what makes the fixed December chart possible: those 31 rows carry
    no market columns of their own.
    """
    parts = [f[["date", "market_index", "quote_signal"]] for f in frames]
    pooled = pd.concat(parts, ignore_index=True)
    daily = pooled.groupby("date", as_index=False).agg(
        market_index_daily=("market_index", "mean"),
        quote_signal_daily=("quote_signal", "mean"),
        loads_posted=("market_index", "size"),
    )
    return daily


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One row per city with its (constant) coordinates."""
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            parts.append(
                frame[[side, f"{side}_lat", f"{side}_lon"]].rename(
                    columns={side: "city", f"{side}_lat": "lat", f"{side}_lon": "lon"}
                )
            )
    return (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("city")
        .reset_index(drop=True)
    )


def clean(frame: pd.DataFrame, weight_fill: float | None = None) -> pd.DataFrame:
    """Repair the input columns. Returns a copy; never touches the target.

    Fixes applied
    -------------
    weight  : 292 training rows carry a sign-flipped value (down to -47,500);
              the magnitude is plausible, so the sign is corrected rather than
              the row dropped. Missing weights are median-imputed and flagged.
    market_index : missing values fall back to that day's market mean.
    """
    out = frame.copy()

    out["weight_sign_flipped"] = (out["weight"] < 0).fillna(False).astype(int)
    out["weight"] = out["weight"].abs()

    out["weight_missing"] = out["weight"].isna().astype(int)
    if weight_fill is None:
        weight_fill = float(out["weight"].median())
    out["weight"] = out["weight"].fillna(weight_fill)
    out["weight_at_cap"] = (out["weight"] >= WEIGHT_CAP).astype(int)

    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["market_index"] = out["market_index"].fillna(out["market_index_daily"])
    # Belt and braces: a day absent from the daily table would leave a gap.
    out["market_index"] = out["market_index"].fillna(out["market_index"].median())

    out["quote_signal_missing"] = out["quote_signal"].isna().astype(int)
    out["quote_signal"] = out["quote_signal"].fillna(out["quote_signal_daily"])
    return out


def engineer(
    frame: pd.DataFrame,
    daily: pd.DataFrame,
    weight_fill: float | None = None,
) -> pd.DataFrame:
    """Attach market aggregates, then derive the modelling features."""
    out = frame.merge(daily, on="date", how="left")
    out = clean(out, weight_fill=weight_fill)

    out["great_circle"] = haversine_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    # Billed miles run ~1.18x great-circle (road circuity). The ratio is stable,
    # which is what rules out corrupted distances.
    out["circuity"] = out["distance"] / out["great_circle"].clip(lower=1.0)
    out["at_distance_floor"] = (out["distance"] <= MIN_BILLED_DISTANCE).astype(int)

    out["log_distance"] = np.log(out["distance"])
    out["log_weight"] = np.log(out["weight"].clip(lower=1.0))
    out["log_market_index"] = np.log(out["market_index"])
    out["log_quote_signal"] = np.log(out["quote_signal"])
    out["log_market_index_daily"] = np.log(out["market_index_daily"])
    out["log_quote_signal_daily"] = np.log(out["quote_signal_daily"])

    out["weight_per_mile"] = out["weight"] / out["distance"]

    out["days_elapsed"] = (out["date"] - pd.Timestamp(TIME_ORIGIN)).dt.days
    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["day_of_month"] = out["date"].dt.day

    out["mid_lat"] = (out["pickup_lat"] + out["delivery_lat"]) / 2.0
    out["mid_lon"] = (out["pickup_lon"] + out["delivery_lon"]) / 2.0
    out["delta_lat"] = out["delivery_lat"] - out["pickup_lat"]
    out["delta_lon"] = out["delivery_lon"] - out["pickup_lon"]
    out["heads_north"] = (out["delta_lat"] > 0).astype(int)
    out["heads_east"] = (out["delta_lon"] > 0).astype(int)

    out["equipment"] = out["equipment"].astype("category")
    out["is_flatbed"] = (out["equipment"] == "Flatbed").astype(float)
    out["is_reefer"] = (out["equipment"] == "Reefer").astype(float)
    return out


# Columns fed to the linear base. Every one of these extrapolates sensibly
# past the end of the training window, which is what the Nov/Dec task needs.
LINEAR_FEATURES = [
    "log_distance",
    "log_weight",
    "log_market_index",
    "log_quote_signal",
    "log_market_index_daily",
    "log_quote_signal_daily",
    "days_elapsed",
    "is_flatbed",
    "is_reefer",
]

# Columns fed to the gradient-boosted residual stage. Deliberately excludes
# days_elapsed: trees cannot extrapolate a trend, and letting them see the
# clock only lets them fit in-sample wiggles that do not carry to December.
GBM_FEATURES = [
    "log_distance",
    "distance",
    "log_weight",
    "weight",
    "weight_per_mile",
    "log_market_index",
    "log_quote_signal",
    "log_market_index_daily",
    "log_quote_signal_daily",
    "great_circle",
    "circuity",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "mid_lat",
    "mid_lon",
    "delta_lat",
    "delta_lon",
    "heads_north",
    "heads_east",
    "day_of_week",
    "is_weekend",
    "at_distance_floor",
    "weight_at_cap",
    "weight_missing",
    "market_index_missing",
    "equipment",
]
