"""Paths and project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TRAIN_CSV = DATA / "train_test.csv"
VALIDATION_CSV = DATA / "validation.csv"
TEMPLATE_CSV = DATA / "validation_predictions_template.csv"
DECEMBER_CSV = DATA / "december_chart_inputs.csv"

PREDICTIONS_CSV = ROOT / "validation_predictions.csv"
REPORTS = ROOT / "reports"

TARGET = "posted_rate"
# Anchor for the linear time trend; keeps the fitted coefficient interpretable
# as "log-rate drift per day" rather than depending on an arbitrary epoch.
TIME_ORIGIN = "2025-01-01"

# Physical bounds observed in the data and used for cleaning.
WEIGHT_CAP = 47_500.0        # weight is hard-capped at this value upstream
MIN_BILLED_DISTANCE = 70.0   # short hauls are billed at a 70-mile floor

# Label-corruption gate. Clean loads sit inside [0.86, 1.20] times the
# robust model's expectation; corrupted labels sit outside [0.5, 2.2].
# The thresholds below are placed inside those empty gaps.
CORRUPT_LOW = 0.70
CORRUPT_HIGH = 1.40

RANDOM_STATE = 42
