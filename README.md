# Freight Rate Prediction: Spotter ML Assessment

Predicts `posted_rate` for 12,000 freight loads covering 1 November to 31
December 2025, trained on 48,000 labelled loads from 1 January to 31 October
2025.

**Headline result:** MAE **$33.16** and MAPE **1.42%** on a two-month-ahead
holdout built to match the shape of the real task (train through August, predict
September and October together). The full write-up is in
[`reports/Freight_Rate_Report.pdf`](reports/Freight_Rate_Report.pdf).

## The problem in one paragraph

The validation window starts the day *after* the development data ends, so this
is forecasting rather than interpolation. Two properties of the data dominate
everything else: rates drift upward through the year by about 0.66% per 30 days,
and roughly 1.4% of the labels are corrupted by a random multiplier.
Extrapolating the first and removing the second is worth more than the choice of
learning algorithm. A plain LightGBM on raw labels scores MAE $87, against $33
for the model here.

## The model

A trimmed log-linear base, plus a LightGBM correction fitted to its residuals.

| Stage | Carries | Why it sits there |
|---|---|---|
| Trimmed OLS on `log(rate)` | distance elasticity, both market signals, equipment, linear time trend | these must extrapolate past the last training date, and trees cannot |
| LightGBM on the residual | distance and weight curvature, regional premium from coordinates, day of week | non-linear, but only inside ranges already observed |
| Duan smearing | retransformation bias | makes the exponentiated fit mean-unbiased |

Corrupted labels are excluded by a ratio gate: a load is kept when its posted
rate lands within 0.70 to 1.40 times the robust model's expectation. Those cut
points sit inside genuinely empty regions of the ratio histogram, since no
training load falls between 0.50 and 0.86, or between 1.21 and 2.19. The rule is
therefore read off the data rather than tuned.

Geography is expressed only through coordinates, never city names, because 8
cities appear solely in the validation file and account for 12.1% of the rows to
predict.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the submission

```bash
python -m src.train_predict
```

This writes `validation_predictions.csv` and fills the `predicted_rate` column
of `data/december_chart_inputs.csv`. Then run the supplied scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

which validates both files and writes `scorer_results/candidate_december.png`.

## Reproduce the analysis

| Command | What it produces |
|---|---|
| `python -m src.validate` | split study: random K-fold against forward chaining, per-fold detail, trend stability, data-quality ledger |
| `python -m src.ablation` | what each modelling decision is worth on the September and October holdout |
| `python -m src.unseen_cities` | leave-whole-cities-out test for the 8 unseen validation cities |
| `python -m src.figures` | report figures |
| `python -m src.build_report` | `reports/Freight_Rate_Report.pdf` (needs a LaTeX engine) |

Run them in that order. `src.figures` and `src.build_report` read the result
files that the first three write.

## Validation approach

Model selection used **forward chaining** only: expand the training window one
month at a time, and always score the month that follows. The same model scored
with a random 5-fold split reports MAE $24, against $53 forward-chained, so a
random split would have hidden more than half of the real error.

Reported error excludes holdout rows whose *own* label is corrupted. Those rows
are unpredictable by construction, so including them measures the quality of the
data rather than the quality of the model.

## Layout

```
src/
  config.py          paths and cleaning thresholds
  features.py        loading, cleaning, feature engineering
  model.py           HybridRateModel and the trimmed OLS fit
  validate.py        split study and data-quality ledger
  ablation.py        per-decision ablation
  unseen_cities.py   leave-cities-out generalisation test
  figures.py         report figures
  train_predict.py   final fit; writes both prediction files
  build_report.py    emits reports/report.tex and compiles it
data/                inputs, plus the completed December file
reports/             PDF report, LaTeX source, figures, saved result tables
score.py             supplied scorer, unmodified
validation_predictions.csv
```

## Notes on data handling

| Issue | Rows | Treatment |
|---|---|---|
| Corrupted labels (ratio outside the gate) | 677 (1.41%) | excluded from training |
| Sign-flipped weights, down to -47,500 | 292 | absolute value |
| Missing weight | 300 | median impute, plus a flag |
| Missing `market_index` | 374 | that day's market mean, plus a flag |
| Weight at the 47,500 cap | 1,204 | flagged, not altered |
| Distance at the 70-mile floor | 48 | flagged; a real short-haul billing minimum |

`market_index` and `quote_signal` behave like noisy readings of a daily market
level, with a within-day standard deviation of 0.025 against 0.167 across days,
so both the row value and the daily mean are used. The daily means pool the
market columns of every row, labelled or not. These are exogenous inputs that
ship with each load, and no target value is involved. This is also what makes
the fixed December chart possible, since those 31 rows carry no market columns
of their own.
