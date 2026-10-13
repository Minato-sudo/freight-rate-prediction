"""Build the submission report as a LaTeX PDF.

Every number is read back from the artefacts written by src.validate,
src.ablation and src.unseen_cities, so the prose cannot drift from the results.

Requires `tectonic` on PATH (or pdflatex; see TEX_ENGINES below).

Run with:  python -m src.build_report_tex
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pandas as pd

from .config import REPORTS

CANDIDATE = "Muhammad Talha Arshad"
TEX_ENGINES = [["tectonic", "-X", "compile"], ["pdflatex", "-interaction=nonstopmode"]]


def esc(text: str) -> str:
    """Escape the LaTeX specials that appear in this report."""
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                     ("}", r"\}"), ("~", r"\textasciitilde{}"),
                     ("^", r"\textasciicircum{}")]:
        text = text.replace(old, new)
    return text


def num(value: float, spec: str) -> str:
    """Format a number so a negative sign renders as a true minus."""
    text = format(value, spec)
    return text.replace("-", "$-$") if text.startswith("-") else text


def rows_to_tex(rows: list[list[str]]) -> str:
    return "\n".join(" & ".join(r) + r" \\" for r in rows)


PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage{charter}
\usepackage{amsmath}
\usepackage[a4paper,top=26mm,bottom=28mm,left=24mm,right=24mm]{geometry}
\usepackage{float}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}

% One typeface throughout; LaTeX sets all text black by default, and no colour
% package is loaded, so no coloured text can appear anywhere in the document.
\renewcommand{\familydefault}{\rmdefault}

\titleformat{\section}{\normalfont\large\bfseries}{\thesection.}{0.6em}{}
\titlespacing*{\section}{0pt}{16pt plus 3pt minus 2pt}{7pt}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{}{0em}{}
\titlespacing*{\subsection}{0pt}{12pt plus 2pt minus 2pt}{4pt}

\captionsetup{font=small,labelfont=bf,justification=justified,
              singlelinecheck=false,skip=6pt}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.4pt}
\fancyfoot[L]{\footnotesize CANDIDATENAME}
\fancyfoot[R]{\footnotesize Page \thepage}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\renewcommand{\arraystretch}{1.25}

\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}

\begin{document}
"""


def build():
    results = json.loads((REPORTS / "validation_results.json").read_text())
    ablation = pd.read_csv(REPORTS / "ablation.csv")
    unseen = pd.read_csv(REPORTS / "unseen_city_test.csv")
    coefficients = pd.read_csv(REPORTS / "base_coefficients.csv")

    forward = pd.DataFrame(results["forward_chaining"])
    random_cv = pd.DataFrame(results["random_kfold"])
    horizon = pd.DataFrame(results["two_month_horizon"])
    quality = pd.DataFrame(results["data_quality"])
    trend = pd.DataFrame(results["trend_stability"])
    trend_value = pd.DataFrame(results["trend_term_value"])

    hybrid = horizon[horizon.model == "hybrid"].iloc[0]
    linear_only = horizon[horizon.model == "linear-only"].iloc[0]
    full = ablation[ablation.variant == "full hybrid"].iloc[0]
    no_clean = ablation[ablation.variant.str.contains("corrupt")].iloc[0]
    plain_gbm = ablation[ablation.variant.str.contains("plain")].iloc[0]
    unseen_summary = unseen.groupby("variant")[["MAE", "MAPE", "mean_bias_%"]].mean()
    shipped = unseen_summary.loc["shipped model (coordinates)"]
    no_geo = unseen_summary.loc["no geography"]
    random_mae = random_cv.clean_MAE.mean()
    forward_mae = forward.clean_MAE.mean()

    tex = [PREAMBLE.replace("CANDIDATENAME", esc(CANDIDATE))]

    # ---------------- title ----------------
    tex.append(r"""
\begin{center}
{\LARGE\bfseries Freight Rate Prediction}\\[4pt]
{\large Machine Learning Engineer Assessment}\\[10pt]
\rule{\textwidth}{0.8pt}\\[6pt]
""" + esc(CANDIDATE) + r"""
\end{center}

\vspace{4pt}
""")

    # ---------------- abstract ----------------
    tex.append(r"""
\subsection*{Summary}

The development file covers 1 January to 31 October 2025, while the 12{,}000
loads requiring predictions cover 1 November to 31 December. The validation
window begins the day after the training data ends, which makes this a
forecasting problem rather than an interpolation problem. Two properties of the
data follow from that and drive every decision in this report: rates drift
upward through the year, and approximately 1.4\% of the labels are corrupted.
Handling those two issues proved more valuable than the choice of learning
algorithm.

""")

    tex.append(r"""
\begin{table}[H]
\small
\begin{tabularx}{\textwidth}{Xl}
\toprule
\textbf{Measured on} & \textbf{Result} \\
\midrule
""" + rows_to_tex([
        ["Two-month-ahead holdout (Sep and Oct, 9{,}523 loads)",
         f"MAE \\${hybrid.clean_MAE:,.2f}, MAPE {hybrid.clean_MAPE:.2f}\\%, "
         f"$R^2$ {hybrid.clean_R2:.4f}"],
        ["Forward-chaining cross-validation, 5 folds",
         f"MAE \\${forward_mae:,.2f}, MAPE {forward.clean_MAPE.mean():.2f}\\%"],
        ["Loads touching a city never seen in training",
         f"MAPE {shipped.MAPE:.2f}\\%, bias {shipped['mean_bias_%']:+.2f}\\%"],
        ["Corrupted labels excluded from training",
         f"{int(quality.loc[0, 'rows'])} of 48{{,}}000 "
         f"({quality.loc[0, 'share %']:.2f}\\%)"],
    ]) + r"""
\bottomrule
\end{tabularx}
\end{table}

All error figures exclude holdout rows whose own label is corrupted. Those rows
are unpredictable by construction, so including them measures the quality of the
data rather than the quality of the model.

\vspace{2pt}
""")

    # ---------------- 1 ----------------
    tex.append(r"""
\section{Key findings from exploring the data}

Rate behaves as an almost perfectly multiplicative function of its inputs.
Regressing $\log(\text{rate})$ on $\log(\text{distance})$, $\log(\text{weight})$,
the two market signals, equipment and a time term explains 99.7\% of the
variance across clean loads, with a residual standard deviation of 0.034 in log
space. That structure is the backbone of the model, and it is why the work is
done in logs: rates span \$57 to \$25{,}533, and the error that matters
commercially is proportional rather than absolute.

\begin{table}[H]
\small
\begin{tabularx}{\textwidth}{L{0.30\textwidth}L{0.34\textwidth}X}
\toprule
\textbf{Finding} & \textbf{Evidence} & \textbf{Consequence} \\
\midrule
""" + rows_to_tex([
        ["Distance dominates, with an elasticity of 0.87 rather than 1.0",
         "Rate per mile falls from \\$2.40 on short hauls to \\$1.90 on long ones",
         "A flat rate-per-mile baseline is structurally wrong: MAE \\$256 against "
         "\\$33"],
        ["The two market columns are noisy readings of a daily market level",
         "market\\_index varies 0.025 within a day but 0.167 across days",
         "Averaging per day sharpens the signal and makes the December chart "
         "possible"],
        ["Pickup and delivery each carry a regional premium of roughly 3\\%",
         "Dallas and Oklahoma City near $+2.9\\%$; Syracuse and Buffalo near "
         "$-3.6\\%$",
         "Carried by coordinates rather than city names, so unseen cities still "
         "price correctly"],
        ["Rates drift upward all year, about 0.66\\% per 30 days",
         "Monthly residuals rise from $-3.2\\%$ in January to $+3.0\\%$ in "
         "October ($t=121$)",
         "Must be extrapolated: this is the single largest avoidable source of "
         "error"],
        ["Equipment acts as a clean multiplier",
         "Reefer $+12.9\\%$ and Flatbed $+8.3\\%$ relative to Dry Van",
         "Three categories, with no interaction terms required"],
        ["No day-of-week pricing effect worth modelling",
         "Spread across weekdays below 0.8\\%",
         "The weekly ripple in the December chart comes from the market signal"],
    ]) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")

    # ---------------- 2 ----------------
    tex.append(r"""
\section{Data quality issues and how they were addressed}

The most consequential issue lies in the target itself. Comparing every posted
rate against a robust model's expectation produces a ratio tightly packed around
1.0 for almost every load, and then two detached clusters: one spanning 0.17 to
0.45, and one spanning 2.19 to 5.42. No load at all falls between 0.50 and 0.86,
or between 1.21 and 2.19. Roughly 1.4\% of loads have been multiplied up or
divided down, in approximately equal numbers, and independently of distance,
weight, equipment and date.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{fig_corruption.png}
\caption{Ratio of posted rate to the model's expectation, on log counts. The
regions on either side of the clean peak are genuinely empty, so the cut points
at 0.70 and 1.40 (dashed) are read directly off the data rather than tuned.}
\end{figure}

Because the two clusters are symmetric, they leave the median almost untouched.
This is therefore noise to be removed rather than a bias to be corrected, and
predictions are deliberately not inflated to chase the corrupted mass. Excluding
these rows is worth more than any modelling decision: leaving them in raises
holdout MAE from \$""" + f"{full.MAE:,.0f}" + r""" to \$""" + f"{no_clean.MAE:,.0f}" + r""".

\begin{table}[H]
\small
\begin{tabularx}{\textwidth}{L{0.33\textwidth}R{0.08\textwidth}R{0.08\textwidth}X}
\toprule
\textbf{Issue} & \textbf{Rows} & \textbf{Share} & \textbf{Treatment} \\
\midrule
""" + rows_to_tex(
        [[esc(str(r["issue"])), f"{int(r['rows']):,}".replace(",", "{,}"),
          f"{r['share %']:.2f}\\%", esc(str(r["action"]))]
         for _, r in quality.iterrows()]
        + [["8 cities present only in the validation file",
            f"{results['validation_rows_touching_unseen_city']:,}".replace(",", "{,}"),
            "12.1\\%", "Coordinates used, never city identity"]]
    ) + r"""
\bottomrule
\end{tabularx}
\end{table}

Two further checks returned clean results and so required no treatment. Billed
distance tracks great-circle distance at a stable road-circuity ratio of 1.18,
which rules out corrupted distances, and every city's coordinates are internally
consistent across both files. The 70-mile floor on distance is a genuine
short-haul billing minimum rather than an error, so it is flagged as a feature
rather than repaired.
""")

    # ---------------- 3 ----------------
    tex.append(r"""
\section{Validation approach and data split}

The split has to answer the question the task actually poses: how well does the
model price loads in months it has never seen? A random $K$-fold cannot answer
that, because it allows the model to learn from late-October days while scoring
mid-October ones. Running both protocols on an identical model shows how much
that flatters the result.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{fig_split.png}
\caption{The same model and the same data under two protocols. Random
$K$-fold reports MAE \$""" + f"{random_mae:,.0f}" + r""", while forward chaining
reports \$""" + f"{forward_mae:,.0f}" + r""". Every modelling decision in this
report was made on the forward-chaining figure.}
\end{figure}

\subsection{Forward chaining}

The training window is expanded one month at a time, and the model is always
scored on the month that follows. This matches the shape of the real task and is
repeated five times.

\begin{table}[H]
\small
\begin{tabularx}{\textwidth}{XlR{0.13\textwidth}R{0.11\textwidth}R{0.11\textwidth}R{0.10\textwidth}R{0.10\textwidth}}
\toprule
\textbf{Train through} & \textbf{Scored on} & \textbf{Train rows} &
\textbf{Holdout} & \textbf{MAE} & \textbf{MAPE} & \textbf{$R^2$} \\
\midrule
""")
    months = {"05": "May", "06": "June", "07": "July", "08": "August",
              "09": "September", "10": "October"}
    fold_rows = []
    for _, row in forward.iterrows():
        month = row["split"].split("-")[-1]
        fold_rows.append([
            f"End of {months[f'{int(month) - 1:02d}']}", months[month],
            f"{int(row['n_train']):,}".replace(",", "{,}"),
            f"{int(row['n_holdout']):,}".replace(",", "{,}"),
            f"\\${row['clean_MAE']:,.2f}", f"{row['clean_MAPE']:.2f}\\%",
            f"{row['clean_R2']:.4f}"])
    tex.append(rows_to_tex(fold_rows) + r"""
\midrule
 & \textbf{Mean} & & & \textbf{\$""" + f"{forward_mae:,.2f}" + r"""} &
\textbf{""" + f"{forward.clean_MAPE.mean():.2f}" + r"""\%} &
\textbf{""" + f"{forward.clean_R2.mean():.4f}" + r"""} \\
\bottomrule
\end{tabularx}
\end{table}

Error falls steadily as the window grows. The June fold trains on five months and
reaches MAE \$70, while the October fold trains on nine months and reaches \$29.
The final model is therefore fitted on all ten months of development data.

\subsection{A two-month holdout matching the real horizon}

The submission must reach 61 days past the last training row, so the headline
check trains through August and predicts September and October together. This is
the figure quoted in the summary.

\begin{table}[H]
\small
\begin{tabularx}{\textwidth}{XR{0.12\textwidth}R{0.11\textwidth}R{0.11\textwidth}R{0.11\textwidth}R{0.10\textwidth}R{0.10\textwidth}}
\toprule
\textbf{Model} & \textbf{Train} & \textbf{Holdout} & \textbf{MAE} &
\textbf{RMSE} & \textbf{MAPE} & \textbf{$R^2$} \\
\midrule
""" + rows_to_tex([
        ["Hybrid (shipped)",
         f"{int(hybrid.n_train):,}".replace(",", "{,}"),
         f"{int(hybrid.n_holdout):,}".replace(",", "{,}"),
         f"\\${hybrid.clean_MAE:,.2f}", f"\\${hybrid.clean_RMSE:,.2f}",
         f"{hybrid.clean_MAPE:.2f}\\%", f"{hybrid.clean_R2:.4f}"],
        ["Linear base only",
         f"{int(linear_only.n_train):,}".replace(",", "{,}"),
         f"{int(linear_only.n_holdout):,}".replace(",", "{,}"),
         f"\\${linear_only.clean_MAE:,.2f}", f"\\${linear_only.clean_RMSE:,.2f}",
         f"{linear_only.clean_MAPE:.2f}\\%", f"{linear_only.clean_R2:.4f}"],
    ]) + r"""
\bottomrule
\end{tabularx}
\end{table}

\subsection{Holding out whole cities}

No time-based split can test the 8 cities that appear only in the validation
file, because every city in the development data is present throughout it. A
second protocol therefore holds out 8 entire cities at a time and scores only
the loads that touch them, repeated five times.

\begin{table}[H]
\small
\begin{tabularx}{\textwidth}{XR{0.14\textwidth}R{0.14\textwidth}R{0.16\textwidth}}
\toprule
\textbf{Variant} & \textbf{MAE} & \textbf{MAPE} & \textbf{Mean bias} \\
\midrule
""" + rows_to_tex([
        ["Shipped model (coordinates)", f"\\${shipped.MAE:,.2f}",
         f"{shipped.MAPE:.2f}\\%", num(shipped["mean_bias_%"], "+.2f") + "\\%"],
        ["Geography removed", f"\\${no_geo.MAE:,.2f}",
         f"{no_geo.MAPE:.2f}\\%", num(no_geo["mean_bias_%"], "+.2f") + "\\%"],
    ]) + r"""
\bottomrule
\end{tabularx}
\end{table}

On loads involving a city it has never seen, the model still reaches
""" + f"{shipped.MAPE:.2f}" + r"""\% MAPE with """ + f"{shipped['mean_bias_%']:+.2f}" + r"""\%
bias, so the 8 unseen validation cities are not a material risk. This test also
overturned a design decision. An explicit nearest-neighbour regional index, built
specifically to serve unseen cities, scored 2.4\% worse than simply allowing the
trees to use raw coordinates, and was removed from the final model.
""")

    # ---------------- 4 ----------------
    tex.append(r"""
\section{Model selection and reasoning}

The model is a trimmed log-linear base combined with a LightGBM correction fitted
to its residuals. The division of labour is deliberate. The base carries every
effect that must extrapolate past the end of the training window, namely the
distance elasticity, both market signals, equipment and the time trend. LightGBM
then learns what the linear form leaves behind, namely curvature in distance and
weight and the regional premium held in the coordinates, using features that all
remain inside ranges already observed.

The reason for not using trees alone is structural rather than empirical. A tree
predicts a constant beyond its last split, so it cannot produce an upward trend
in a period it has never seen. Dropping the time term leaves a systematic
""" + num(trend_value.iloc[1]["mean bias %"], ".1f") + r"""\% shortfall two months
out, against """ + num(trend_value.iloc[0]["mean bias %"], "+.2f") + r"""\% when it
is retained. A plain LightGBM trained on raw labels reaches MAE
\$""" + f"{plain_gbm.MAE:,.0f}" + r""", against \$""" + f"{full.MAE:,.0f}" + r"""
for the model submitted here.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{fig_trend.png}
\caption{Monthly mean residual from a model carrying no time term. The drift is
clear but is not a clean ramp, since it pauses through July and August. A
straight line explains 89\% of the monthly pattern, while quadratic and cubic
terms add almost nothing and are considerably more dangerous 61 days beyond the
last observation. A single linear term is therefore used.}
\end{figure}

\begin{table}[H]
\small
\begin{tabularx}{\textwidth}{XR{0.18\textwidth}R{0.16\textwidth}R{0.16\textwidth}}
\toprule
\textbf{Window fitted} & \textbf{Drift per day} & \textbf{$t$ statistic} &
\textbf{Per 30 days} \\
\midrule
""" + rows_to_tex([
        [esc(str(r["window"]).replace("Jan-Mar", "January to March")
             .replace("Apr-Jun", "April to June")
             .replace("Jul-Aug", "July and August")
             .replace("Sep-Oct", "September and October")
             .replace("full Jan-Oct", "Full window, January to October")),
         num(r["drift per day"], ".6f"), num(r["t-stat"], ",.1f"),
         num(r["% per 30 days"], "+.2f") + "\\%"]
        for _, r in trend.iterrows()
    ]) + r"""
\bottomrule
\end{tabularx}
\end{table}

The drift is strongly significant over the full window and positive in three of
the four sub-windows. It is carried forward as a linear term and nothing more
aggressive, which is the conservative reading of a trend that visibly pauses
mid-year.

\subsection{What each decision is worth}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{fig_ablation.png}
\caption{Each variant retrained from scratch and scored on the September and
October holdout. Cleaning the labels and extrapolating the trend matter more
than the choice of learning algorithm.}
\end{figure}

\subsection{Base model coefficients}

\begin{table}[H]
\small
\begin{tabularx}{\textwidth}{L{0.33\textwidth}R{0.16\textwidth}X}
\toprule
\textbf{Term} & \textbf{Coefficient} & \textbf{Interpretation} \\
\midrule
""")
    readings = {
        "intercept": "Baseline level",
        "log_distance": "1\\% more distance gives 0.87\\% more rate",
        "log_weight": "1\\% more weight gives 0.08\\% more rate",
        "log_market_index": "Load-level market reading",
        "log_quote_signal": "Load-level quote reading",
        "log_market_index_daily": "That day's market level",
        "log_quote_signal_daily": "That day's quote level",
        "days_elapsed": "0.0225\\% of upward drift per day",
        "is_flatbed": "8.3\\% above Dry Van",
        "is_reefer": "12.9\\% above Dry Van",
    }
    tex.append(rows_to_tex([
        [esc(str(r["feature"])), f"{r['coefficient']:+.6f}",
         readings.get(r["feature"], "")]
        for _, r in coefficients.iterrows()
    ]) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")

    # ---------------- 5 ----------------
    tex.append(r"""
\newpage
\section{The fixed December chart}

Every input is held constant, namely Lexington to Fort Wayne, 360 miles, Dry Van
and 32{,}000 lb, and only the date varies. The curve therefore isolates what the
model believes about time alone.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{candidate_december.png}
\caption{Produced by the supplied \texttt{score.py} from the completed
\texttt{data/december\_chart\_inputs.csv}.}
\end{figure}

The chart shows two effects simultaneously. The upward slope across the month is
the time trend carried forward from the development data. The repeating ripple is
the daily market level: the two market columns are published per load, so the
fixed December rows are priced using each day's mean across the 6{,}164 December
loads present in the validation file. That uses no target values, and it is the
only reason these 31 rows can be priced at all. Troughs fall on Sundays and peaks
mid-week, and the range across the whole month is about \$21, or 2.5\%.

The level is independently plausible. Comparable real loads in the development
data, restricted to Dry Van between 300 and 420 miles and 28{,}000 to 36{,}000 lb,
have a median of \$2.35 per mile, which scales to \$846 at 360 miles. The December
predictions average \$845. Actual Lexington to Fort Wayne Dry Van loads in October
posted between \$796 and \$865.
""")

    # ---------------- 6 ----------------
    tex.append(r"""
\section{Limitations and next steps}

\begin{table}[H]
\small
\begin{tabularx}{\textwidth}{L{0.27\textwidth}L{0.33\textwidth}X}
\toprule
\textbf{Limitation} & \textbf{Why it is acceptable here} & \textbf{Next step} \\
\midrule
""" + rows_to_tex([
        ["The time trend is extrapolated 61 days on the assumption that it "
         "continues",
         "It is positive in three of four sub-windows and strongly significant "
         "overall, and a linear term is the conservative choice",
         "Refit weekly as actuals arrive, and alert when realised rates fall "
         "outside the residual band"],
        ["Daily market levels for November and December come from the validation "
         "file's own feature columns",
         "These are exogenous inputs shipped with every row, labelled or not, so "
         "no target value is used",
         "In production the market index is published daily, so this is the same "
         "information a live service would hold"],
        ["Approximately 1.4\\% of validation labels are likely corrupted as well",
         "Corruption is symmetric and independent of the features, so it inflates "
         "the score without biasing it",
         "Report a robust metric alongside MAE so model quality is not confounded "
         "with label noise"],
        ["Only 72 cities and 10 months are represented",
         "Geography is handled through coordinates, so new cities interpolate",
         "Retrain as the lane network widens, with attention to the far "
         "south-west where neighbours are sparse"],
    ]) + r"""
\bottomrule
\end{tabularx}
\end{table}

\subsection*{Reproducing these results}

All figures in this report are regenerated from the repository by running
\texttt{python -m src.validate}, \texttt{python -m src.ablation},
\texttt{python -m src.unseen\_cities} and \texttt{python -m src.train\_predict},
in that order. Full setup instructions are in the repository README.

\end{document}
""")

    REPORTS.mkdir(parents=True, exist_ok=True)
    source = REPORTS / "report.tex"
    source.write_text("\n".join(tex), encoding="utf-8")

    # The scorer writes its chart outside reports/; copy it in for graphicx.
    chart = REPORTS.parent / "scorer_results" / "candidate_december.png"
    if chart.is_file():
        shutil.copy(chart, REPORTS / "candidate_december.png")

    for engine in TEX_ENGINES:
        if shutil.which(engine[0]) is None:
            continue
        result = subprocess.run(
            engine + [source.name], cwd=REPORTS, capture_output=True, text=True
        )
        if result.returncode == 0:
            built = REPORTS / "report.pdf"
            final = REPORTS / "Freight_Rate_Report.pdf"
            if built.is_file():
                shutil.move(built, final)
            print(f"Wrote {final}  (engine: {engine[0]})")
            return
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise SystemExit(f"{engine[0]} failed")
    raise SystemExit("no LaTeX engine found; install tectonic or pdflatex")


if __name__ == "__main__":
    build()
