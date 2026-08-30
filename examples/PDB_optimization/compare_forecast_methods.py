"""Head-to-head forecast accuracy: repeating-persistence vs. sine model.

For heat demand, gas price, and electricity spot price, compares two
forecasting methods against what actually happened, at several lookahead
horizons:

    1. The repeating-24h persistence forecast already used in
       opt_dispatch.py's causal dispatch case (make_persistence_forecast):
       tile the last known day's pattern forward.
    2. The sine-sum model from fit_forecast_model.py: a fixed function of
       calendar time (annual + weekly + daily harmonics), fit ONCE on the
       whole year.

IMPORTANT CAVEAT: the sine model here is fit on the ENTIRE year, including
data that is "in the future" relative to any given forecast point being
evaluated -- this is a best-case, NON-CAUSAL comparison (an "oracle"
version of the sine model). With only one year of data available, there is
no way to fit a genuinely causal version: reliably estimating an annual
cycle requires having already observed at least one full cycle, which a
single year cannot provide. Read the results as "if reliable long-term
history were available to fit this model's parameters, would using it
instead of persistence help?" -- not as a result that could be deployed
as-is from just this one year's data.

Only horizons AT OR BEYOND N_KNOWN (from opt_dispatch.py) are evaluated:
anything shorter is real, already-known data for both methods trivially
(zero error), not a genuine forecast test.

Output:
    <RESULTS_DIR>/forecast_method_comparison.csv
        RMSE/MAE for both methods, per variable, per horizon.
    <PLOT_DIR>/forecast_method_comparison.png
        RMSE vs. horizon, one panel per variable, both methods overlaid.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import opt_dispatch as od
import fit_forecast_model as ffm

PLOT_DIR = Path("plots")
RESULTS_DIR = Path("results")
for _dir in (PLOT_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

HORIZONS = [24, 30, 36, 42, 48]  # must all be >= od.N_KNOWN
STRIDE = 24  # one comparison point per day

VARIABLE_LABELS = {
    "heat_demand": "Heat demand (MW)",
    "gas_price": "Gas price (EUR/MWh)",
    "el_spot_price": "Electricity price (EUR/MWh)",
}


def fit_oracle_sine_models(data, t_full):
    """Fit each variable's sine model once on the whole year -- see the
    module docstring's caveat about why this isn't a causal comparison.
    """
    fits = {}
    for col, spec in ffm.VARIABLES.items():
        y = data[col].to_numpy()
        coeffs, _names, _m = ffm.fit_model(t_full, y, spec)
        fits[spec["key"]] = coeffs
    return fits


def compare(data, fits):
    n_known = od.N_KNOWN
    n_total = len(data) - 1
    max_h = max(HORIZONS)
    rows = []

    for col, spec in ffm.VARIABLES.items():
        y_real = data[col].to_numpy()
        errs_persist = {h: [] for h in HORIZONS}
        errs_sine = {h: [] for h in HORIZONS}

        for t0 in range(n_known, n_total - max_h, STRIDE):
            window = od.make_persistence_forecast(
                data, t0, max_h, n_known=n_known
            )
            forecast_persist = window[col].to_numpy()
            for h in HORIZONS:
                real_val = y_real[t0 + h]
                errs_persist[h].append(real_val - forecast_persist[h])
                sine_val = ffm.predict(
                    fits[spec["key"]], np.array([t0 + h]), spec
                )[0]
                errs_sine[h].append(real_val - sine_val)

        for h in HORIZONS:
            pe, se = np.array(errs_persist[h]), np.array(errs_sine[h])
            rows.append(
                {
                    "variable": spec["key"],
                    "hours_into_forecast": h - n_known + 1,
                    "persistence_rmse": np.sqrt(np.mean(pe**2)),
                    "sine_model_rmse": np.sqrt(np.mean(se**2)),
                    "persistence_mae": np.mean(np.abs(pe)),
                    "sine_model_mae": np.mean(np.abs(se)),
                }
            )
    return pd.DataFrame(rows)


def plot_comparison(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (var, label) in zip(axes, VARIABLE_LABELS.items()):
        sub = df[df["variable"] == var]
        ax.plot(
            sub["hours_into_forecast"],
            sub["persistence_rmse"],
            "o-",
            label="persistence",
            color="#EC6707",
        )
        ax.plot(
            sub["hours_into_forecast"],
            sub["sine_model_rmse"],
            "o-",
            label="sine model",
            color="#00395B",
        )
        ax.set_title(label)
        ax.set_xlabel("Hours into forecast")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Forecast RMSE vs. horizon: persistence vs. sine model")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "forecast_method_comparison.png", dpi=150)


if __name__ == "__main__":
    data = pd.read_csv(
        od.DATA_DIR / "input_data.csv", sep=";", index_col=0, parse_dates=True
    )
    t_full = np.arange(len(data))

    fits = fit_oracle_sine_models(data, t_full)
    df = compare(data, fits)

    pd.set_option("display.width", 140)
    print(df.round(3).to_string(index=False))

    print("\nWinner by variable (lower mean RMSE across tested horizons):")
    for var in VARIABLE_LABELS:
        sub = df[df["variable"] == var]
        winner = (
            "persistence"
            if sub["persistence_rmse"].mean() < sub["sine_model_rmse"].mean()
            else "sine model"
        )
        print(f"  {var}: {winner}")

    df.to_csv(RESULTS_DIR / "forecast_method_comparison.csv", index=False)
    plot_comparison(df)
    print(
        f"\nSaved {RESULTS_DIR / 'forecast_method_comparison.csv'} and {PLOT_DIR / 'forecast_method_comparison.png'}"
    )
