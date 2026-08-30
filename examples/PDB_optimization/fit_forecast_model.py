"""Sinusoidal forecast model for heat demand, gas price, and electricity
spot price.

Fits, to one year of historical data (<DATA_DIR>/input_data.csv -- used
here as a stand-in for genuine multi-year historical data, since only one
year is available), a per-variable model built from a small set of
building blocks, all at FIXED, KNOWN periods so the whole fit stays
LINEAR in its coefficients (solved exactly via ordinary least squares --
no iterative nonlinear fitting, no risk of local minima):

    * ANNUAL harmonics (period 8760h, 4380h, ...): the winter/summer cycle
      and its first overtone (captures e.g. a spring/autumn dip distinct
      from a single winter-high/summer-low sine).
    * WEEKLY cycle (168h): weekday/weekend effects -- tested and found to
      matter only for electricity price (industrial/commercial activity
      driving the market), not heat demand or gas price.
    * DAILY harmonics (24h fundamental + 12h first overtone, i.e. two
      humps per day): needed because heat demand and electricity price
      both show a genuine morning-AND-evening double peak, which a single
      24h sine cannot represent at all.
    * SEASONAL MODULATION of the daily terms: the daily cycle's own
      amplitude/phase is allowed to vary smoothly over the year (daily
      sin/cos terms multiplied by the annual fundamental's sin/cos) --
      still linear, since it's just more fixed basis functions -- capturing
      e.g. a bigger daily swing on a cold winter day than a mild one.

Model-building history (see forecast_model_fit.png/params.csv from each
stage for the actual measured numbers):
    1. Single annual + single daily sine per variable: heat demand
       R2=0.867, gas price R2=0.830, electricity price R2=0.056.
    2. Added a daily 2nd harmonic (24h+12h) for heat demand/electricity
       price (gas price has ~no daily pattern, so kept annual-only):
       heat demand R2=0.880, electricity price R2=0.161.
    3. Confirmed (empirically, via a separate nonlinear amplitude+phase
       fit) that the sin+cos parameterization already has fully free
       phase -- adding explicit phase parameters cannot improve the fit,
       since a*sin(wt)+b*cos(wt) spans the same function space as
       R*sin(wt+phi) for any phase phi.
    4. Added seasonal modulation of the daily terms: heat demand
       R2=0.891, electricity price R2=0.199.
    5. Tested more annual harmonics: gas price's R2 climbs sharply
       (0.83 -> 0.94 by 5 harmonics), which is a red flag rather than a
       win -- it means gas price actually moves in discrete steps
       (e.g. monthly resets), and a Fourier series needs many harmonics
       to approximate a step function (Gibbs phenomenon). That isn't a
       repeatable seasonal pattern, so annual harmonics are deliberately
       CAPPED at 2 for every variable here rather than chasing R2.
    6. Added a weekly (168h) cycle -- tested and found to matter only for
       electricity price (0.173 -> 0.255 with weekly alone, -> 0.293
       combined with seasonal daily modulation); negligible for heat
       demand/gas price, so it's electricity-price-only in the final
       model below.

Even the final electricity-price model only reaches R2~0.29: the
remaining variance is very likely genuine weather/renewable-output-driven
volatility with no periodic structure at all, which no sum of fixed-period
sinusoids can capture without actual exogenous weather/generation data.

Output:
    <RESULTS_DIR>/forecast_model_params.csv
        One row per variable: fitted coefficients and the basis-function
        spec used (serialized, since it differs by variable), plus
        fit-quality metrics (R2, RMSE, MAE) and a simple daily-amplitude
        min/max summary (min/max because modulated daily amplitude is now
        itself a function of time of year, not a single number).
    <PLOT_DIR>/forecast_model_fit.png
        Monthly average 24-hour profiles, real data as points and the
        fitted model's prediction for a representative day of each month
        overlaid as a curve.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")
RESULTS_DIR = Path("results")
for _dir in (DATA_DIR, PLOT_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

ANNUAL_HOURS = 8760  # exactly one non-leap year of hourly data
WEEKLY_HOURS = 168
DAILY_HOURS = 24
DAILY_2ND_HARMONIC_HOURS = 12

# Per-variable model spec: how many annual harmonics (period 8760/k for
# k=1..n), whether to include a weekly term, which daily periods to
# include, and whether those daily terms get seasonal-modulation columns.
# See the module docstring for why each choice was made.
VARIABLES = {
    "heat demand": dict(
        key="heat_demand", label="Heat demand", unit="MW",
        n_annual_harmonics=2, weekly=False, weekend_effect=False,
        daily_periods=[DAILY_HOURS, DAILY_2ND_HARMONIC_HOURS],
        daily_seasonal_modulation=True,
    ),
    "gas price": dict(
        key="gas_price", label="Gas price", unit="EUR/MWh",
        n_annual_harmonics=2, weekly=False, weekend_effect=False,
        daily_periods=[], daily_seasonal_modulation=False,
    ),
    "el_spot_price": dict(
        key="el_spot_price", label="Electricity spot price", unit="EUR/MWh",
        n_annual_harmonics=2, weekly=True, weekend_effect=True,
        daily_periods=[DAILY_HOURS, DAILY_2ND_HARMONIC_HOURS],
        daily_seasonal_modulation=True,
    ),
}
# weekend_effect (tested for both heat demand and electricity price, kept
# for electricity price only): a discrete is_weekend dummy, PLUS is_weekend
# interacted with each daily period's sin/cos so the daily SHAPE (not just
# its average level) can differ on weekends -- e.g. a flatter midday when
# less commercial/industrial activity is happening. Both pieces measured
# to help meaningfully for electricity price:
#   no weekly term at all:                        R2=0.211
#   smooth weekly sinusoid (168h) alone:           R2=0.293
#   is_weekend level-shift alone (1 parameter):    R2=0.309 (beats the
#     168h sinusoid's 2 parameters with fewer parameters -- the weekly
#     effect is more of a step than a smooth wave)
#   is_weekend level+shape (interacts daily terms): R2=0.331
#   is_weekend level+shape + smooth 168h sinusoid:  R2=0.337 (final choice
#     here -- the smooth term still adds a little on top, for whatever
#     partial-week structure the binary split doesn't capture)
# For heat demand, every version of this was measured to change nothing
# (R2 stays ~0.895 regardless): heating demand tracks outdoor temperature,
# not the workweek, so weekend_effect is off there. A full day-of-week (6
# dummy variables) breakdown was also tried for electricity price and
# matched the level+shape result almost exactly (R2=0.315 alone, same as
# level+shape) -- so weekday/weekend is essentially the whole story; finer
# day-of-week granularity wasn't kept since it added parameters for no
# real gain.

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def build_design_matrix(t, spec, dow0=0):
    """Build the design matrix and matching column-name list for one
    variable's spec (see VARIABLES). Seasonal modulation of a daily period
    p multiplies that period's sin/cos by the ANNUAL FUNDAMENTAL's
    (period=ANNUAL_HOURS, i.e. k=1) sin/cos -- not higher annual harmonics
    -- to keep the modulation itself simple (one extra full annual cycle
    of variation in the daily amplitude/phase, not several).

    `dow0` is the day-of-week (Monday=0 .. Sunday=6) of hour-of-year t=0,
    needed only when `spec["weekend_effect"]` is set -- t alone doesn't
    carry calendar information, so the caller must supply this once from
    the actual data (`data.index[0].dayofweek`).
    """
    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t)]
    names = ["const"]

    annual_fundamental = None
    for k in range(1, spec["n_annual_harmonics"] + 1):
        p = ANNUAL_HOURS / k
        s, c = np.sin(2 * np.pi * t / p), np.cos(2 * np.pi * t / p)
        cols += [s, c]
        names += [f"sin_annual{k}", f"cos_annual{k}"]
        if k == 1:
            annual_fundamental = (s, c)

    if spec["weekly"]:
        s, c = np.sin(2 * np.pi * t / WEEKLY_HOURS), np.cos(2 * np.pi * t / WEEKLY_HOURS)
        cols += [s, c]
        names += ["sin_weekly", "cos_weekly"]

    daily_sincos = {}
    for p in spec["daily_periods"]:
        s, c = np.sin(2 * np.pi * t / p), np.cos(2 * np.pi * t / p)
        daily_sincos[p] = (s, c)
        cols += [s, c]
        names += [f"sin_{p}h", f"cos_{p}h"]
        if spec["daily_seasonal_modulation"]:
            sa, ca = annual_fundamental
            cols += [s * sa, s * ca, c * sa, c * ca]
            names += [
                f"sin_{p}h_x_sin_annual", f"sin_{p}h_x_cos_annual",
                f"cos_{p}h_x_sin_annual", f"cos_{p}h_x_cos_annual",
            ]

    if spec.get("weekend_effect"):
        day_index = np.floor(t / 24).astype(np.int64)
        dow = (day_index + dow0) % 7
        is_weekend = (dow >= 5).astype(float)  # Sat=5, Sun=6 (Monday=0 convention)
        cols.append(is_weekend)
        names.append("is_weekend")
        # Weekend-specific daily SHAPE, not just a level shift: interact
        # is_weekend with each daily period's own sin/cos (measured to
        # help substantially more than the level shift alone -- see
        # VARIABLES' weekend_effect comment).
        for p in spec["daily_periods"]:
            s, c = daily_sincos[p]
            cols += [is_weekend * s, is_weekend * c]
            names += [f"is_weekend_x_sin_{p}h", f"is_weekend_x_cos_{p}h"]

    return np.column_stack(cols), names


def fit_model(t, y, spec, dow0=0):
    X, names = build_design_matrix(t, spec, dow0=dow0)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    metrics = {
        "r2": 1 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
    }
    return coeffs, names, metrics


def predict(coeffs, t, spec, dow0=0):
    X, _ = build_design_matrix(t, spec, dow0=dow0)
    return X @ coeffs


def summarize_params(coeffs, names, t_full, spec):
    """Interpretable summary: baseline, each simple (non-modulated)
    term's amplitude/peak time, and -- for seasonally-modulated daily
    terms, whose amplitude is itself a function of time of year -- the
    min/max amplitude reached anywhere across the year.
    """
    idx = {name: i for i, name in enumerate(names)}
    summary = {"baseline": float(coeffs[idx["const"]])}

    for k in range(1, spec["n_annual_harmonics"] + 1):
        p = ANNUAL_HOURS / k
        a, b = coeffs[idx[f"sin_annual{k}"]], coeffs[idx[f"cos_annual{k}"]]
        amplitude = float(np.hypot(a, b))
        phase = np.arctan2(b, a)
        peak = ((np.pi / 2 - phase) / (2 * np.pi / p)) % p
        summary[f"annual_harmonic{k}_amplitude"] = amplitude
        summary[f"annual_harmonic{k}_peak_day_of_year"] = float(peak / 24)

    if spec["weekly"]:
        a, b = coeffs[idx["sin_weekly"]], coeffs[idx["cos_weekly"]]
        amplitude = float(np.hypot(a, b))
        phase = np.arctan2(b, a)
        peak = ((np.pi / 2 - phase) / (2 * np.pi / WEEKLY_HOURS)) % WEEKLY_HOURS
        summary["weekly_amplitude"] = amplitude
        summary["weekly_peak_day_of_week"] = float(peak / 24)  # 0=same weekday as t=0

    for p in spec["daily_periods"]:
        a0, b0 = coeffs[idx[f"sin_{p}h"]], coeffs[idx[f"cos_{p}h"]]
        if spec["daily_seasonal_modulation"]:
            a_ss, a_sc = coeffs[idx[f"sin_{p}h_x_sin_annual"]], coeffs[idx[f"sin_{p}h_x_cos_annual"]]
            b_ss, b_sc = coeffs[idx[f"cos_{p}h_x_sin_annual"]], coeffs[idx[f"cos_{p}h_x_cos_annual"]]
            sa = np.sin(2 * np.pi * t_full / ANNUAL_HOURS)
            ca = np.cos(2 * np.pi * t_full / ANNUAL_HOURS)
            eff_a = a0 + a_ss * sa + a_sc * ca
            eff_b = b0 + b_ss * sa + b_sc * ca
            amplitude_series = np.hypot(eff_a, eff_b)
            summary[f"daily_{p}h_amplitude_min"] = float(amplitude_series.min())
            summary[f"daily_{p}h_amplitude_max"] = float(amplitude_series.max())
        else:
            summary[f"daily_{p}h_amplitude_min"] = float(np.hypot(a0, b0))
            summary[f"daily_{p}h_amplitude_max"] = summary[f"daily_{p}h_amplitude_min"]

    if spec.get("weekend_effect"):
        summary["weekend_level_shift"] = float(coeffs[idx["is_weekend"]])
        for p in spec["daily_periods"]:
            a, b = coeffs[idx[f"is_weekend_x_sin_{p}h"]], coeffs[idx[f"is_weekend_x_cos_{p}h"]]
            summary[f"weekend_daily_{p}h_shape_change_amplitude"] = float(np.hypot(a, b))

    return summary


def hourly_month_average(series):
    df = pd.DataFrame(
        {
            "value": series.to_numpy(),
            "month": series.index.month,
            "hour": series.index.hour,
        }
    )
    return df.pivot_table(index="month", columns="hour", values="value", aggfunc="mean")


def representative_day_start(data_index, month):
    positions = np.where(data_index.month == month)[0]
    mid_ts = data_index[positions[len(positions) // 2]]
    return data_index.get_loc(mid_ts.normalize())


def plot_fit(data, fits, metrics, dow0):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cmap = plt.get_cmap("twilight_shifted")
    curve_hours = np.linspace(0, 23, 100)

    panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    for ax, (col, spec) in zip(panel_axes, VARIABLES.items()):
        series = data[col]
        profiles = hourly_month_average(series)
        coeffs = fits[spec["key"]]
        for month in range(1, 13):
            color = cmap((month - 1) / 11)
            ax.scatter(
                profiles.columns, profiles.loc[month],
                color=color, s=16, zorder=3,
                label=MONTH_NAMES[month - 1] if ax is panel_axes[0] else None,
            )
            day_start = representative_day_start(series.index, month)
            t_curve = day_start + curve_hours
            ax.plot(curve_hours, predict(coeffs, t_curve, spec, dow0=dow0), color=color, linewidth=1.3, zorder=2)
        note = f"{spec['n_annual_harmonics']}A"
        if spec["weekly"]:
            note += "+W"
        if spec.get("weekend_effect"):
            note += "+WE"
        if spec["daily_periods"]:
            note += "+D(" + "+".join(f"{p}h" for p in spec["daily_periods"]) + ")"
            if spec["daily_seasonal_modulation"]:
                note += " mod"
        ax.set_title(f"{spec['label']} -- data vs. model [{note}]")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(spec["unit"])
        ax.set_xticks(range(0, 24, 4))
        ax.grid(alpha=0.3)

    handles, labels = panel_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04))

    ax_text = axes[1, 1]
    ax_text.axis("off")
    lines = ["Fit quality:\n"]
    for col, spec in VARIABLES.items():
        m = metrics[spec["key"]]
        lines.append(f"{spec['label']}: R^2={m['r2']:.3f}  RMSE={m['rmse']:.2f} {spec['unit']}  MAE={m['mae']:.2f} {spec['unit']}")
    ax_text.text(0.0, 0.8, "\n".join(lines), fontsize=11, va="top", family="monospace")

    fig.suptitle("Forecast model fit: monthly 24-hour profiles", y=1.08, fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "forecast_model_fit.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    data = pd.read_csv(
        DATA_DIR / "input_data.csv", sep=";", index_col=0, parse_dates=True
    )
    t_full = np.arange(len(data))
    dow0 = int(data.index[0].dayofweek)  # Monday=0 .. Sunday=6, needed for weekend_effect

    fits = {}
    metrics = {}
    rows = []
    for col, spec in VARIABLES.items():
        y = data[col].to_numpy()
        coeffs, names, m = fit_model(t_full, y, spec, dow0=dow0)
        fits[spec["key"]] = coeffs
        metrics[spec["key"]] = m
        summary = summarize_params(coeffs, names, t_full, spec)

        note = f"{spec['n_annual_harmonics']} annual harmonic(s)"
        if spec["weekly"]:
            note += " + weekly"
        if spec.get("weekend_effect"):
            note += " + weekend level+shape"
        if spec["daily_periods"]:
            note += f" + daily {spec['daily_periods']}" + (" (seasonally modulated)" if spec["daily_seasonal_modulation"] else "")
        print(f"{spec['label']} [{note}]: R^2={m['r2']:.4f} RMSE={m['rmse']:.3f} {spec['unit']} MAE={m['mae']:.3f} {spec['unit']}")
        print(f"  baseline={summary['baseline']:.3f}")
        for k in range(1, spec["n_annual_harmonics"] + 1):
            print(f"  annual harmonic {k}: amplitude={summary[f'annual_harmonic{k}_amplitude']:.3f}, peaks ~day {summary[f'annual_harmonic{k}_peak_day_of_year']:.0f}")
        if spec["weekly"]:
            print(f"  weekly: amplitude={summary['weekly_amplitude']:.3f}")
        if spec.get("weekend_effect"):
            print(f"  weekend level shift: {summary['weekend_level_shift']:.3f}")
            for p in spec["daily_periods"]:
                print(f"  weekend daily {p}h shape-change amplitude: {summary[f'weekend_daily_{p}h_shape_change_amplitude']:.3f}")
        for p in spec["daily_periods"]:
            lo, hi = summary[f"daily_{p}h_amplitude_min"], summary[f"daily_{p}h_amplitude_max"]
            print(f"  daily {p}h: amplitude ranges {lo:.3f} to {hi:.3f} across the year")

        rows.append(
            {
                "variable": spec["key"],
                "column": col,
                "column_names": ",".join(names),
                "coeffs": ",".join(f"{c:.6f}" for c in coeffs),
                "dow0": dow0,
                **summary,
                **m,
            }
        )

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "forecast_model_params.csv", index=False)
    plot_fit(data, fits, metrics, dow0)
    print(f"\nSaved {RESULTS_DIR / 'forecast_model_params.csv'} and {PLOT_DIR / 'forecast_model_fit.png'}")
