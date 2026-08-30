"""Plots for the gas boiler + heat pump + heat storage Pareto study.

Reads the CSVs written by opt_step1.py (the computation script) and
produces every plot -- this script never builds or solves an oemof model
itself.

Input (all under RESULTS_DIR, written by opt_step1.py):
    cases_summary.csv
        One row per solved case (both anchors, the HP+storage optimum,
        and the 6-point lambda sweep).
    dispatch_<case>.csv
        Hourly dispatch time series for the two cases plotted in detail
        (the cheapest full-system design and the lambda = 20 design),
        including gas/electricity price and storage content columns.

Output (under PLOT_DIR):
    pareto_1_anchors.png, pareto_2_hp_storage.png, pareto_3_full.png
    dispatch_<case>.png, storage_content_<case>.png
    monthly_profiles_<case>.png
        12-month average 24-hour profiles of heat demand, electricity
        price, gas price, and end-of-hour storage level, for each of the
        two detailed cases.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = Path("plots")
RESULTS_DIR = Path("results")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def slugify(label):
    """Filename-safe form of a case label, e.g. 'full λ=0' -> 'full_lambda_0'.

    Kept identical to opt_step1.py's own slugify so filenames match.
    """
    label = label.replace("λ", "lambda").replace("=", " ")
    return "_".join(
        "".join(ch for ch in part if ch.isascii() and ch.isalnum())
        for part in label.split()
    )


# =============================================================================
# Load everything from CSV
# =============================================================================
cases_df = pd.read_csv(RESULTS_DIR / "cases_summary.csv")
_all_case_dicts = cases_df.to_dict(orient="records")

boiler_only = next(c for c in _all_case_dicts if c["case_label"] == "gas boiler only")
hp_only = next(c for c in _all_case_dicts if c["case_label"] == "heat pump only")
hp_storage_opt = next(c for c in _all_case_dicts if c["case_label"] == "HP+sto optimum")
full_sweep = [c for c in _all_case_dicts if c["case_label"].startswith("full ")]
full_sweep.sort(key=lambda c: c["co2_price"])

# best_full/case_l20 mirror opt_step1.py's own selection logic exactly, so
# the plot script always picks up the same two cases even if the lambda
# sweep values change.
best_full = min(full_sweep, key=lambda c: c["lcoh"])
case_l20 = next(c for c in full_sweep if c["co2_price"] == 20)


# =============================================================================
# Pareto plots (1, 2, 3) -- ported from opt_step1.py, reading from the
# case dicts reconstructed above instead of from a live solve.
# =============================================================================
_all_cases = [boiler_only, hp_only, hp_storage_opt] + full_sweep
_co2 = [c["co2"] for c in _all_cases]
_lcoh = [c["lcoh"] for c in _all_cases]

XLIM = (
    min(_co2) - 0.06 * (max(_co2) - min(_co2)),
    max(_co2) + 0.06 * (max(_co2) - min(_co2)),
)
YLIM = (
    min(_lcoh) - 0.18 * (max(_lcoh) - min(_lcoh)),
    max(_lcoh) + 0.12 * (max(_lcoh) - min(_lcoh)),
)
XLIM3 = (1500, 13500)
YLIM3 = (14, 26)

SIZE_NOTE = "labels: heat pump [MW] / storage [MWh] / gas boiler [MW]"


def style_pareto(ax, xlim=XLIM, ylim=YLIM):
    ax.set_xlabel("Total CO2 [tCO2]")
    ax.set_ylabel("LCOH [€/MWh]")
    ax.grid(True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(
        0.01,
        0.02,
        SIZE_NOTE,
        transform=ax.transAxes,
        fontsize=7.5,
        style="italic",
        color="0.35",
        va="bottom",
        ha="left",
    )


def size_label(c):
    return (
        f"{c['cap_heat_pump_mw']:.1f} / {c['cap_storage_mwh']:.0f}"
        f" / {c['cap_gas_boiler_mw']:.1f}"
    )


def annotate_case(ax, c, dx=9, dy=-3, ha="left"):
    ax.annotate(
        size_label(c),
        (c["co2"], c["lcoh"]),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=7.5,
        ha=ha,
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
        },
    )


def plot_anchors(ax, annotate=False):
    ax.scatter(
        boiler_only["co2"],
        boiler_only["lcoh"],
        label="gas boiler only",
        marker="s",
        s=130,
        color="#EC6707",
        zorder=4,
    )
    ax.scatter(
        hp_only["co2"],
        hp_only["lcoh"],
        label="heat pump only",
        marker="^",
        s=130,
        color="#B54036",
        zorder=4,
    )
    if annotate:
        annotate_case(ax, boiler_only, dx=-11, ha="right")
        annotate_case(ax, hp_only)


def plot_hp_storage_opt(ax, annotate=True, hollow=False):
    fill = "none" if hollow else "#00395B"
    ax.scatter(
        hp_storage_opt["co2"],
        hp_storage_opt["lcoh"],
        label="heat pump + storage (single-objective)",
        marker="o",
        s=110,
        facecolors=fill,
        edgecolors="#00395B",
        linewidths=1.8,
        zorder=4,
    )
    if annotate:
        annotate_case(ax, hp_storage_opt, dx=11)


def pareto_front(cases):
    front = [
        c
        for c in cases
        if not any(
            o["co2"] <= c["co2"]
            and o["lcoh"] <= c["lcoh"]
            and (o["co2"] < c["co2"] or o["lcoh"] < c["lcoh"])
            for o in cases
        )
    ]
    return sorted(front, key=lambda c: c["co2"])


def plot_pareto_line(ax, cases, extend_right=False):
    front = pareto_front(cases)
    xs = [c["co2"] for c in front]
    ys = [c["lcoh"] for c in front]
    if extend_right:
        xs.append(XLIM3[1])
        ys.append(ys[-1])
    ax.plot(
        xs, ys, color="black", linestyle="--", linewidth=1.3, zorder=2,
        label="Pareto front",
    )
    return front


def plot_sweep(ax, cases, label, marker, color, annotate=lambda c: f"{c['co2_price']:.0f}"):
    ax.scatter(
        [c["co2"] for c in cases],
        [c["lcoh"] for c in cases],
        label=label, marker=marker, s=70, color=color, zorder=3,
    )
    if annotate is None:
        return
    for c in cases:
        ax.annotate(
            annotate(c), (c["co2"], c["lcoh"]), textcoords="offset points",
            xytext=(6, 4), fontsize=7,
        )


def make_pareto_plots():
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    plot_anchors(ax1, annotate=True)
    style_pareto(ax1)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    plot_anchors(ax2, annotate=True)
    plot_hp_storage_opt(ax2)
    style_pareto(ax2)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(9, 6))
    plot_anchors(ax3, annotate=True)
    plot_hp_storage_opt(ax3, annotate=False, hollow=True)
    plot_sweep(
        ax3, full_sweep, "heat pump + storage + gas boiler (multi-objective)",
        "D", "#7A9A01", annotate=None,
    )
    plot_pareto_line(
        ax3, [boiler_only, hp_only, hp_storage_opt] + list(full_sweep),
        extend_right=True,
    )

    cluster = sorted(
        [hp_storage_opt] + [c for c in full_sweep if c["co2_price"] > 0],
        key=lambda c: -c["lcoh"],
    )
    for i, c in enumerate(cluster):
        ax3.annotate(
            size_label(c),
            (c["co2"], c["lcoh"]),
            xytext=(4600, 20.2 - 0.5 * i),
            textcoords="data",
            fontsize=7.5,
            va="center",
            ha="left",
            arrowprops={
                "arrowstyle": "-", "color": "0.6", "linewidth": 0.6,
                "shrinkA": 0, "shrinkB": 4,
            },
            bbox={
                "boxstyle": "square,pad=0.15", "facecolor": "white",
                "edgecolor": "none",
            },
        )
    annotate_case(ax3, next(c for c in full_sweep if c["co2_price"] == 0))
    style_pareto(ax3, XLIM3, YLIM3)
    fig3.tight_layout()

    for fig, name in (
        (fig1, "pareto_1_anchors"),
        (fig2, "pareto_2_hp_storage"),
        (fig3, "pareto_3_full"),
    ):
        fig.savefig(PLOT_DIR / f"{name}.png", dpi=150)


# =============================================================================
# Per-case dispatch + storage plots -- ported from opt_step1.py's
# plot_dispatch, reading the saved CSV instead of a live oemof result.
# =============================================================================
def plot_case_dispatch(dispatch, case_label, slug):
    unit_colors = {
        "gas boiler": "#EC6707",
        "heat pump": "#B54036",
        "heat storage (discharge)": "#BFBFBF",
        "heat storage (charge)": "#696969",
    }
    fig_dispatch, ax = plt.subplots(figsize=[10, 6])
    bottom = 0
    for col, label in [
        ("heat_pump", "heat pump"),
        ("gas_boiler", "gas boiler"),
        ("storage_discharge", "heat storage (discharge)"),
    ]:
        ax.bar(dispatch.index, dispatch[col], label=label, color=unit_colors[label], bottom=bottom)
        bottom = bottom + dispatch[col]
    ax.bar(
        dispatch.index, -dispatch["storage_charge"],
        label="heat storage (charge)", color=unit_colors["heat storage (charge)"],
    )
    ax.legend(loc="upper center", ncol=2)
    ax.grid(axis="y")
    ax.set_ylim(-22, 22)
    ax.set_ylabel("Hourly heat production in MWh")
    ax.set_title(f"Dispatch -- {case_label}")
    fig_dispatch.tight_layout()
    fig_dispatch.savefig(PLOT_DIR / f"dispatch_{slug}.png", dpi=150)

    fig_storage, ax = plt.subplots(figsize=[10, 6])
    ax.plot(dispatch.index, dispatch["storage_content"], color="#00395B")
    ax.grid(axis="y")
    ax.set_ylabel("Hourly heat storage content in MWh")
    ax.set_title(f"Storage content -- {case_label}")
    fig_storage.tight_layout()
    fig_storage.savefig(PLOT_DIR / f"storage_content_{slug}.png", dpi=150)


# =============================================================================
# Monthly average 24-hour profiles (new): heat demand, electricity price,
# gas price, and end-of-hour storage level, for a given detailed case.
# =============================================================================
def hourly_month_average(series):
    df = pd.DataFrame(
        {"value": series.to_numpy(), "month": series.index.month, "hour": series.index.hour}
    )
    return df.pivot_table(index="month", columns="hour", values="value", aggfunc="mean")


def plot_monthly_profile_panel(profiles, title, ylabel, ax):
    cmap = plt.get_cmap("twilight_shifted")
    for month in range(1, 13):
        ax.plot(profiles.columns, profiles.loc[month], color=cmap((month - 1) / 11), label=MONTH_NAMES[month - 1])
    ax.set_title(title)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(0, 24, 4))
    ax.grid(alpha=0.3)


def plot_case_monthly_profiles(dispatch, case_label, slug):
    """dispatch must have heat_demand, el_spot_price, gas_price, and
    storage_content columns (all present in opt_step1.py's saved CSVs).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_monthly_profile_panel(hourly_month_average(dispatch["heat_demand"]), "Heat demand", "MW", axes[0, 0])
    plot_monthly_profile_panel(hourly_month_average(dispatch["el_spot_price"]), "Electricity spot price", "EUR/MWh", axes[0, 1])
    plot_monthly_profile_panel(hourly_month_average(dispatch["gas_price"]), "Gas price", "EUR/MWh", axes[1, 0])
    plot_monthly_profile_panel(hourly_month_average(dispatch["storage_content"]), "End-of-hour storage level", "MWh", axes[1, 1])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Monthly average 24-hour profiles -- {case_label}", y=1.08, fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"monthly_profiles_{slug}.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    make_pareto_plots()

    for case in (best_full, case_l20):
        slug = slugify(case["case_label"])
        dispatch = pd.read_csv(
            RESULTS_DIR / f"dispatch_{slug}.csv", index_col=0, parse_dates=True
        )
        plot_case_dispatch(dispatch, case["case_label"], slug)
        plot_case_monthly_profiles(dispatch, case["case_label"], slug)

    print(f"Plots written to {PLOT_DIR}/")
