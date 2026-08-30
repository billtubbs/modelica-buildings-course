"""Plots for the perfect-foresight vs. causal dispatch comparison.

Reads the CSVs written by opt_dispatch.py (the computation script) and
produces every plot -- this script never builds or solves an oemof model
itself.

Input (all under RESULTS_DIR, written by opt_dispatch.py):
    dispatch_perfect_foresight.csv, dispatch_causal.csv
        Hourly dispatch time series (gas_boiler, heat_pump,
        storage_discharge, storage_charge, heat_demand, unserved_heat).
    storage_perfect_foresight.csv, storage_causal.csv
        Boundary-indexed storage-content (state of charge) series.
Also reads <DATA_DIR>/input_data.csv directly for the raw gas/electricity
price columns, since those are original input data, not computed results,
and were never duplicated into the dispatch CSVs above.

Output (under PLOT_DIR):
    dispatch_comparison_perfect_foresight.png, dispatch_comparison_causal.png
    storage_soc_comparison.png
    monthly_profiles.png
        12-month average 24-hour profiles of heat demand, electricity
        price, gas price, and end-of-hour storage level (perfect
        foresight case).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")
RESULTS_DIR = Path("results")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def plot_dispatch(dispatch, case_label, slug):
    unit_colors = {
        "heat pump": "#B54036",
        "gas boiler": "#EC6707",
        "heat storage (discharge)": "#BFBFBF",
        "heat storage (charge)": "#696969",
        "unserved heat": "#D7263D",
    }
    fig, ax = plt.subplots(figsize=[10, 6])
    bottom = 0
    columns = [
        ("heat_pump", "heat pump"),
        ("gas_boiler", "gas boiler"),
        ("storage_discharge", "heat storage (discharge)"),
    ]
    if dispatch["unserved_heat"].max() > 1e-6:
        columns.append(("unserved_heat", "unserved heat"))
    for col, label in columns:
        ax.bar(
            dispatch.index,
            dispatch[col],
            label=label,
            color=unit_colors[label],
            bottom=bottom,
        )
        bottom = bottom + dispatch[col]
    ax.bar(
        dispatch.index,
        -dispatch["storage_charge"],
        label="heat storage (charge)",
        color=unit_colors["heat storage (charge)"],
    )
    ax.legend(loc="upper center", ncol=2)
    ax.grid(axis="y")
    ax.set_ylabel("Hourly heat production in MWh")
    ax.set_title(f"Dispatch -- {case_label}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"dispatch_comparison_{slug}.png", dpi=150)


def plot_storage_comparison(storage_pf, storage_causal):
    fig, ax = plt.subplots(figsize=[10, 6])
    ax.plot(
        storage_pf.index,
        storage_pf.to_numpy(),
        label="perfect foresight",
        color="#00395B",
    )
    ax.plot(
        storage_causal.index,
        storage_causal.to_numpy(),
        label="causal (persistence forecast)",
        color="#EC6707",
        alpha=0.8,
    )
    ax.legend()
    ax.grid(axis="y")
    ax.set_ylabel("Storage content [MWh]")
    ax.set_title("Storage state of charge: perfect foresight vs. causal")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "storage_soc_comparison.png", dpi=150)


def hourly_month_average(series):
    """Return a (12 months x 24 hours) DataFrame of average values --
    index is month (1-12), columns are hour of day (0-23).
    """
    df = pd.DataFrame(
        {
            "value": series.to_numpy(),
            "month": series.index.month,
            "hour": series.index.hour,
        }
    )
    return df.pivot_table(index="month", columns="hour", values="value", aggfunc="mean")


def plot_monthly_profile_panel(profiles, title, ylabel, ax):
    cmap = plt.get_cmap("twilight_shifted")
    for month in range(1, 13):
        ax.plot(
            profiles.columns,
            profiles.loc[month],
            color=cmap((month - 1) / 11),
            label=MONTH_NAMES[month - 1],
        )
    ax.set_title(title)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(0, 24, 4))
    ax.grid(alpha=0.3)


def plot_monthly_profiles(data, dispatch_pf, storage_pf):
    """12-month average 24-hour profiles: heat demand, electricity price,
    gas price (all raw input data), and end-of-hour storage level for the
    PERFECT-FORESIGHT case.
    """
    demand = data["heat demand"].iloc[:-1]
    el_price = data["el_spot_price"].iloc[:-1]
    gas_price = data["gas price"].iloc[:-1]

    # storage_pf is boundary-indexed (one extra point at the very start of
    # the year, iloc[0]); "end of hour t" is storage_pf.iloc[t + 1].
    storage_eoh = pd.Series(
        storage_pf.iloc[1 : len(demand) + 1].to_numpy(), index=demand.index
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_monthly_profile_panel(
        hourly_month_average(demand), "Heat demand", "MW", axes[0, 0]
    )
    plot_monthly_profile_panel(
        hourly_month_average(el_price),
        "Electricity spot price",
        "EUR/MWh",
        axes[0, 1],
    )
    plot_monthly_profile_panel(
        hourly_month_average(gas_price), "Gas price", "EUR/MWh", axes[1, 0]
    )
    plot_monthly_profile_panel(
        hourly_month_average(storage_eoh),
        "End-of-hour storage level (perfect foresight)",
        "MWh",
        axes[1, 1],
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Monthly average 24-hour profiles", y=1.08, fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "monthly_profiles.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    data = pd.read_csv(
        DATA_DIR / "input_data.csv", sep=";", index_col=0, parse_dates=True
    )
    dispatch_pf = pd.read_csv(
        RESULTS_DIR / "dispatch_perfect_foresight.csv", index_col=0, parse_dates=True
    )
    dispatch_causal = pd.read_csv(
        RESULTS_DIR / "dispatch_causal.csv", index_col=0, parse_dates=True
    )
    storage_pf = pd.read_csv(
        RESULTS_DIR / "storage_perfect_foresight.csv", index_col=0, parse_dates=True
    )["storage_content_mwh"]
    storage_causal = pd.read_csv(
        RESULTS_DIR / "storage_causal.csv", index_col=0, parse_dates=True
    )["storage_content_mwh"]

    plot_dispatch(dispatch_pf, "perfect foresight", "perfect_foresight")
    plot_dispatch(dispatch_causal, "causal (persistence forecast)", "causal")
    plot_storage_comparison(storage_pf, storage_causal)
    plot_monthly_profiles(data, dispatch_pf, storage_pf)

    print(f"Plots written to {PLOT_DIR}/")
