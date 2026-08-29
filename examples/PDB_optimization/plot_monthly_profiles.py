"""Monthly average 24-hour profiles: heat demand, prices, and storage.

For each of the 12 calendar months, plots the average 24-hour profile
(mean value at each hour of day, 0-23) of:
    * heat demand [MW]
    * electricity spot price [EUR/MWh]
    * gas price [EUR/MWh]
    * end-of-hour thermal storage level [MWh], from the PERFECT-FORESIGHT
      dispatch (re-solved here as a single full-year LP -- fast, unlike
      the causal case's 365 re-solves -- so the storage profile matches
      exactly what opt_dispatch.py itself produces)

Run:
    python plot_monthly_profiles.py

Reuses opt_dispatch.py's SOLUTIONS/capacities and run_perfect_foresight
directly (import), so this always reflects whatever design/parameters
opt_dispatch.py is currently configured with.

Output: <PLOT_DIR>/monthly_profiles.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import opt_dispatch as od

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


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


def plot_monthly_profiles(profiles, title, ylabel, ax):
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


if __name__ == "__main__":
    data = pd.read_csv(
        od.DATA_DIR / "input_data.csv", sep=";", index_col=0, parse_dates=True
    )
    max_heat_demand = data["heat demand"].max()

    print("Solving perfect-foresight dispatch to get the storage profile...")
    _, storage_pf, _ = od.run_perfect_foresight(data, max_heat_demand)

    # Hourly series (N points, matching the dispatch convention elsewhere
    # in opt_dispatch.py: N+1 boundary timestamps -> N intervals).
    demand = data["heat demand"].iloc[:-1]
    el_price = data["el_spot_price"].iloc[:-1]
    gas_price = data["gas price"].iloc[:-1]

    # storage_pf is boundary-indexed (one extra point at the very start of
    # the year, iloc[0]); "end of hour t" is storage_pf.iloc[t + 1].
    storage_eoh = pd.Series(storage_pf.iloc[1:].to_numpy(), index=demand.index)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_monthly_profiles(
        hourly_month_average(demand), "Heat demand", "MW", axes[0, 0]
    )
    plot_monthly_profiles(
        hourly_month_average(el_price),
        "Electricity spot price",
        "EUR/MWh",
        axes[0, 1],
    )
    plot_monthly_profiles(
        hourly_month_average(gas_price), "Gas price", "EUR/MWh", axes[1, 0]
    )
    plot_monthly_profiles(
        hourly_month_average(storage_eoh),
        "End-of-hour storage level (perfect foresight)",
        "MWh",
        axes[1, 1],
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Monthly average 24-hour profiles", y=1.08, fontsize=14)
    fig.tight_layout()
    out_path = PLOT_DIR / "monthly_profiles.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
