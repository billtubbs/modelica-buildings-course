"""Plots for the RTO vs. perfect-foresight cold-snap comparison.

Reads the CSVs written by rto_coldsnap.py -- never solves anything.

Produces, for each of the two dispatch cases:
    - An hourly view zoomed into the cold snap itself (grouped bars:
      demand vs. stacked actual supply, storage level below).
    - A daily-aggregated view (same style, but one bar per 24h period)
      over the WHOLE simulated period, to see when perfect foresight
      actually starts building up storage ahead of the cold snap.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CAP_STORAGE_MWH = 187.176 * 1.002


def plot_dispatch(dispatch, storage, title, path, zoom_start=None, zoom_periods=48, freq="h", ylim_dispatch=None, ylim_storage=None):
    """Two subplots: demand vs. stacked actual supply on top, storage
    level below. `freq="h"` for the hourly view (zoom_start required);
    `freq="D"` for the whole-period daily-aggregated view (zoom_start
    ignored, dispatch/storage are pre-aggregated by the caller).
    """
    if zoom_start is not None:
        dispatch = dispatch.loc[zoom_start:].iloc[:zoom_periods]
        storage = storage.loc[zoom_start:].iloc[: zoom_periods + 1]

    x = np.arange(len(dispatch))
    width = 0.4

    if freq == "h":
        # Storage is boundary-indexed (one more point than dispatch,
        # state AFTER each hour) -- offset so boundary i sits between
        # bar i-1 and bar i, with a step-post line/fill.
        x_storage = np.arange(len(storage)) - 0.5
        step_kw = {"step": "post"}
        draw_kw = {"drawstyle": "steps-post"}
    else:
        # Daily MEAN storage level -- one value per day, aligned directly
        # under each day's bars.
        x_storage = np.arange(len(storage))
        step_kw = {}
        draw_kw = {}

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax.bar(x - width / 2, dispatch["heat_demand"].to_numpy(), width=width, label="demand", color="#8EC6E6")

    bottom = np.zeros(len(dispatch))
    for col, label, color in [
        ("gas_boiler", "gas boiler", "#1B5E20"),
        ("heat_pump", "heat pump", "#4CAF50"),
        ("storage_discharge", "storage discharge", "#A5D6A7"),
    ]:
        values = dispatch[col].to_numpy()
        ax.bar(x + width / 2, values, width=width, bottom=bottom, label=label, color=color)
        bottom += values
    ax.bar(x + width / 2, -dispatch["storage_charge"].to_numpy(), width=width, label="storage charge", color="#FDD835")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MW" if freq == "h" else "MWh/day")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=3)
    ax.grid(axis="y", alpha=0.3)
    if ylim_dispatch is not None:
        ax.set_ylim(ylim_dispatch)

    ax2.fill_between(x_storage, storage.to_numpy(), color="#FDD835", alpha=0.4, **step_kw)
    ax2.plot(x_storage, storage.to_numpy(), color="#F9A825", linewidth=1.5, **draw_kw)
    ax2.set_ylabel("Storage (MWh)")
    ax2.set_ylim(ylim_storage if ylim_storage is not None else (0, None))
    ax2.grid(axis="y", alpha=0.3)

    step = max(1, len(x) // 12)
    date_fmt = "%d-%b %Hh" if freq == "h" else "%d-%b"
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([dispatch.index[i].strftime(date_fmt) for i in x[::step]], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def aggregate_daily(dispatch, storage):
    """Daily totals (MWh/day) for dispatch flows, daily mean for storage
    level (MWh) -- one row per calendar day.
    """
    daily_dispatch = dispatch.resample("D").sum()
    daily_storage = storage.resample("D").mean()
    return daily_dispatch, daily_storage


if __name__ == "__main__":
    dispatch_rto = pd.read_csv(RESULTS_DIR / "rto_hourly_dispatch.csv", index_col=0, parse_dates=True)
    storage_rto = pd.read_csv(RESULTS_DIR / "rto_hourly_storage.csv", index_col=0, parse_dates=True).iloc[:, 0]
    dispatch_pf = pd.read_csv(RESULTS_DIR / "perfect_foresight_dispatch.csv", index_col=0, parse_dates=True)
    storage_pf = pd.read_csv(RESULTS_DIR / "perfect_foresight_storage.csv", index_col=0, parse_dates=True).iloc[:, 0]

    # --- Hourly view, zoomed into the cold snap itself ---
    ZOOM_START = "2019-01-23"
    ZOOM_HOURS = 48
    w_rto = dispatch_rto.loc[ZOOM_START:].iloc[:ZOOM_HOURS]
    w_pf = dispatch_pf.loc[ZOOM_START:].iloc[:ZOOM_HOURS]
    supply_rto = w_rto["gas_boiler"] + w_rto["heat_pump"] + w_rto["storage_discharge"]
    supply_pf = w_pf["gas_boiler"] + w_pf["heat_pump"] + w_pf["storage_discharge"]
    top = max(w_rto["heat_demand"].max(), w_pf["heat_demand"].max(), supply_rto.max(), supply_pf.max()) * 1.45
    max_charge = max(w_rto["storage_charge"].max(), w_pf["storage_charge"].max())
    ylim_dispatch_hourly = (-max_charge * 1.15 if max_charge > 0 else -1, top)
    ylim_storage = (0, CAP_STORAGE_MWH)

    plot_dispatch(
        dispatch_rto, storage_rto, "RTO (hourly) dispatch around cold snap", PLOT_DIR / "rto_coldsnap.png",
        ZOOM_START, ZOOM_HOURS, freq="h", ylim_dispatch=ylim_dispatch_hourly, ylim_storage=ylim_storage,
    )
    plot_dispatch(
        dispatch_pf, storage_pf, "Perfect foresight dispatch around cold snap", PLOT_DIR / "perfect_foresight_coldsnap.png",
        ZOOM_START, ZOOM_HOURS, freq="h", ylim_dispatch=ylim_dispatch_hourly, ylim_storage=ylim_storage,
    )

    # --- Daily-aggregated view, whole simulated period ---
    daily_dispatch_rto, daily_storage_rto = aggregate_daily(dispatch_rto, storage_rto)
    daily_dispatch_pf, daily_storage_pf = aggregate_daily(dispatch_pf, storage_pf)

    daily_supply_rto = daily_dispatch_rto["gas_boiler"] + daily_dispatch_rto["heat_pump"] + daily_dispatch_rto["storage_discharge"]
    daily_supply_pf = daily_dispatch_pf["gas_boiler"] + daily_dispatch_pf["heat_pump"] + daily_dispatch_pf["storage_discharge"]
    daily_top = max(
        daily_dispatch_rto["heat_demand"].max(), daily_dispatch_pf["heat_demand"].max(),
        daily_supply_rto.max(), daily_supply_pf.max(),
    ) * 1.45
    daily_max_charge = max(daily_dispatch_rto["storage_charge"].max(), daily_dispatch_pf["storage_charge"].max())
    ylim_dispatch_daily = (-daily_max_charge * 1.15 if daily_max_charge > 0 else -1, daily_top)

    plot_dispatch(
        daily_dispatch_rto, daily_storage_rto, "RTO (hourly) dispatch -- daily totals", PLOT_DIR / "rto_daily.png",
        zoom_start=None, freq="D", ylim_dispatch=ylim_dispatch_daily, ylim_storage=ylim_storage,
    )
    plot_dispatch(
        daily_dispatch_pf, daily_storage_pf, "Perfect foresight dispatch -- daily totals", PLOT_DIR / "perfect_foresight_daily.png",
        zoom_start=None, freq="D", ylim_dispatch=ylim_dispatch_daily, ylim_storage=ylim_storage,
    )

    print(f"Saved plots to {PLOT_DIR}/")
