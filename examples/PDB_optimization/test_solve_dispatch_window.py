"""Standalone reproduction / diagnostic test for solve_dispatch_window.

Run directly:
    python test_solve_dispatch_window.py

Or with pytest:
    pytest test_solve_dispatch_window.py -s

What this does: runs the allow_shortfall diagnostic over the full year
(using whatever SOLUTIONS[SELECTED_SOLUTION] is currently configured in
opt_dispatch.py) to find the first hour with an unmet-demand shortfall,
then extracts a small window of real data around just that hour and
re-solves *only that window* in isolation, with the start/end storage
level left completely free (balanced=False, initial_storage_level=None --
i.e. as unconstrained as possible).

This isolation is the key diagnostic:
    * If the isolated window SOLVES -- the infeasibility is a whole-year,
      cumulative effect (most likely `balanced=True` forcing a specific
      start=end SOC across the full 8759 hours that these exact fixed
      capacities cannot actually sustain over a full year, even though
      they can clearly cover any local hour by itself).
    * If the isolated window is STILL infeasible -- it's a genuine local
      capacity shortfall: at this specific hour, gas boiler + heat pump +
      max storage discharge power together cannot cover heat demand,
      regardless of what the storage's state of charge is. That points to
      an error in the capacities themselves (e.g. rounding, or a units
      mismatch) or a bug in how capacities are being fixed in
      solve_dispatch_window, rather than a whole-year energy-balance issue.

Editing MARGIN_HOURS below changes how much context is kept around the
first shortfall hour.
"""

import numpy as np
import pandas as pd

import opt_dispatch as od

MARGIN_HOURS = 48  # hours of context kept on each side of the first shortfall


def find_first_shortfall_position(data, max_heat_demand):
    """Run the allow_shortfall diagnostic over the full year and return the
    0-based hour index of the first unmet-demand hour, or None if the full
    year is actually feasible under this diagnostic.
    """
    dispatch, _ = od.solve_dispatch_window(
        data,
        od.CAP_GAS_BOILER_MW,
        od.CAP_HEAT_PUMP_MW,
        od.CAP_STORAGE_MWH,
        od.CO2_PRICE,
        max_heat_demand,
        initial_storage_level=None,
        balanced=True,
        allow_shortfall=True,
    )
    shortfall = dispatch["emergency_shortfall"]
    short_mask = (shortfall > 1e-6).to_numpy()
    if not short_mask.any():
        return None, dispatch
    return int(np.flatnonzero(short_mask)[0]), dispatch


def test_first_shortfall_window_isolated():
    data = pd.read_csv(
        od.DATA_DIR / "input_data.csv", sep=";", index_col=0, parse_dates=True
    )
    max_heat_demand = data["heat demand"].max()

    print(
        f"Using capacities: gas boiler {od.CAP_GAS_BOILER_MW} MW / "
        f"heat pump {od.CAP_HEAT_PUMP_MW} MW / storage "
        f"{od.CAP_STORAGE_MWH} MWh (SOLUTIONS['{od.SELECTED_SOLUTION}'])"
    )

    first_pos, full_dispatch = find_first_shortfall_position(data, max_heat_demand)
    assert first_pos is not None, (
        "Full year is actually feasible under allow_shortfall=True (no "
        "shortfall hours found) -- the infeasibility must come from "
        "something other than a capacity/energy shortfall. Re-check the "
        "balanced=True year-end energy accounting, or a units/sign issue "
        "elsewhere in the model, rather than debugging capacities."
    )
    first_ts = data.index[first_pos]
    print(f"First shortfall at hour index {first_pos} ({first_ts})")
    print(
        full_dispatch.iloc[max(0, first_pos - 3) : first_pos + 4][
            [
                "gas_boiler",
                "heat_pump",
                "storage_discharge",
                "storage_charge",
                "heat_demand",
                "emergency_shortfall",
            ]
        ].to_string()
    )

    lo = max(0, first_pos - MARGIN_HOURS)
    hi = min(len(data) - 1, first_pos + MARGIN_HOURS)
    window = data.iloc[lo : hi + 1]  # +1 boundary row for N intervals in [lo, hi]
    print(
        f"\nIsolated window: hours {lo}..{hi} "
        f"({window.index[0]} to {window.index[-1]})"
    )

    try:
        od.solve_dispatch_window(
            window,
            od.CAP_GAS_BOILER_MW,
            od.CAP_HEAT_PUMP_MW,
            od.CAP_STORAGE_MWH,
            od.CO2_PRICE,
            max_heat_demand,
            initial_storage_level=None,
            balanced=False,
        )
        print(
            "\nResult: isolated window SOLVED (feasible) with a free "
            "start/end SOC.\n"
            "=> Likely a WHOLE-YEAR cumulative effect (e.g. balanced=True "
            "forcing a start=end SOC these capacities can't sustain over "
            "the full year), not a local shortfall at this hour."
        )
    except RuntimeError:
        print(
            "\nResult: isolated window is STILL infeasible even with a "
            "free start/end SOC.\n"
            "=> Likely a genuine LOCAL capacity shortfall at this hour: "
            "gas boiler + heat pump + max storage discharge power cannot "
            "cover heat demand here regardless of storage state. Check the "
            "capacity values themselves (rounding vs. the sizing run's "
            "true precision) and/or how nominal_capacity is being set in "
            "solve_dispatch_window."
        )
        raise


if __name__ == "__main__":
    test_first_shortfall_window_isolated()
