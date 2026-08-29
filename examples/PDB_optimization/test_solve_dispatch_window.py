"""Standalone reproduction / diagnostic test for solve_dispatch_window.

Run directly:
    python test_solve_dispatch_window.py

Or with pytest:
    pytest test_solve_dispatch_window.py -s

What this does: solves the full year (using whatever
SOLUTIONS[SELECTED_SOLUTION] is currently configured in opt_dispatch.py) to
find the first hour with unserved heat (opt_dispatch.py always connects an
unlimited, heavily-penalized "unserved heat" source, so this never raises --
see VOLL_COST_EUR_PER_MWH), then extracts a small window of real data around
just that hour and re-solves *only that window* in isolation, with the
start/end storage level left completely free (balanced=False,
initial_storage_level=None -- i.e. as unconstrained as possible).

Comparing unserved heat in the isolated window against the full year is the
key diagnostic:
    * If the isolated window has ~0 unserved heat -- the full-year result
      is a whole-year, cumulative effect (most likely `balanced=True`
      forcing a specific start=end SOC across the full 8759 hours that
      these exact fixed capacities cannot actually sustain over a full
      year, even though they can clearly cover any local hour by itself).
    * If the isolated window ALSO shows unserved heat -- it's a genuine
      local capacity shortfall: at this specific hour, gas boiler + heat
      pump + max storage discharge power together cannot cover heat
      demand, regardless of what the storage's state of charge is. That
      points to an error in the capacities themselves (e.g. rounding, or a
      units mismatch) or a bug in how capacities are being fixed in
      solve_dispatch_window, rather than a whole-year energy-balance issue.

Editing MARGIN_HOURS below changes how much context is kept around the
first shortfall hour.
"""

import numpy as np
import opt_dispatch as od
import pandas as pd


MARGIN_HOURS = 48  # hours of context kept on each side of the first shortfall


def find_first_unserved_position(dispatch):
    """Return the 0-based row position of the first unserved-heat hour in
    an already-solved dispatch DataFrame, or None if there is none.
    """
    short_mask = (dispatch["unserved_heat"] > 1e-6).to_numpy()
    if not short_mask.any():
        return None
    return int(np.flatnonzero(short_mask)[0])


def test_first_unserved_window_isolated():
    data = pd.read_csv(
        od.DATA_DIR / "input_data.csv", sep=";", index_col=0, parse_dates=True
    )
    max_heat_demand = data["heat demand"].max()

    print(
        f"Using capacities: gas boiler {od.CAP_GAS_BOILER_MW} MW / "
        f"heat pump {od.CAP_HEAT_PUMP_MW} MW / storage "
        f"{od.CAP_STORAGE_MWH} MWh (SOLUTIONS['{od.SELECTED_SOLUTION}'])"
    )

    full_dispatch, _ = od.solve_dispatch_window(
        data,
        od.CAP_GAS_BOILER_MW,
        od.CAP_HEAT_PUMP_MW,
        od.CAP_STORAGE_MWH,
        od.CO2_PRICE,
        max_heat_demand,
        initial_storage_level=None,
        balanced=True,
    )

    first_pos = find_first_unserved_position(full_dispatch)
    assert first_pos is not None, (
        "Full year solved with NO unserved heat -- the fixed capacities "
        "are sufficient for the whole year under perfect foresight. If "
        "you were expecting a shortfall, check CAPACITY_SAFETY_MARGIN and "
        "the SELECTED_SOLUTION values in opt_dispatch.py."
    )
    first_ts = data.index[first_pos]
    print(f"First unserved-heat hour at index {first_pos} ({first_ts})")
    print(
        full_dispatch.iloc[max(0, first_pos - 3) : first_pos + 4][
            [
                "gas_boiler",
                "heat_pump",
                "storage_discharge",
                "storage_charge",
                "heat_demand",
                "unserved_heat",
            ]
        ].to_string()
    )

    lo = max(0, first_pos - MARGIN_HOURS)
    hi = min(len(data) - 1, first_pos + MARGIN_HOURS)
    window = data.iloc[
        lo : hi + 1
    ]  # +1 boundary row for N intervals in [lo, hi]
    print(
        f"\nIsolated window: hours {lo}..{hi} "
        f"({window.index[0]} to {window.index[-1]})"
    )

    isolated_dispatch, _ = od.solve_dispatch_window(
        window,
        od.CAP_GAS_BOILER_MW,
        od.CAP_HEAT_PUMP_MW,
        od.CAP_STORAGE_MWH,
        od.CO2_PRICE,
        max_heat_demand,
        initial_storage_level=None,
        balanced=False,
    )
    isolated_unserved = isolated_dispatch["unserved_heat"].sum()

    if isolated_unserved < 1e-6:
        print(
            f"\nResult: isolated window has ~0 unserved heat "
            f"({isolated_unserved:.4f} MWh) with a free start/end SOC.\n"
            "=> Likely a WHOLE-YEAR cumulative effect (e.g. balanced=True "
            "forcing a start=end SOC these capacities can't sustain over "
            "the full year), not a local shortfall at this hour. Consider "
            "increasing CAPACITY_SAFETY_MARGIN slightly."
        )
    else:
        print(
            f"\nResult: isolated window STILL shows unserved heat "
            f"({isolated_unserved:.4f} MWh) even with a free start/end "
            "SOC.\n"
            "=> Likely a genuine LOCAL capacity shortfall at this hour: "
            "gas boiler + heat pump + max storage discharge power cannot "
            "cover heat demand here regardless of storage state. Check the "
            "capacity values themselves (rounding vs. the sizing run's "
            "true precision) and/or how nominal_capacity is being set in "
            "solve_dispatch_window."
        )
    assert isolated_unserved < 1e-6, (
        "Isolated window still has unserved heat even with a free "
        "start/end SOC -- see printed diagnosis above."
    )


if __name__ == "__main__":
    test_first_unserved_window_isolated()
