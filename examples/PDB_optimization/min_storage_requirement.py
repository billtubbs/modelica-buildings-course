"""Minimum required thermal storage level, by hour, for a fixed design.

For each hour of the year, this computes the MINIMUM thermal storage level
(MWh) that must be present at the START of that hour to guarantee the
fixed-capacity system (gas boiler + heat pump + storage) can meet ALL
remaining heat demand for the rest of the year -- assuming the best
possible use of firm generation and storage capacity from that point
forward. This is a pure feasibility/reliability question, independent of
cost or price: "how much do I need in reserve, right now, to not run out
later?" -- not "what's the cheapest way to run the system?".

Method: a single backward pass per design (no LP/MILP needed). Starting
from the last hour of the year and working backward:
    * If demand this hour is <= combined firm capacity (gas boiler + heat
      pump), no storage discharge is needed -- and any leftover firm
      capacity can recharge the storage (up to its charge-power limit),
      reducing how much must already be stored before this hour.
    * If demand exceeds firm capacity, the storage must discharge the
      difference (capped by its discharge-power limit -- if the gap is
      larger than that, the design is infeasible at this hour regardless
      of storage level, and this is flagged separately).
    * Whatever level is required at the START of the NEXT hour becomes
      part of what must be achieved by the END of this hour, working
      backward through the storage's round-trip loss.

Prints a table of the highest such requirement (the single tightest
moment of the year) and when it occurs, for both SOLUTIONS defined below
(kept in sync with opt_dispatch.py's SOLUTIONS -- update both if the
design changes).

Input: <DATA_DIR>/input_data.csv (same file used by opt_dispatch.py).
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

# Storage round-trip parameters, matching opt_dispatch.py.
STORAGE_LOSS_RATE = 0.001  # fractional self-discharge per hour

# The two solved capacity-sizing designs (see opt_dispatch.py's SOLUTIONS
# for the same values with more context/provenance).
SOLUTIONS = {
    "solution_1": {
        "cap_gas_boiler_mw": 4.419,
        "cap_heat_pump_mw": 10.649,
        "cap_storage_mwh": 187.176,
    },
    "solution_2": {
        "cap_gas_boiler_mw": 10.234,
        "cap_heat_pump_mw": 5.985,
        "cap_storage_mwh": 98.592,
    },
}


def minimum_required_storage(
    demand, cap_gas_boiler, cap_heat_pump, cap_storage, loss_rate=STORAGE_LOSS_RATE
):
    """Backward pass: minimum storage level (MWh) required at the START of
    each hour in `demand` to meet all remaining demand through the end of
    the series, for a fixed-capacity design.

    Returns (min_required, infeasible_hours): `min_required` is a Series
    indexed like `demand`; `infeasible_hours` lists timestamps where
    demand exceeds firm capacity + max storage discharge power (infeasible
    regardless of storage level) or where the required level would exceed
    the storage's own energy capacity (infeasible regardless of starting
    level).
    """
    firm_capacity_mw = cap_gas_boiler + cap_heat_pump
    # Charge/discharge power limit, matching opt_dispatch.py's convention
    # (storage flow capacity = energy capacity / 24).
    power_limit_mw = cap_storage / 24

    n = len(demand)
    min_required = pd.Series(index=demand.index, dtype=float)
    infeasible_hours = []

    required_next = 0.0  # required level at the START of the next hour
    for k in range(n - 1, -1, -1):
        d = demand.iloc[k]
        if d <= firm_capacity_mw:
            # Firm capacity covers demand outright; any leftover can
            # recharge the storage, up to its charge-power limit.
            available_charge = min(power_limit_mw, firm_capacity_mw - d)
            s_min = max(0.0, (required_next - available_charge) / (1 - loss_rate))
        else:
            discharge_needed = d - firm_capacity_mw
            if discharge_needed > power_limit_mw + 1e-9:
                infeasible_hours.append(demand.index[k])
                discharge_needed = power_limit_mw  # cap for the recursion to continue
            s_min = (required_next + discharge_needed) / (1 - loss_rate)

        if s_min > cap_storage + 1e-9:
            infeasible_hours.append(demand.index[k])

        min_required.iloc[k] = s_min
        required_next = s_min

    return min_required, infeasible_hours


def main():
    data = pd.read_csv(
        DATA_DIR / "input_data.csv", sep=";", index_col=0, parse_dates=True
    )
    # Match opt_dispatch.py's convention: N demand values for N intervals
    # (the CSV has N+1 boundary timestamps).
    demand = data["heat demand"].iloc[:-1]

    rows = []
    for name, sol in SOLUTIONS.items():
        min_required, infeasible = minimum_required_storage(
            demand,
            sol["cap_gas_boiler_mw"],
            sol["cap_heat_pump_mw"],
            sol["cap_storage_mwh"],
        )
        peak_mwh = min_required.max()
        peak_time = min_required.idxmax()
        rows.append(
            {
                "solution": name,
                "cap_gas_boiler_mw": sol["cap_gas_boiler_mw"],
                "cap_heat_pump_mw": sol["cap_heat_pump_mw"],
                "cap_storage_mwh": sol["cap_storage_mwh"],
                "highest_min_storage_mwh": peak_mwh,
                "highest_min_storage_pct": 100 * peak_mwh / sol["cap_storage_mwh"],
                "datetime_needed": peak_time,
                "infeasible_hours": len(set(infeasible)),
            }
        )

    table = pd.DataFrame(rows).set_index("solution")
    pd.set_option("display.width", 160)
    numeric_cols = table.columns.drop("datetime_needed")
    table[numeric_cols] = table[numeric_cols].round(3)
    print(table.to_string())


if __name__ == "__main__":
    main()
