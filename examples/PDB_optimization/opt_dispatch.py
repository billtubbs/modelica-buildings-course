"""Perfect-foresight vs. causal (receding-horizon) dispatch comparison.

This script takes a *fixed* gas boiler / heat pump / heat storage design
(capacities decided beforehand, e.g. by running the capacity-optimization
sweep in the companion script and reading off one of the resulting designs)
and compares two ways of operating that fixed hardware over the same year
of data:

    1. "Perfect foresight" dispatch: a single oemof.solph LP solved once
       over the whole year, with full knowledge of future heat demand, gas
       price and electricity spot price at every hour. This is the same
       kind of dispatch produced by the capacity-optimization script, but
       now with capacities pinned rather than optimized -- it is the
       best possible operation of this exact hardware.

    2. "Causal" (real-time / receding-horizon) dispatch: the system is
       re-optimized repeatedly over a short look-ahead window using only
       (a) the true, known values up to and including the current hour and
       (b) a simple persistence forecast for the rest of the look-ahead
       window (future values assumed equal to the current value). Only the
       first CONTROL_STEP hours of each window's plan are "implemented"
       (kept), the storage state at the end of that implemented block is
       carried forward as the initial condition, and the window then slides
       forward and re-solves -- i.e. a standard MPC-style rolling horizon.

Both cases use IDENTICAL fixed capacities, so the comparison isolates the
effect of information (perfect foresight vs. causal + persistence forecast)
on operating cost and CO2, with investment cost held constant and equal in
both cases.

Assumptions / things to check when running this:
    * FIX_CAP_* below must be filled in from a solved design (e.g. the
      lambda=20 full-system case from the capacity-sizing sweep script).
    * The persistence forecast is deliberately the simplest possible
      (flat: value(t+k) = value(t) for every k inside the look-ahead
      window). This is a strong assumption and is expected to make the
      causal case look considerably worse than perfect foresight,
      especially for the storage, which exists specifically to exploit
      future price/demand information it does not have here.
    * A causal operator can, in principle, run into an infeasible window
      (not enough capacity/storage to meet the *known, fixed* heat demand
      given the storage state it was left with) if earlier decisions -- made
      under a bad forecast -- turn out to have been the wrong call. This
      script raises a clear error if that happens rather than silently
      producing nonsense; see `solve_dispatch_window`.
    * The perfect-foresight case solves with `balanced=True` and the
      start-of-year storage level left free, exactly like the
      capacity-sizing run -- it does NOT assume any particular starting
      SOC. The resulting start-of-year SOC is then reused to seed the
      causal case's very first window, so both cases start from the same
      feasible condition instead of an arbitrary guess (an arbitrary guess,
      e.g. 50%, can make the very first hours of the year infeasible if the
      combined converter capacity is smaller than peak demand and relies on
      the storage being at the "right" level to cover the gap).

Input:
    ``<DATA_DIR>/input_data.csv`` -- semicolon-separated hourly time series
    with a parseable datetime index and (at least) the columns:
        * "heat demand"   heat load of the network [MW]
        * "gas price"     gas commodity price [EUR/MWh]
        * "el_spot_price" electricity spot price [EUR/MWh]
    (Any additional columns, e.g. an emission-factor column, are ignored.)

Outputs:
    ``<PLOT_DIR>/dispatch_comparison_<case>.png``
    ``<PLOT_DIR>/storage_soc_comparison.png``
        Dispatch stacks for each case and an overlaid storage
        state-of-charge comparison.
    ``<RESULTS_DIR>/dispatch_perfect_foresight.csv``
    ``<RESULTS_DIR>/dispatch_causal.csv``
        Hourly dispatch time series for each case.
    ``<RESULTS_DIR>/comparison_summary.csv``
        One-row-per-case summary of cost, CO2 and LCOH.

    A summary table is also printed to the console.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import oemof.solph as solph
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        print(
            "(tqdm not installed -- install with `pip install tqdm` for a "
            "progress bar; continuing without one.)"
        )
        return iterable

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")
RESULTS_DIR = Path("results")

for directory in (DATA_DIR, PLOT_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed system designs -- two solved capacity-optimization cases to choose
# from. lcoh/co2 here are the *design-stage* values from that sizing run
# (i.e. under perfect-foresight dispatch at whatever CO2 price produced this
# design) and are carried along only as a reference point to sanity check
# against this script's own perfect-foresight re-evaluation below -- they
# are not used in any calculation.
# ---------------------------------------------------------------------------
SOLUTIONS = {
    "solution_1": {
        # from "full λ=20" in the sizing sweep output
        "cap_gas_boiler_mw": 4.419,
        "cap_heat_pump_mw": 10.649,
        "cap_storage_mwh": 187.176,
        "design_lcoh_eur_per_mwh": 18.903,
        "design_co2_t": 3506.820,
    },
    "solution_2": {
        # from "full λ=0" in the sizing sweep output
        "cap_gas_boiler_mw": 10.234,
        "cap_heat_pump_mw": 5.985,
        "cap_storage_mwh": 98.592,
        "design_lcoh_eur_per_mwh": 18.626,
        "design_co2_t": 6859.743,
    },
}

SELECTED_SOLUTION = "solution_1"   # change to "solution_2" to evaluate that design instead

# Safety margin applied to all three fixed capacities. The exact LP-optimal
# capacities are only available to 3 decimal places (as printed by the
# sizing sweep), and this design has essentially zero slack in its annual
# storage energy balance by construction ("optimal capacity" means no more
# storage than exactly needed) -- so rounding the true value down by even a
# fraction of a percent can make the *whole year* infeasible, even though
# every individual hour is comfortably within capacity. This was confirmed
# with test_solve_dispatch_window.py: the local window around the first
# infeasible hour solves fine in isolation with a free start/end SOC, and
# the shortfall there is tiny (0.033 MW against ~16 MW demand -- a 0.2% gap
# at the single coldest, most storage-depleted hour of the year).
# This margin is a documented workaround for that precision limit, not a
# real hardware change. Set to 0.0 to test with the exact reported values.
# For an exact (no-margin) fix instead, export the sizing run's Investment
# results at full float precision (e.g. to a small JSON/CSV) and load them
# here directly, rather than hand-copying the printed, rounded figures.
CAPACITY_SAFETY_MARGIN = 0.002   # 0.2% headroom

_design = SOLUTIONS[SELECTED_SOLUTION]
CAP_GAS_BOILER_MW = _design["cap_gas_boiler_mw"] * (1 + CAPACITY_SAFETY_MARGIN)
CAP_HEAT_PUMP_MW = _design["cap_heat_pump_mw"] * (1 + CAPACITY_SAFETY_MARGIN)
CAP_STORAGE_MWH = _design["cap_storage_mwh"] * (1 + CAPACITY_SAFETY_MARGIN)

# CO2 price [EUR/tCO2] used inside the dispatch cost minimization (i.e. the
# carbon price the design above was actually optimized against). This
# affects the dispatch decision, not the physical hardware.
CO2_PRICE = 20

# Rolling-horizon settings for the causal case.
HORIZON = 48         # hours the causal solve "sees" ahead at each re-solve
CONTROL_STEP = 24    # hours actually implemented before re-solving
N_KNOWN = 1          # hours of *true* data available at each re-solve (=1: only "now")

# Storage initial condition: NOT hard-coded. The combined converter capacity
# is deliberately smaller than peak heat demand (the design relies on the
# storage covering the gap at the right moments), so an arbitrary starting
# SOC can make the very first hours of the year infeasible. Instead, the
# perfect-foresight run below determines its own optimal start-of-year SOC
# (via `balanced=True` with the initial level left free -- exactly what the
# capacity-sizing run did), and that value is then used to seed the causal
# case, so both cases start from the same *feasible, meaningful* condition.

# ---------------------------------------------------------------------------
# Technical / cost parameters (kept identical to the capacity-sizing script
# so results are on the same footing).
# ---------------------------------------------------------------------------
COP = 3.5
VAR_COST_HEAT_PUMP = 1.2

GAS_BOILER_EFFICIENCY = 0.95
VAR_COST_GAS_BOILER = 1.10

VAR_COST_STORAGE = 0.1
STORAGE_LOSS_RATE = 0.001

SPEC_INV_GAS_BOILER = 60000
SPEC_INV_HEAT_PUMP = 500000
SPEC_INV_STORAGE = 1060

CO2_GAS = 0.2   # tCO2/MWh gas
CO2_EL = 0.15   # tCO2/MWh electricity


def epc(invest_cost, i=0.05, n=20):
    af = (i * (1 + i) ** n) / ((1 + i) ** n - 1)
    return invest_cost * af


def LCOH(invest_cost, operation_cost, heat_produced, revenue=0, i=0.05, n=20):
    pvf = ((1 + i) ** n - 1) / (((1 + i) ** n) * i)
    return (invest_cost + pvf * (operation_cost - revenue)) / (
        pvf * heat_produced
    )


# ---------------------------------------------------------------------------
# Core dispatch solve for one window of data (used by both cases: the
# perfect-foresight case just calls this once with window == the whole year).
# ---------------------------------------------------------------------------
def solve_dispatch_window(
    window,
    cap_gas_boiler,
    cap_heat_pump,
    cap_storage,
    co2_price,
    max_heat_demand,
    initial_storage_level,
    balanced,
    allow_shortfall=False,
):
    """Solve a fixed-capacity dispatch LP over one window of data.

    ``window`` must have N+1 timestamps to produce N dispatch intervals
    (oemof convention with ``infer_last_interval=False``): the heat-demand
    ``fix`` profile uses ``window["heat demand"].iloc[:-1]`` and prices are
    likewise only "consumed" for the first N rows.

    Returns ``(dispatch, storage_content)``: ``dispatch`` is a DataFrame
    indexed by the first N timestamps of ``window`` (one row per interval);
    ``storage_content`` is a Series indexed by all N+1 timestamps of
    ``window`` (one value per boundary, in absolute MWh) -- so
    ``storage_content.iloc[0]`` is the state at the start of the window and
    ``storage_content.iloc[k]`` is the state after ``k`` intervals.

    ``allow_shortfall=True`` is an infeasibility-diagnosis mode: it adds an
    unlimited, heavily-penalized "emergency heat" source directly to the
    heat network so the LP always solves, and the returned ``dispatch``
    gets an extra ``emergency_shortfall`` column. Any hour where that
    column is nonzero is an hour the fixed hardware genuinely cannot cover
    from generation + storage alone -- i.e. the real cause of an
    infeasibility. Never use this mode for an actual result, only to find
    where/how large a shortfall is.
    """
    # Per-interval cost sequences need N values (one per dispatch interval),
    # matching the N-values-for-N+1-boundaries convention used everywhere
    # else here -- truncate the same way as the demand `fix` profile below.
    # (Leaving these at full window length, N+1, is what previously
    # triggered oemof's "Sequence longer than needed" FutureWarning; a
    # future oemof release turns that into a hard error.)
    gas_source_cost = (window["gas price"] + co2_price * CO2_GAS).iloc[:-1]
    electricity_source_cost = (
        window["el_spot_price"] + co2_price * CO2_EL
    ).iloc[:-1]

    es = solph.EnergySystem(timeindex=window.index, infer_last_interval=False)

    heat_bus = solph.Bus(label="heat network")
    gas_bus = solph.Bus(label="gas network")
    waste_heat_bus = solph.Bus(label="waste heat network")
    electricity_bus = solph.Bus(label="electricity network")
    es.add(heat_bus, gas_bus, waste_heat_bus, electricity_bus)

    gas_source = solph.components.Source(
        label="gas source",
        outputs={gas_bus: solph.flows.Flow(variable_costs=gas_source_cost)},
    )
    electricity_source = solph.components.Source(
        label="electricity source",
        outputs={
            electricity_bus: solph.flows.Flow(
                variable_costs=electricity_source_cost
            )
        },
    )
    waste_heat_source = solph.components.Source(
        label="waste heat source",
        outputs={waste_heat_bus: solph.flows.Flow()},
    )
    heat_sink = solph.components.Sink(
        label="heat sink",
        inputs={
            heat_bus: solph.flows.Flow(
                nominal_capacity=max_heat_demand,
                fix=(window["heat demand"] / max_heat_demand).iloc[:-1],
            )
        },
    )
    es.add(gas_source, electricity_source, waste_heat_source, heat_sink)

    if allow_shortfall:
        # Unlimited, heavily-penalized source -- lets the LP always solve so
        # the *pattern* of any remaining shortfall can be inspected.
        emergency_source = solph.components.Source(
            label="emergency heat",
            outputs={
                heat_bus: solph.flows.Flow(variable_costs=1e6)
            },
        )
        es.add(emergency_source)

    gas_boiler = solph.components.Converter(
        label="gas boiler",
        inputs={gas_bus: solph.flows.Flow()},
        outputs={
            heat_bus: solph.flows.Flow(
                nominal_capacity=cap_gas_boiler,
                variable_costs=VAR_COST_GAS_BOILER,
            )
        },
        conversion_factors={gas_bus: GAS_BOILER_EFFICIENCY},
    )

    heat_pump = solph.components.Converter(
        label="heat pump",
        inputs={
            electricity_bus: solph.flows.Flow(),
            waste_heat_bus: solph.flows.Flow(),
        },
        outputs={
            heat_bus: solph.flows.Flow(
                nominal_capacity=cap_heat_pump,
                variable_costs=VAR_COST_HEAT_PUMP,
            )
        },
        conversion_factors={
            electricity_bus: 1 / COP,
            waste_heat_bus: (COP - 1) / COP,
        },
    )

    # initial_storage_level=None leaves the starting SOC free (bounded 0-1
    # like any other timestep); combined with balanced=True this reproduces
    # exactly what the capacity-sizing run did -- the optimizer picks
    # whichever start/end SOC makes the whole horizon work best, rather than
    # a value imposed from outside that might not be feasible.
    storage_kwargs = dict(
        label="heat storage",
        nominal_capacity=cap_storage,
        inputs={
            heat_bus: solph.flows.Flow(
                variable_costs=VAR_COST_STORAGE,
                nominal_capacity=cap_storage / 24,
            )
        },
        outputs={
            heat_bus: solph.flows.Flow(
                variable_costs=VAR_COST_STORAGE,
                nominal_capacity=cap_storage / 24,
            )
        },
        balanced=balanced,
        loss_rate=STORAGE_LOSS_RATE,
    )
    if initial_storage_level is not None:
        storage_kwargs["initial_storage_level"] = initial_storage_level
    heat_storage = solph.components.GenericStorage(**storage_kwargs)

    es.add(gas_boiler, heat_pump, heat_storage)

    model = solph.Model(es)
    model.solve(solver="cbc", solve_kwargs={"tee": False})

    term_condition = str(
        model.solver_results["Solver"][0]["Termination condition"]
    )
    if term_condition != "optimal":
        if allow_shortfall:
            # Should not happen -- the emergency source should make this
            # always feasible. If it still fails, something else is wrong.
            raise RuntimeError(
                "Diagnostic (allow_shortfall=True) solve still did not "
                f"reach optimality (termination condition: {term_condition})."
            )
        raise RuntimeError(
            f"Dispatch window starting at {window.index[0]} was not solved "
            f"to optimality (termination condition: {term_condition}). "
            "For the causal case this typically means the fixed capacities "
            "cannot meet the known heat demand given the storage state "
            "handed over from the previous window -- i.e. an earlier "
            "decision, made under an imperfect forecast, has left the "
            "system unable to cope. Re-run with allow_shortfall=True on "
            "this same window to see exactly which hour(s) are short and "
            "by how much."
        )

    results = solph.processing.results(model)
    data_heat_bus = solph.views.node(results, "heat network")["sequences"]
    data_heat_storage = solph.views.node(results, "heat storage")["sequences"]

    idx = window.index[:-1]
    n = len(idx)

    def _flow(key):
        """Extract a flow column, truncated to the first `n` values.

        oemof's returned flow sequences can come back at length N+1 (one
        per time POINT, same convention as storage_content) rather than N
        (one per interval) -- truncating defensively here avoids a pandas
        length-mismatch error regardless of which convention this oemof
        version/model configuration actually uses.
        """
        return data_heat_bus[key].to_numpy()[:n]

    dispatch = pd.DataFrame(
        {
            "gas_boiler": _flow((("gas boiler", "heat network"), "flow")),
            "heat_pump": _flow((("heat pump", "heat network"), "flow")),
            "storage_discharge": _flow(
                (("heat storage", "heat network"), "flow")
            ),
            "storage_charge": _flow(
                (("heat network", "heat storage"), "flow")
            ),
            "heat_demand": _flow((("heat network", "heat sink"), "flow")),
        },
        index=idx,
    )
    if allow_shortfall:
        dispatch["emergency_shortfall"] = _flow(
            (("emergency heat", "heat network"), "flow")
        )

    # Sanity check the extraction itself: if the `[:n]` truncation above
    # grabbed the wrong end of an N+1-length sequence (or grabbed the wrong
    # sequence entirely), the heat balance below will be badly violated --
    # catch that immediately rather than silently returning misaligned
    # data.
    balance = (
        dispatch["gas_boiler"]
        + dispatch["heat_pump"]
        + dispatch["storage_discharge"]
        - dispatch["storage_charge"]
        - dispatch["heat_demand"]
    )
    if allow_shortfall:
        balance = balance + dispatch["emergency_shortfall"]
    max_imbalance = balance.abs().max()
    if max_imbalance > 1e-3:
        raise RuntimeError(
            f"Heat balance check failed after extracting dispatch results "
            f"(max imbalance {max_imbalance:.4f} MW at "
            f"{balance.abs().idxmax()}). The flow-sequence truncation "
            f"assumption in `_flow()` is likely misaligned for this oemof "
            f"version -- inspect `data_heat_bus` directly (its raw shape "
            f"and index) rather than trusting these dispatch values."
        )

    # storage_content is defined at every TIME POINT (boundary), not every
    # interval -- i.e. one value per entry of `window.index` (N+1 values for
    # N dispatch intervals): storage_content.iloc[0] is the true state at
    # the start of the window (matching `initial_storage_level`), and
    # storage_content.iloc[k] is the state after k intervals have run.
    # Keep the FULL series (do not truncate to N) so that indexing by
    # "number of intervals implemented" is unambiguous for the caller.
    soc_raw = data_heat_storage[
        (("heat storage", "None"), "storage_content")
    ].to_numpy()
    n_points = min(len(soc_raw), len(window.index))
    storage_content = pd.Series(
        soc_raw[:n_points], index=window.index[:n_points]
    )

    return dispatch, storage_content


def make_persistence_forecast(data, t0, horizon, n_known=1):
    """Build a forecast window: real data for the first ``n_known`` steps,
    then a flat persistence forecast (value held at the last known value)
    for the remaining ``horizon + 1 - n_known`` steps.

    Returns a DataFrame with ``horizon + 1`` rows (boundaries for
    ``horizon`` dispatch intervals), indexed with the *real* timestamps
    from ``data`` (only the values beyond ``n_known`` are forecast, not the
    timestamps).
    """
    end = t0 + horizon + 1
    idx = data.index[t0:end]
    known = data.iloc[t0 : t0 + n_known].copy()
    n_forecast = len(idx) - len(known)
    if n_forecast > 0:
        last_known_row = data.iloc[t0 + n_known - 1]
        forecast_rows = pd.DataFrame(
            [last_known_row.to_dict()] * n_forecast,
            index=idx[len(known):],
        )
        window = pd.concat([known, forecast_rows])
    else:
        window = known
    window.index = idx
    return window


def diagnose_infeasibility(window, cap_gas_boiler, cap_heat_pump, cap_storage, co2_price, max_heat_demand, initial_storage_level, balanced):
    """Re-solve a window with an unlimited penalized 'emergency heat' source
    so the LP always solves, then report which hours needed it -- i.e. the
    actual location(s) of an otherwise-opaque CBC infeasibility.
    """
    dispatch, _ = solve_dispatch_window(
        window,
        cap_gas_boiler,
        cap_heat_pump,
        cap_storage,
        co2_price,
        max_heat_demand,
        initial_storage_level,
        balanced,
        allow_shortfall=True,
    )
    shortfall = dispatch["emergency_shortfall"]
    short_mask = (shortfall > 1e-6).to_numpy()
    n_short = int(short_mask.sum())
    if n_short == 0:
        print(
            "Diagnostic solve found NO shortfall hours -- the infeasibility "
            "is not a simple capacity shortfall (check balanced-year energy "
            "accounting, or a units/sign issue elsewhere in the model)."
        )
        return dispatch

    first_pos = int(np.flatnonzero(short_mask)[0])
    first_ts = shortfall.index[first_pos]
    print(
        f"Diagnostic solve found {n_short} hour(s) with an unmet-demand "
        f"shortfall, totalling {shortfall.sum():.2f} MWh, peak "
        f"{shortfall.max():.3f} MW at {shortfall.idxmax()}."
    )
    print(
        f"FIRST shortfall: time step {first_pos} of {len(shortfall)} "
        f"(timestamp {first_ts}), shortfall {shortfall.iloc[first_pos]:.3f} MW."
    )
    context_lo = max(0, first_pos - 3)
    context_hi = min(len(dispatch), first_pos + 4)
    print(f"Dispatch around that time step (rows {context_lo}..{context_hi - 1}):")
    print(
        dispatch.iloc[context_lo:context_hi][
            ["gas_boiler", "heat_pump", "storage_discharge", "storage_charge", "heat_demand", "emergency_shortfall"]
        ].to_string()
    )
    return dispatch


def run_perfect_foresight(data, max_heat_demand):
    """Single solve over the whole year with full knowledge of the future.

    The starting SOC is left free (``balanced=True``, no
    ``initial_storage_level``) so the optimizer picks whatever start/end
    state makes the whole year work -- exactly as the capacity-sizing run
    did with these same fixed capacities. Returns the dispatch, the full
    boundary-indexed storage_content series, and the resulting start-of-year
    SOC as a fraction of capacity (for seeding the causal case).
    """
    print("Solving perfect-foresight dispatch (single solve, full year)...")
    try:
        dispatch, storage_content = solve_dispatch_window(
            data,
            CAP_GAS_BOILER_MW,
            CAP_HEAT_PUMP_MW,
            CAP_STORAGE_MWH,
            CO2_PRICE,
            max_heat_demand,
            initial_storage_level=None,
            balanced=True,
        )
    except RuntimeError:
        print("Infeasible -- re-solving with an emergency heat source to locate the shortfall...")
        diagnose_infeasibility(
            data,
            CAP_GAS_BOILER_MW,
            CAP_HEAT_PUMP_MW,
            CAP_STORAGE_MWH,
            CO2_PRICE,
            max_heat_demand,
            initial_storage_level=None,
            balanced=True,
        )
        raise
    initial_level = float(storage_content.iloc[0] / CAP_STORAGE_MWH)
    print(
        f"  optimal start-of-year storage level: {initial_level:.3f} "
        f"({storage_content.iloc[0]:.1f} MWh)"
    )
    return dispatch, storage_content, initial_level


def run_causal_dispatch(
    data, max_heat_demand, initial_storage_level, horizon, control_step
):
    """Receding-horizon dispatch using only current + persisted-forecast data."""
    n_total = len(data) - 1  # number of dispatch intervals in the full year
    storage_level = initial_storage_level

    implemented_dispatch = []
    implemented_storage = []

    t0 = 0
    n_windows = 0
    pbar = tqdm(total=n_total, desc="Causal dispatch", unit="hr")
    while t0 < n_total:
        this_horizon = min(horizon, n_total - t0)
        this_control_step = min(control_step, this_horizon)

        window = make_persistence_forecast(
            data, t0, this_horizon, n_known=N_KNOWN
        )
        try:
            dispatch, storage_content = solve_dispatch_window(
                window,
                CAP_GAS_BOILER_MW,
                CAP_HEAT_PUMP_MW,
                CAP_STORAGE_MWH,
                CO2_PRICE,
                max_heat_demand,
                storage_level,
                balanced=False,
            )
        except RuntimeError:
            pbar.close()
            print(
                f"Causal dispatch infeasible at re-solve #{n_windows} "
                f"(hours {t0}..{t0 + this_horizon - 1} of {n_total}, "
                f"window starts {window.index[0]}, storage level handed in "
                f"was {storage_level:.4f} = "
                f"{storage_level * CAP_STORAGE_MWH:.2f} MWh)."
            )
            print("Re-solving this window with an emergency heat source to locate the shortfall...")
            diagnose_infeasibility(
                window,
                CAP_GAS_BOILER_MW,
                CAP_HEAT_PUMP_MW,
                CAP_STORAGE_MWH,
                CO2_PRICE,
                max_heat_demand,
                storage_level,
                balanced=False,
            )
            raise

        implemented_dispatch.append(dispatch.iloc[:this_control_step])
        # storage_content.iloc[0] is the window's start (== previous window's
        # hand-off, already recorded); keep it only for the very first
        # window so the stitched series isn't full of duplicate boundaries.
        if n_windows == 0:
            implemented_storage.append(storage_content.iloc[: this_control_step + 1])
        else:
            implemented_storage.append(
                storage_content.iloc[1 : this_control_step + 1]
            )

        # State AFTER `this_control_step` intervals have been implemented --
        # this is the correct hand-off point (not this_control_step - 1).
        storage_level = float(
            storage_content.iloc[this_control_step] / CAP_STORAGE_MWH
        )

        t0 += this_control_step
        n_windows += 1
        pbar.update(this_control_step)
    pbar.close()

    print(f"Causal dispatch solved in {n_windows} re-solves.")
    full_dispatch = pd.concat(implemented_dispatch)
    full_storage = pd.concat(implemented_storage)
    return full_dispatch, full_storage


def evaluate_dispatch(dispatch, price_data):
    """Compute operation cost, CO2 and LCOH for an (already implemented)
    dispatch time series, evaluated against the *real* (not forecast)
    prices -- i.e. what this dispatch would actually have cost/emitted in
    the real world, however it was decided.
    """
    gas_price = price_data["gas price"].reindex(dispatch.index)
    el_price = price_data["el_spot_price"].reindex(dispatch.index)

    # NOTE: given how `conversion_factors={gas_bus: GAS_BOILER_EFFICIENCY}` is
    # specified in the model (factor attached to the INPUT bus, output bus
    # defaults to 1), oemof's Converter equation actually gives
    # gas_consumed = heat_output * GAS_BOILER_EFFICIENCY (not divided). This
    # was verified against the original sizing script's own printed "gas
    # boiler only" case: heat_produced=66485.8 MWh * 0.95 * co2_gas(0.2)
    # = 12632.3 tCO2, which matches its printed CO2 figure exactly. It also
    # means this model implies gas-to-heat conversion >100% (1 MWh gas ->
    # 1/0.95 MWh heat) -- physically backwards for "boiler efficiency", and
    # worth revisiting in the original sizing script if that wasn't
    # intended. This evaluation function must match the model as actually
    # built, so it uses the same (as-built) relationship here.
    gas_consumed = dispatch["gas_boiler"] * GAS_BOILER_EFFICIENCY
    electricity_consumed = dispatch["heat_pump"] / COP

    operation_cost = (
        VAR_COST_GAS_BOILER * dispatch["gas_boiler"]
        + gas_price * gas_consumed
        + VAR_COST_HEAT_PUMP * dispatch["heat_pump"]
        + el_price * electricity_consumed
        + VAR_COST_STORAGE * dispatch["storage_charge"]
        + VAR_COST_STORAGE * dispatch["storage_discharge"]
    ).sum()

    total_co2 = (CO2_GAS * gas_consumed + CO2_EL * electricity_consumed).sum()

    invest_cost = (
        SPEC_INV_GAS_BOILER * CAP_GAS_BOILER_MW
        + SPEC_INV_HEAT_PUMP * CAP_HEAT_PUMP_MW
        + SPEC_INV_STORAGE * CAP_STORAGE_MWH
    )
    heat_produced = dispatch["heat_demand"].sum()
    lcoh = LCOH(invest_cost, operation_cost, heat_produced)

    return {
        "operation_cost_eur": operation_cost,
        "total_co2_t": total_co2,
        "lcoh_eur_per_mwh": lcoh,
        "heat_produced_mwh": heat_produced,
        "storage_cycles": dispatch["storage_discharge"].sum()
        / CAP_STORAGE_MWH,
    }


def plot_dispatch(dispatch, case_label, slug):
    unit_colors = {
        "heat pump": "#B54036",
        "gas boiler": "#EC6707",
        "heat storage (discharge)": "#BFBFBF",
        "heat storage (charge)": "#696969",
    }
    fig, ax = plt.subplots(figsize=[10, 6])
    bottom = 0
    for col, label in [
        ("heat_pump", "heat pump"),
        ("gas_boiler", "gas boiler"),
        ("storage_discharge", "heat storage (discharge)"),
    ]:
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


if __name__ == "__main__":
    print(
        f"Evaluating dispatch for '{SELECTED_SOLUTION}' "
        f"(+{CAPACITY_SAFETY_MARGIN * 100:.1f}% capacity safety margin): "
        f"gas boiler {CAP_GAS_BOILER_MW:.4f} MW / heat pump "
        f"{CAP_HEAT_PUMP_MW:.4f} MW / storage {CAP_STORAGE_MWH:.4f} MWh "
        f"(design-stage reference: LCOH {_design['design_lcoh_eur_per_mwh']} "
        f"EUR/MWh, CO2 {_design['design_co2_t']} t)"
    )

    data = pd.read_csv(
        DATA_DIR / "input_data.csv", sep=";", index_col=0, parse_dates=True
    )
    max_heat_demand = data["heat demand"].max()

    dispatch_pf, storage_pf, initial_storage_level = run_perfect_foresight(
        data, max_heat_demand
    )
    dispatch_causal, storage_causal = run_causal_dispatch(
        data, max_heat_demand, initial_storage_level, HORIZON, CONTROL_STEP
    )

    metrics_pf = evaluate_dispatch(dispatch_pf, data)
    metrics_causal = evaluate_dispatch(dispatch_causal, data)

    dispatch_pf.to_csv(RESULTS_DIR / "dispatch_perfect_foresight.csv")
    dispatch_causal.to_csv(RESULTS_DIR / "dispatch_causal.csv")

    plot_dispatch(dispatch_pf, "perfect foresight", "perfect_foresight")
    plot_dispatch(dispatch_causal, "causal (persistence forecast)", "causal")
    plot_storage_comparison(storage_pf, storage_causal)

    summary = pd.DataFrame(
        {"perfect_foresight": metrics_pf, "causal": metrics_causal}
    ).T
    summary["design_lcoh_eur_per_mwh"] = _design["design_lcoh_eur_per_mwh"]
    summary["design_co2_t"] = _design["design_co2_t"]
    summary["lcoh_gap_pct"] = (
        100
        * (summary["lcoh_eur_per_mwh"] - metrics_pf["lcoh_eur_per_mwh"])
        / metrics_pf["lcoh_eur_per_mwh"]
    )
    summary.to_csv(RESULTS_DIR / "comparison_summary.csv")

    print("\nComparison summary:")
    print(summary.round(3).to_string())

    plt.show()
