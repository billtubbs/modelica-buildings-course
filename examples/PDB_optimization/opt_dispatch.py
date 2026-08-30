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
    * The forecast for the causal case is a REPEATING PROFILE: the pattern
      of the most-recently-known block of hours (n_known, ideally a full
      day) is tiled forward to fill the rest of the look-ahead window,
      rather than a single flat value. This still only ever uses real,
      already-observed data (see make_persistence_forecast), but captures
      the diurnal demand cycle -- a flat forecast anchored to a single
      hour (e.g. a re-solve at midnight) badly underestimates a later
      daily peak, which was measured here to make a storage reserve
      target based on it look unnecessary right up until an actual
      shortfall. This is still a simplifying assumption (it assumes
      "tomorrow looks like the pattern I just observed"), and is expected
      to make the causal case look considerably worse than perfect
      foresight for genuinely unanticipated multi-day events (e.g. a cold
      snap that gets colder each day), since the storage exists
      specifically to exploit future information it does not have here.
    * A causal operator can, in principle, run into a window where the
      fixed hardware genuinely cannot meet demand (e.g. the storage was
      left empty by an earlier decision made under a bad forecast, right
      when a cold snap needs it). Rather than making the model infeasible,
      an unlimited "unserved heat" source is always connected at a heavy
      penalty cost (VOLL_COST_EUR_PER_MWH) -- see that constant and
      `report_unserved_heat`. Any resulting unserved heat is a genuine,
      honestly-reported part of the causal case's result (see
      `unserved_heat_mwh` / `unserved_heat_hours` in the summary), not a
      bug to eliminate. The perfect-foresight case should show ~0 unserved
      heat; if it doesn't, that's worth investigating (e.g. via
      test_solve_dispatch_window.py), since it would mean the capacities
      are undersized even under ideal information.
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

Outputs (computation only -- see opt_dispatch_plot.py for plots, which
reads these CSVs directly rather than re-solving anything):
    ``<RESULTS_DIR>/dispatch_perfect_foresight.csv``
    ``<RESULTS_DIR>/dispatch_causal.csv``
        Hourly dispatch time series for each case.
    ``<RESULTS_DIR>/storage_perfect_foresight.csv``
    ``<RESULTS_DIR>/storage_causal.csv``
        Boundary-indexed storage-content (state of charge) series for each
        case (see solve_dispatch_window's docstring for the boundary vs.
        interval indexing distinction).
    ``<RESULTS_DIR>/comparison_summary.csv``
        One-row-per-case summary of cost, CO2 and LCOH.

    A summary table is also printed to the console.
"""

# Changelog (was a TODO list; all items addressed):
#    1. DONE -- forecast now uses repeating 24-hour profiles, not flat
#       persistence (see make_persistence_forecast).
#    2. DONE -- a minimum end-of-horizon (well, end-of-CONTROL_STEP)
#       storage target now incentivizes planning ahead (see
#       terminal_reserve_target_mwh and BELIEF_HORIZON_EXTRA_HOURS).
#    3. TESTED, NOT ADOPTED -- lengthening HORIZON to 72 or 96 hours gave
#       IDENTICAL results to 48 (see HORIZON's comment for why); left at
#       48 since a longer horizon has a real compute cost for no benefit
#       under this forecast method.
#    4. ALREADY DONE -- both cases start from the same initial storage
#       amount (perfect foresight's own optimal start-of-year SOC, reused
#       to seed the causal case; see run_perfect_foresight/__main__).
#    5. DONE -- the hard min_storage_level bound and its clamping logic
#       were removed (measured to make things worse; superseded by the
#       soft terminal reserve target from item 2).
#    6. CONFIRMED -- all cost/technical parameters (COP, efficiencies,
#       specific investment costs, CO2 factors, storage loss rate, and the
#       epc/LCOH discount-rate defaults) match the capacity-sizing script.

from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as po
from oemof import solph
from tqdm import tqdm

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

for directory in (DATA_DIR, RESULTS_DIR):
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

SELECTED_SOLUTION = (
    "solution_1"  # change to "solution_2" to evaluate that design instead
)

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
CAPACITY_SAFETY_MARGIN = 0.002  # 0.2% headroom

_design = SOLUTIONS[SELECTED_SOLUTION]
CAP_GAS_BOILER_MW = _design["cap_gas_boiler_mw"] * (1 + CAPACITY_SAFETY_MARGIN)
CAP_HEAT_PUMP_MW = _design["cap_heat_pump_mw"] * (1 + CAPACITY_SAFETY_MARGIN)
CAP_STORAGE_MWH = _design["cap_storage_mwh"] * (1 + CAPACITY_SAFETY_MARGIN)

# CO2 price [EUR/tCO2] used inside the dispatch cost minimization (i.e. the
# carbon price the design above was actually optimized against). This
# affects the dispatch decision, not the physical hardware.
CO2_PRICE = 20

# Rolling-horizon settings for the causal case.
HORIZON = 48  # hours the causal solve "sees" ahead at each re-solve
# Tested lengthening this to 72 and 96 hours -- gave IDENTICAL results
# (129.410 MWh unserved, 46 hours, both times) to HORIZON=48. This makes
# sense given how the forecast works: make_persistence_forecast tiles the
# SAME single observed day forward repeatedly, so a longer horizon just
# shows the LP more repetitions of that identical day, not any new
# information -- and the terminal reserve target is keyed to CONTROL_STEP,
# not HORIZON, so it's unaffected too. A longer horizon would only help if
# paired with a forecast that actually carries new information further
# out (e.g. a real multi-day weather forecast, or extrapolating a trend
# across several recently-observed days rather than tiling just the last
# one) -- with the current forecast, keep this at the shorter/cheaper
# value since there is no benefit to raising it.
CONTROL_STEP = 24  # hours actually implemented before re-solving
# N_KNOWN MUST equal CONTROL_STEP: every hour that gets implemented (kept
# as "what actually happened") must be based on data that was genuinely
# real/known at the time it was committed to, not a persistence forecast.
# Setting N_KNOWN < CONTROL_STEP would silently record forecast values as
# if they were the real outcome for the un-implemented-yet-kept hours --
# this was an actual bug here (N_KNOWN=1, CONTROL_STEP=24) caught by
# comparing the stitched causal dispatch's total heat_demand against the
# real annual total: they must always match exactly, and they didn't
# (58,129 MWh vs the true 66,486 MWh) until this was fixed.
N_KNOWN = CONTROL_STEP

# Soft storage RESERVE for the CAUSAL case only (perfect foresight is left
# unconstrained -- it already achieves 0 unserved heat). Intent: stop the
# causal controller's forecast-driven dispatch from fully draining the
# tank within its own planning horizon, leaving nothing in reserve for
# what happens right after the horizon ends -- the classic finite-horizon
# MPC "no terminal cost" failure mode.
#
# This is a genuine MPC TERMINAL target, computed fresh at every re-solve
# from the worst hour of the same real data the controller has just
# observed, held flat (see terminal_reserve_target_mwh): "if conditions
# this severe continued a bit further than my visible horizon, how much
# would I need left in the tank by the end of that horizon?" This is
# causal (uses only currently-observable information, never the real
# future) and self-adapting (higher target during an observed cold snap,
# near-zero during mild weather) rather than one fixed number applied
# uniformly all year. Deliberately more conservative than the repeating
# profile used for actual dispatch decisions (see
# terminal_reserve_target_mwh's docstring for why: a target based on the
# realistic repeating shape, including its low hours, was measured to
# perform worse).
#
# A flat, hand-picked target (15% of capacity, at every hour in the
# horizon) was tried first and measured to help (138.0 MWh unserved vs.
# 183.5 MWh with no reserve at all), but the target was arbitrary and
# didn't adapt to how severe the visible forecast actually looked; a HARD
# bound (oemof's native min_storage_level) was tried before that and
# measured to make things WORSE (188.7 MWh unserved) since the storage got
# stuck exactly at the floor during the real emergency, unable to draw
# down further even though doing so would have helped.
#
# BELIEF_HORIZON_EXTRA_HOURS: how much further past the actual optimization
# horizon (HORIZON) the observed pattern is extended, PURELY to compute
# the terminal target -- these extra hours are never part of the LP's own
# decision variables, only of the target calculation.
BELIEF_HORIZON_EXTRA_HOURS = 24
SOFT_RESERVE_PENALTY_EUR_PER_MWH = 50

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

CO2_GAS = 0.2  # tCO2/MWh gas
CO2_EL = 0.15  # tCO2/MWh electricity

# Value of Lost Load: an unlimited "unserved heat" source is always added to
# the model at this cost so a window can never be genuinely infeasible from
# a demand shortfall. Physically this represents heat the real hardware
# could not actually supply given its fixed capacities and the storage
# state it was left with (mainly relevant for the causal case, where an
# imperfect forecast can leave the storage empty right when it's needed) --
# it is NOT a real, buildable backup source. It must sit far above any real
# operating cost so it's only ever used as an absolute last resort, but stay
# finite so the LP always solves and the shortfall can be measured rather
# than crashing the run.
VOLL_COST_EUR_PER_MWH = 10_000


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
    terminal_reserve_mwh=None,
    terminal_reserve_penalty_eur_per_mwh=None,
    terminal_reserve_position=None,
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

    ``terminal_reserve_mwh``/``terminal_reserve_penalty_eur_per_mwh`` (both
    None by default = no terminal reserve) add a cost penalty, directly to
    the Pyomo objective, for the storage's level at ONE specific timepoint
    in this window -- `terminal_reserve_position` (defaulting to the
    window's last timepoint if not given) -- falling below
    `terminal_reserve_mwh`. This is a genuine MPC terminal cost, not a
    target applied to every hour, and NOT a hard bound: the storage can
    still end lower than this if that's genuinely cheaper (e.g. avoiding
    unserved heat during a real emergency). See
    BELIEF_HORIZON_EXTRA_HOURS's docstring for how the target itself is
    computed, and why the position matters: when the optimization horizon
    is longer than what actually gets implemented before the next
    re-solve, the target needs to sit at the END OF THE IMPLEMENTED
    PORTION, not the end of the full lookahead -- a target placed beyond
    the implemented portion can be satisfied by "catching up" during the
    discarded-and-replanned tail of the window, with ~zero effect on what
    actually gets kept (measured here: with the target at the full
    horizon's end, results were identical to having no reserve at all).

    A HARD bound (oemof's native min_storage_level) was tried before this
    and measured to make things WORSE (188.7 MWh unserved vs. 183.5 MWh
    without any floor): the storage hit exactly the floor during the
    actual emergency and got stuck there, unable to draw down further even
    though doing so would have reduced the shortfall -- so that approach
    was removed rather than kept as unused dead code.

    An unlimited "unserved heat" source, penalized at VOLL_COST_EUR_PER_MWH,
    is always connected to the heat network (see that constant's docstring)
    so demand is met subject to a heavy economic penalty rather than being a
    hard constraint that can make the whole window infeasible. The returned
    ``dispatch`` always has an ``unserved_heat`` column; call
    `report_unserved_heat` on the result to see whether/where it was used.
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

    # Unlimited, heavily-penalized source -- see VOLL_COST_EUR_PER_MWH.
    # Always present so a demand shortfall degrades the result (as
    # "unserved heat") rather than making the whole window infeasible.
    unserved_heat_source = solph.components.Source(
        label="unserved heat",
        outputs={
            heat_bus: solph.flows.Flow(variable_costs=VOLL_COST_EUR_PER_MWH)
        },
    )
    es.add(unserved_heat_source)

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
    storage_kwargs = {
        "label": "heat storage",
        "nominal_capacity": cap_storage,
        "inputs": {
            heat_bus: solph.flows.Flow(
                variable_costs=VAR_COST_STORAGE,
                nominal_capacity=cap_storage / 24,
            )
        },
        "outputs": {
            heat_bus: solph.flows.Flow(
                variable_costs=VAR_COST_STORAGE,
                nominal_capacity=cap_storage / 24,
            )
        },
        "balanced": balanced,
        "loss_rate": STORAGE_LOSS_RATE,
    }
    if initial_storage_level is not None:
        storage_kwargs["initial_storage_level"] = initial_storage_level
    heat_storage = solph.components.GenericStorage(**storage_kwargs)

    es.add(gas_boiler, heat_pump, heat_storage)

    model = solph.Model(es)

    if terminal_reserve_mwh is not None:
        # Terminal reserve: penalize (don't forbid) storage_content at ONE
        # specific timepoint in this window falling below
        # `terminal_reserve_mwh`. oemof has no native feature for this, so
        # it's added directly as a Pyomo Var + Constraint + objective term
        # on top of the built model. `reserve_deficit >= target -
        # storage_content` with reserve_deficit >= 0 means the deficit is
        # exactly max(0, target - storage_content) at the optimum (never
        # more, since anything above that only adds needless objective
        # cost).
        storage_content_var = model.GenericStorageBlock.storage_content
        storage_keys = [
            k for k in storage_content_var if k[0] is heat_storage
        ]
        position = (
            terminal_reserve_position
            if terminal_reserve_position is not None
            else max(k[1] for k in storage_keys)
        )
        terminal_key = next(k for k in storage_keys if k[1] == position)

        # NOTE: a genuine scalar po.Var() breaks oemof's
        # solph.processing.results() (a hashing issue in its internals
        # when walking all Var objects) -- index it over a trivial
        # single-element set instead.
        model.reserve_deficit = po.Var([0], within=po.NonNegativeReals)
        model.reserve_deficit_constraint = po.Constraint(
            expr=model.reserve_deficit[0]
            >= terminal_reserve_mwh - storage_content_var[terminal_key]
        )
        model.objective.expr += (
            model.reserve_deficit[0] * terminal_reserve_penalty_eur_per_mwh
        )

    model.solve(solver="cbc", solve_kwargs={"tee": False})

    term_condition = str(
        model.solver_results["Solver"][0]["Termination condition"]
    )
    if term_condition != "optimal":
        raise RuntimeError(
            f"Dispatch window starting at {window.index[0]} was not solved "
            f"to optimality (termination condition: {term_condition}), "
            "despite the unlimited unserved-heat source that should make "
            "any demand shortfall solvable. This points to something other "
            "than a capacity/storage shortfall -- check the model/data for "
            "this window directly (e.g. via test_solve_dispatch_window.py)."
        )

    if terminal_reserve_mwh is not None:
        # The LP is already solved -- all real flow/storage values are
        # fixed. Remove the auxiliary terminal-reserve Var/Constraint
        # before extracting results: solph.processing.results() walks
        # every Var/Constraint in the model expecting oemof's own
        # node-based key structure, and chokes on this custom addition
        # otherwise.
        model.del_component(model.reserve_deficit_constraint)
        model.del_component(model.reserve_deficit)

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
            "unserved_heat": _flow(
                (("unserved heat", "heat network"), "flow")
            ),
        },
        index=idx,
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
        + dispatch["unserved_heat"]
        - dispatch["heat_demand"]
    )
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
    then a REPEATING-PROFILE forecast for the remaining steps -- the
    pattern of the ``n_known`` most-recently-known hours (ideally a full
    day, i.e. n_known=24) is tiled forward to fill the rest of the
    horizon, rather than holding flat at a single value.

    This still only ever uses real, already-observed data (stays fully
    causal), but is a materially better assumption than flat persistence:
    a flat forecast anchored to a single hour is blind to the diurnal
    demand cycle -- e.g. if `n_known` ends at midnight (a naturally low
    hour), a flat forecast badly underestimates the day's actual peak a
    few hours later. Tiling the whole observed pattern forward carries
    that shape (including its peak) into the forecast instead.

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
        pattern = known.reset_index(drop=True)
        n_pattern = len(pattern)
        tiled_rows = [
            pattern.iloc[i % n_pattern] for i in range(n_forecast)
        ]
        forecast_rows = pd.DataFrame(tiled_rows, index=idx[len(known) :])
        window = pd.concat([known, forecast_rows])
    else:
        window = known
    window.index = idx
    return window


def minimum_required_storage(
    demand, cap_gas_boiler, cap_heat_pump, cap_storage, loss_rate=STORAGE_LOSS_RATE
):
    """Backward pass: minimum storage level (MWh) required at the START of
    each hour in `demand` to meet all remaining demand through the end of
    the series, for a fixed-capacity design -- a pure feasibility question,
    independent of cost. See min_storage_requirement.py for the full
    standalone version (used there for the whole real year); this copy is
    used here only over short, forecast-based windows to derive a causal
    terminal reserve target -- see terminal_reserve_target_mwh.

    Returns a Series indexed like `demand`.
    """
    firm_capacity_mw = cap_gas_boiler + cap_heat_pump
    power_limit_mw = cap_storage / 24

    n = len(demand)
    min_required = pd.Series(index=demand.index, dtype=float)
    required_next = 0.0
    for k in range(n - 1, -1, -1):
        d = demand.iloc[k]
        if d <= firm_capacity_mw:
            available_charge = min(power_limit_mw, firm_capacity_mw - d)
            s_min = max(0.0, (required_next - available_charge) / (1 - loss_rate))
        else:
            discharge_needed = min(d - firm_capacity_mw, power_limit_mw)
            s_min = (required_next + discharge_needed) / (1 - loss_rate)
        s_min = min(s_min, cap_storage)
        min_required.iloc[k] = s_min
        required_next = s_min
    return min_required


def terminal_reserve_target_mwh(
    data, t0, control_step, belief_extra_hours, cap_gas_boiler, cap_heat_pump, cap_storage
):
    """Causal terminal reserve target (MWh) for a re-solve starting at
    `t0`, for the boundary at `control_step` hours from now (i.e. the end
    of what will actually be IMPLEMENTED before the next re-solve -- not
    necessarily the end of the LP's full lookahead horizon; see
    solve_dispatch_window's terminal_reserve_position docstring for why
    that distinction matters): "if conditions as severe as the worst hour
    I've just actually observed continued for `belief_extra_hours` past
    that boundary, how much would I need left in storage by the time it's
    reached?"

    Uses only the N_KNOWN real hours just observed (never the real future
    demand), so this stays causal. IMPORTANTLY, this deliberately does NOT
    reuse make_persistence_forecast's repeating-profile pattern: a
    precautionary reserve target should stay conservative, and tiling the
    full observed day (including its LOW hours) into the belief window
    lets the backward calculation assume recharging opportunities during
    those lows, which relaxes the target -- measured here to perform
    worse (176.9 MWh unserved) than holding the day's PEAK flat for the
    whole belief extension (129.2 MWh unserved), which has no such
    recovery periods to lean on. The actual dispatch forecast (used for
    real-time economic decisions, not this safety margin) is correctly
    the more realistic repeating profile; this target calculation is
    intentionally more pessimistic.
    """
    known = data["heat demand"].iloc[t0 : t0 + N_KNOWN]
    reference_level = known.max()

    # Near the very end of the year there may not be enough real timestamps
    # left to build the full belief window; shrink it rather than error.
    max_extra = max(0, len(data) - 1 - (t0 + control_step))
    belief_extra_hours = min(belief_extra_hours, max_extra)

    n_future = control_step + belief_extra_hours - len(known)
    if n_future <= 0:
        demand = known.iloc[: control_step + 1]
    else:
        demand = pd.concat(
            [known, pd.Series([reference_level] * n_future)]
        )
    if len(demand) <= control_step:
        # No room left for a belief extension (right at year-end) -- no
        # further information to base a terminal target on.
        return 0.0
    min_required = minimum_required_storage(
        demand, cap_gas_boiler, cap_heat_pump, cap_storage
    )
    # Position `control_step` in this extended series is exactly the
    # boundary we want a target for.
    return float(min_required.iloc[control_step])


def report_unserved_heat(dispatch, label=""):
    """Print a summary of any unserved heat in an already-solved dispatch
    (the `unserved_heat` column is always present -- see
    VOLL_COST_EUR_PER_MWH). Prints nothing when there is none.

    For the perfect-foresight case, any nonzero unserved heat is genuinely
    surprising (it means the fixed hardware cannot meet demand even with
    full knowledge of the future, i.e. the capacities are undersized) and
    is worth investigating -- e.g. via test_solve_dispatch_window.py.
    For the causal case, some unserved heat is an expected, honest result
    of dispatching under an imperfect forecast; the point of this
    comparison is to quantify how much.
    """
    shortfall = dispatch["unserved_heat"]
    short_mask = (shortfall > 1e-6).to_numpy()
    n_short = int(short_mask.sum())
    if n_short == 0:
        return
    first_pos = int(np.flatnonzero(short_mask)[0])
    first_ts = shortfall.index[first_pos]
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}Unserved heat: {n_short} hour(s), "
        f"{shortfall.sum():.3f} MWh total, peak {shortfall.max():.3f} MW "
        f"at {shortfall.idxmax()}. First at time step {first_pos} "
        f"({first_ts})."
    )
    context_lo = max(0, first_pos - 3)
    context_hi = min(len(dispatch), first_pos + 4)
    print(
        f"{prefix}Dispatch around the first unserved hour "
        f"(rows {context_lo}..{context_hi - 1}):"
    )
    print(
        dispatch.iloc[context_lo:context_hi][
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
    report_unserved_heat(dispatch, label="perfect foresight")
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
        terminal_target_mwh = terminal_reserve_target_mwh(
            data,
            t0,
            this_control_step,
            BELIEF_HORIZON_EXTRA_HOURS,
            CAP_GAS_BOILER_MW,
            CAP_HEAT_PUMP_MW,
            CAP_STORAGE_MWH,
        )
        dispatch, storage_content = solve_dispatch_window(
            window,
            CAP_GAS_BOILER_MW,
            CAP_HEAT_PUMP_MW,
            CAP_STORAGE_MWH,
            CO2_PRICE,
            max_heat_demand,
            storage_level,
            balanced=False,
            terminal_reserve_mwh=terminal_target_mwh,
            terminal_reserve_penalty_eur_per_mwh=SOFT_RESERVE_PENALTY_EUR_PER_MWH,
            terminal_reserve_position=this_control_step,
        )

        implemented_dispatch.append(dispatch.iloc[:this_control_step])
        # storage_content.iloc[0] is the window's start (== previous window's
        # hand-off, already recorded); keep it only for the very first
        # window so the stitched series isn't full of duplicate boundaries.
        if n_windows == 0:
            implemented_storage.append(
                storage_content.iloc[: this_control_step + 1]
            )
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
    report_unserved_heat(full_dispatch, label="causal")
    return full_dispatch, full_storage


def storage_and_reserve_stats(dispatch, storage_content):
    """Storage state-of-charge extremes and generation-capacity reserve
    margin -- simple risk indicators for how close a dispatch strategy
    came to its hard operating limits, independent of cost/CO2/LCOH.

    ``storage_min/max_mwh`` (and the capacity-fraction versions) show how
    close the storage came to running empty or overflowing full over the
    simulated year.

    ``min_daily_reserve_mw`` is the smallest margin, on the single worst
    day of the year, between the fixed converters' combined installed
    capacity (gas boiler + heat pump) and what was actually dispatched
    from them that day -- i.e. how little spare *firm* generation
    headroom was left at the tightest moment. This deliberately excludes
    the storage's own contribution (that risk is captured separately by
    the storage min/max above): a small or negative converter reserve
    means the system was leaning heavily, or entirely, on storage to
    cover the gap at that moment, which is exactly the situation that
    produced unserved heat in the causal case once the storage ran out.
    """
    storage_min_mwh = float(storage_content.min())
    storage_max_mwh = float(storage_content.max())

    installed_generation_mw = CAP_GAS_BOILER_MW + CAP_HEAT_PUMP_MW
    reserve_mw = installed_generation_mw - (
        dispatch["gas_boiler"] + dispatch["heat_pump"]
    )
    daily_min_reserve = reserve_mw.groupby(reserve_mw.index.date).min()
    worst_day = daily_min_reserve.idxmin()

    return {
        "storage_min_mwh": storage_min_mwh,
        "storage_max_mwh": storage_max_mwh,
        "storage_min_pct_of_capacity": 100 * storage_min_mwh / CAP_STORAGE_MWH,
        "storage_max_pct_of_capacity": 100 * storage_max_mwh / CAP_STORAGE_MWH,
        "min_daily_generation_reserve_mw": float(daily_min_reserve.min()),
        "min_daily_generation_reserve_date": str(worst_day),
    }


def evaluate_dispatch(dispatch, price_data):
    """Compute operation cost, CO2 and LCOH for an (already implemented)
    dispatch time series, evaluated against the *real* (not forecast)
    prices -- i.e. what this dispatch would actually have cost/emitted in
    the real world, however it was decided.

    LCOH here is the cost of energy ACTUALLY delivered by the real
    hardware (gas boiler + heat pump + storage), excluding both the
    unserved-heat penalty (VOLL_COST_EUR_PER_MWH) and the unserved MWh
    themselves from the denominator. Physically, no fuel was burned to
    produce heat that was never delivered -- a building went cold instead
    -- so folding a fictional VOLL cost into a €/MWh production-cost
    figure would conflate "cost of energy delivered" with "cost of failing
    to deliver," which are different things. The VOLL penalty and the
    unserved-heat volume are still reported (see
    `unserved_heat_cost_eur`/`unserved_heat_mwh`) as separate reliability
    metrics rather than being blended into LCOH.
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

    unserved_heat_mwh = dispatch["unserved_heat"].sum()
    unserved_heat_cost = VOLL_COST_EUR_PER_MWH * unserved_heat_mwh

    # Real operating cost: fuel, electricity, and storage O&M only. The
    # unserved-heat penalty is deliberately NOT included here -- it enters
    # `unserved_heat_cost_eur` separately instead of into LCOH's numerator.
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
    # Heat ACTUALLY delivered by the real hardware -- total demand minus
    # whatever the unserved-heat placeholder had to cover. This is the
    # correct denominator for a cost-of-energy-delivered figure; using
    # total demand (unchanged by any shortfall, since the model's internal
    # balance always "meets" it via the placeholder) would understate the
    # true cost per MWh actually produced.
    heat_produced = dispatch["heat_demand"].sum() - unserved_heat_mwh
    lcoh = LCOH(invest_cost, operation_cost, heat_produced)

    return {
        "operation_cost_eur": operation_cost,
        "total_co2_t": total_co2,
        "lcoh_eur_per_mwh": lcoh,
        "heat_produced_mwh": heat_produced,
        "storage_cycles": dispatch["storage_discharge"].sum()
        / CAP_STORAGE_MWH,
        "unserved_heat_mwh": unserved_heat_mwh,
        "unserved_heat_hours": int((dispatch["unserved_heat"] > 1e-6).sum()),
        "unserved_heat_cost_eur": unserved_heat_cost,
        # Optional secondary/blended metric: folds the VOLL penalty back in
        # and divides by total demand (not just what was actually
        # delivered) -- a single number for anyone who wants unreliability
        # priced into a €/MWh figure. NOT the primary metric above, and not
        # used anywhere else in this script; provided for reference only.
        "lcoh_including_voll_eur_per_mwh": LCOH(
            invest_cost,
            operation_cost + unserved_heat_cost,
            dispatch["heat_demand"].sum(),
        ),
    }


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
    metrics_pf.update(storage_and_reserve_stats(dispatch_pf, storage_pf))
    metrics_causal.update(
        storage_and_reserve_stats(dispatch_causal, storage_causal)
    )

    # Both cases must always account for exactly the real annual demand,
    # split between heat actually delivered and unserved heat (since
    # heat_produced_mwh now deliberately EXCLUDES unserved heat -- see
    # evaluate_dispatch). If the two don't add up, some hours in the
    # stitched dispatch are reflecting forecast values rather than what
    # genuinely happened (see the N_KNOWN == CONTROL_STEP requirement).
    true_annual_demand = data["heat demand"].iloc[:-1].sum()
    for label, m in (
        ("perfect_foresight", metrics_pf),
        ("causal", metrics_causal),
    ):
        accounted_for = m["heat_produced_mwh"] + m["unserved_heat_mwh"]
        gap = abs(accounted_for - true_annual_demand)
        if gap > 1.0:
            raise RuntimeError(
                f"[{label}] heat_produced_mwh + unserved_heat_mwh "
                f"({accounted_for:.1f}) does not match the true annual "
                f"demand ({true_annual_demand:.1f} MWh, gap {gap:.1f} MWh) "
                "-- some hours in the stitched dispatch are not reflecting "
                "real data. Check N_KNOWN == CONTROL_STEP and the window "
                "stitching logic in run_causal_dispatch."
            )

    dispatch_pf.to_csv(RESULTS_DIR / "dispatch_perfect_foresight.csv")
    dispatch_causal.to_csv(RESULTS_DIR / "dispatch_causal.csv")
    # Storage content is boundary-indexed (N+1 points per case, see
    # solve_dispatch_window's docstring) -- saved separately from the
    # interval-indexed dispatch flows above so the plot script never needs
    # to re-solve anything to get the SOC trajectory.
    storage_pf.rename("storage_content_mwh").to_csv(
        RESULTS_DIR / "storage_perfect_foresight.csv"
    )
    storage_causal.rename("storage_content_mwh").to_csv(
        RESULTS_DIR / "storage_causal.csv"
    )

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
