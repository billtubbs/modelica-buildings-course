"""Hourly RTO dispatch simulation over a reduced window (one week before
and after the January cold snap) plus a perfect-foresight comparison,
saving results to CSV. No terminal reserve mechanism yet (that mechanism
was the trigger for a CBC/pyomo solution-loading bug, reported
separately) -- this establishes the basic rolling-horizon loop first.

See rto_coldsnap_plot.py for plots -- it reads the CSVs saved here rather
than re-solving anything.
"""

from pathlib import Path

import pandas as pd
import pyomo.environ as po
import oemof.solph as solph
from tqdm import tqdm

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_GAS_BOILER_MW = 4.419 * 1.002
CAP_HEAT_PUMP_MW = 10.649 * 1.002
CAP_STORAGE_MWH = 187.176 * 1.002
COP = 3.5
VAR_COST_HEAT_PUMP = 1.2
GAS_BOILER_EFFICIENCY = 0.95
VAR_COST_GAS_BOILER = 1.10
VAR_COST_STORAGE = 0.1
STORAGE_LOSS_RATE = 0.001
CO2_GAS = 0.2
CO2_EL = 0.15
CO2_PRICE = 20
VOLL_COST_EUR_PER_MWH = 10_000

HORIZON = 48
CONTROL_STEP = 1
N_KNOWN = 24
PATTERN_HOURS = 24


def _build_and_solve(window, storage_level, cap_gas_boiler, cap_heat_pump, cap_storage, co2_price, max_heat_demand):
    """Build and solve one dispatch LP for the given data window (already
    real or forecast, as the caller prefers). Returns (dispatch_df,
    storage_content_list). Direct pyomo variable extraction, not oemof's
    solph.processing.results() (which has a separate, confirmed bug in
    grouping storage_content -- reported separately).
    """
    n = len(window) - 1
    gas_cost = (window["gas price"] + co2_price * CO2_GAS).iloc[:-1]
    el_cost = (window["el_spot_price"] + co2_price * CO2_EL).iloc[:-1]

    es = solph.EnergySystem(timeindex=window.index, infer_last_interval=False)
    heat_bus = solph.Bus(label="heat")
    gas_bus = solph.Bus(label="gas")
    waste_bus = solph.Bus(label="waste_heat")
    el_bus = solph.Bus(label="elec")
    es.add(heat_bus, gas_bus, waste_bus, el_bus)

    gas_source = solph.components.Source(label="gas_source", outputs={gas_bus: solph.flows.Flow(variable_costs=gas_cost)})
    el_source = solph.components.Source(label="el_source", outputs={el_bus: solph.flows.Flow(variable_costs=el_cost)})
    waste_source = solph.components.Source(label="waste_source", outputs={waste_bus: solph.flows.Flow()})
    unserved = solph.components.Source(label="unserved", outputs={heat_bus: solph.flows.Flow(variable_costs=VOLL_COST_EUR_PER_MWH)})
    sink = solph.components.Sink(
        label="sink",
        inputs={heat_bus: solph.flows.Flow(nominal_capacity=max_heat_demand, fix=(window["heat demand"] / max_heat_demand).iloc[:-1])},
    )
    es.add(gas_source, el_source, waste_source, unserved, sink)

    gas_boiler = solph.components.Converter(
        label="gas_boiler",
        inputs={gas_bus: solph.flows.Flow()},
        outputs={heat_bus: solph.flows.Flow(nominal_capacity=cap_gas_boiler, variable_costs=VAR_COST_GAS_BOILER)},
        conversion_factors={gas_bus: GAS_BOILER_EFFICIENCY},
    )
    heat_pump = solph.components.Converter(
        label="heat_pump",
        inputs={el_bus: solph.flows.Flow(), waste_bus: solph.flows.Flow()},
        outputs={heat_bus: solph.flows.Flow(nominal_capacity=cap_heat_pump, variable_costs=VAR_COST_HEAT_PUMP)},
        conversion_factors={el_bus: 1 / COP, waste_bus: (COP - 1) / COP},
    )
    storage = solph.components.GenericStorage(
        label="storage",
        nominal_capacity=cap_storage,
        inputs={heat_bus: solph.flows.Flow(variable_costs=VAR_COST_STORAGE, nominal_capacity=cap_storage / 24)},
        outputs={heat_bus: solph.flows.Flow(variable_costs=VAR_COST_STORAGE, nominal_capacity=cap_storage / 24)},
        balanced=False,
        loss_rate=STORAGE_LOSS_RATE,
        initial_storage_level=storage_level,
    )
    es.add(gas_boiler, heat_pump, storage)

    model = solph.Model(es)
    solver = po.SolverFactory("cbc", solver_io="lp")
    result = solver.solve(model, tee=False)
    if str(result.solver.termination_condition) != "optimal":
        raise RuntimeError(f"Not optimal for window starting {window.index[0]}: {result.solver.termination_condition}")

    def flow(src, dst):
        return [model.flow[src, dst, t].value for t in range(n)]

    dispatch = pd.DataFrame(
        {
            "gas_boiler": flow(gas_boiler, heat_bus),
            "heat_pump": flow(heat_pump, heat_bus),
            "storage_discharge": flow(storage, heat_bus),
            "storage_charge": flow(heat_bus, storage),
            "heat_demand": flow(heat_bus, sink),
            "unserved_heat": flow(unserved, heat_bus),
        },
        index=window.index[:-1],
    )
    storage_content = [model.GenericStorageBlock.storage_content[storage, t].value for t in range(n + 1)]

    if dispatch.isnull().values.any() or any(v is None for v in storage_content):
        raise RuntimeError(f"Missing solved value(s) for window starting {window.index[0]}")

    return dispatch, storage_content


def rto_step(
    data, t0, storage_level, cap_gas_boiler, cap_heat_pump, cap_storage,
    co2_price, max_heat_demand, horizon, control_step, n_known, pattern_hours,
):
    """One RTO iteration: build the forecast window, solve the dispatch
    LP, and return (implemented dispatch, implemented storage content
    values, updated storage level, hours advanced).
    """
    n_total = len(data) - 1
    this_horizon = min(horizon, n_total - t0)
    this_control_step = min(control_step, this_horizon)
    this_n_known = min(n_known, this_horizon + 1)

    # Forecast window: real data for the first this_n_known hours, then a
    # repeating pattern_hours-long tile for the rest.
    idx = data.index[t0 : t0 + this_horizon + 1]
    known = data.iloc[t0 : t0 + this_n_known].copy()
    n_forecast = len(idx) - len(known)
    if n_forecast > 0:
        pstart = max(0, t0 + this_n_known - pattern_hours)
        pattern = data.iloc[pstart : t0 + this_n_known].reset_index(drop=True)
        tiled = [pattern.iloc[i % len(pattern)] for i in range(n_forecast)]
        window = pd.concat([known, pd.DataFrame(tiled, index=idx[len(known):])])
    else:
        window = known
    window.index = idx

    dispatch, storage_content = _build_and_solve(
        window, storage_level, cap_gas_boiler, cap_heat_pump, cap_storage, co2_price, max_heat_demand
    )
    new_storage_level = min(1.0, max(0.0, storage_content[this_control_step] / cap_storage))
    implemented_storage = storage_content[1 : this_control_step + 1]  # end-of-hour values, one per implemented hour
    return dispatch.iloc[:this_control_step], implemented_storage, new_storage_level, this_control_step


def solve_perfect_foresight(data, cap_gas_boiler, cap_heat_pump, cap_storage, co2_price, initial_storage_level):
    """Single solve over the whole period using real data throughout --
    the causal case's forecast uncertainty removed entirely. Same
    starting storage level as the RTO run, for a fair comparison.
    """
    max_heat_demand = data["heat demand"].max()
    dispatch, storage_content = _build_and_solve(
        data, initial_storage_level, cap_gas_boiler, cap_heat_pump, cap_storage, co2_price, max_heat_demand
    )
    storage_series = pd.Series(storage_content, index=data.index, name="storage_content_mwh")
    return dispatch, storage_series


def run_rto(
    data, cap_gas_boiler, cap_heat_pump, cap_storage, co2_price,
    initial_storage_level, horizon, control_step, n_known, pattern_hours=PATTERN_HOURS,
):
    max_heat_demand = data["heat demand"].max()
    n_total = len(data) - 1
    t0 = 0
    storage_level = initial_storage_level
    chunks = []
    storage_values = [initial_storage_level * cap_storage]
    storage_index = [data.index[0]]
    pbar = tqdm(total=n_total, unit="hr")
    while t0 < n_total:
        chunk, implemented_storage, storage_level, step = rto_step(
            data, t0, storage_level, cap_gas_boiler, cap_heat_pump, cap_storage,
            co2_price, max_heat_demand, horizon, control_step, n_known, pattern_hours,
        )
        chunks.append(chunk)
        storage_values.extend(implemented_storage)
        storage_index.extend(data.index[t0 + 1 : t0 + 1 + step])
        t0 += step
        pbar.update(step)
    pbar.close()
    dispatch = pd.concat(chunks)
    storage_series = pd.Series(storage_values, index=storage_index, name="storage_content_mwh")
    return dispatch, storage_series


if __name__ == "__main__":
    data_full = pd.read_csv(DATA_DIR / "input_data.csv", sep=";", index_col=0, parse_dates=True)
    data = data_full.loc["2019-01-12":"2019-02-03"]
    initial_storage_level = 0.5

    dispatch_rto, storage_rto = run_rto(
        data, CAP_GAS_BOILER_MW, CAP_HEAT_PUMP_MW, CAP_STORAGE_MWH, CO2_PRICE,
        initial_storage_level=initial_storage_level, horizon=HORIZON, control_step=CONTROL_STEP, n_known=N_KNOWN,
    )
    dispatch_rto.to_csv(RESULTS_DIR / "rto_hourly_dispatch.csv")
    storage_rto.to_csv(RESULTS_DIR / "rto_hourly_storage.csv")

    dispatch_pf, storage_pf = solve_perfect_foresight(
        data, CAP_GAS_BOILER_MW, CAP_HEAT_PUMP_MW, CAP_STORAGE_MWH, CO2_PRICE, initial_storage_level
    )
    dispatch_pf.to_csv(RESULTS_DIR / "perfect_foresight_dispatch.csv")
    storage_pf.to_csv(RESULTS_DIR / "perfect_foresight_storage.csv")

    for label, dispatch in [("RTO (hourly)", dispatch_rto), ("Perfect foresight", dispatch_pf)]:
        unserved_mwh = dispatch["unserved_heat"].sum()
        unserved_hours = int((dispatch["unserved_heat"] > 1e-6).sum())
        print(f"{label}: unserved heat {unserved_mwh:.3f} MWh over {unserved_hours} hour(s)")
    print(f"Saved results to {RESULTS_DIR}/")
