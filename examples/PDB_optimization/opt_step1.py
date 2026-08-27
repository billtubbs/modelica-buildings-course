"""Gas boiler + heat pump + heat storage: cost/CO2 Pareto front via MILP.

This script builds and solves a family of mixed-integer linear programming
(MILP) capacity-investment + dispatch models of a district-heat supply
system using ``oemof.solph``, with the CBC solver as backend. The system can
combine three heat sources feeding a common heat network:

    * a gas boiler (gas -> heat),
    * a heat pump (electricity + waste heat -> heat, fixed COP), and
    * a thermal storage tank.

Each unit's installed capacity is either optimised as an investment
variable or pinned to a fixed value, and the fuel/electricity cost together
with a CO2 price ("lambda", in EUR/tCO2) determine dispatch. Solving the
model repeatedly over a sweep of CO2 prices traces out a cost (LCOH) vs.
emissions (CO2) Pareto front for the supply system.

Cases solved:
    1. Gas boiler only (reference anchor).
    2. Heat pump only (reference anchor).
    3. Heat pump + storage, cost-optimal design (no gas boiler).
    4. Gas boiler + heat pump + storage, swept over 6 CO2 prices
       (lambda = 0, 20, 50, 100, 200, 400 EUR/tCO2).

Input:
    ``<DATA_DIR>/input_data.csv`` -- semicolon-separated hourly time series
    with a parseable datetime index and columns:
        * "heat demand"   heat load of the network [MW]
        * "gas price"     gas commodity price [EUR/MWh]
        * "el_spot_price" electricity spot price [EUR/MWh]

Outputs (directories are created automatically if they do not exist):
    ``<PLOT_DIR>/pareto_1_anchors.png``
    ``<PLOT_DIR>/pareto_2_hp_storage.png``
    ``<PLOT_DIR>/pareto_3_full.png``
        Cost vs. CO2 Pareto-front scatter plots, each building on the last.
    ``<PLOT_DIR>/dispatch_<case>.png``, ``<PLOT_DIR>/storage_content_<case>.png``
        Hourly dispatch stack and storage state-of-charge plots, generated
        for the cheapest full-system design and for the lambda = 20 design.
    ``<RESULTS_DIR>/dispatch_<case>.csv``
        Hourly dispatch time series (per technology, plus gas price and
        energy balance check) for the same two cases.
    ``<RESULTS_DIR>/dispatch_combitimetable_<case>.txt``
        Modelica ``CombiTimeTable`` export of the lambda = 20 dispatch, in
        watts, for use as a boundary condition in a downstream Modelica model.

    A summary table of all solved cases is also printed to the console.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import oemof.solph as solph
import pandas as pd
import pyomo.environ as po

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")
RESULTS_DIR = Path("results")

for directory in (DATA_DIR, PLOT_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def epc(invest_cost, i=0.05, n=20):
    af = (i * (1 + i) ** n) / ((1 + i) ** n - 1)
    return invest_cost * af


def LCOH(invest_cost, operation_cost, heat_produced, revenue=0, i=0.05, n=20):
    pvf = ((1 + i) ** n - 1) / (((1 + i) ** n) * i)
    return (invest_cost + pvf * (operation_cost - revenue)) / (
        pvf * heat_produced
    )


def print_sizing_bar(res):
    scale = 2

    def bar(value):
        return "█" * int(value * scale)

    print(f"\nCase = {res['case_label']}")
    sizing_rows = [
        ("Gas boiler ", res["cap_gas_boiler_mw"], "MW", res["cap_gas_boiler_mw"]),
        ("Heat pump  ", res["cap_heat_pump_mw"], "MW", res["cap_heat_pump_mw"]),
        ("Storage (E)", res["cap_storage_mwh"], "MWh", res["cap_storage_mwh"] / 10),
        ("Storage (P)", res["cap_storage_out_mw"], "MW", res["cap_storage_out_mw"]),
    ]
    for label, value, unit, bar_value in sizing_rows:
        print(f"{label} ({value:.1f} {unit}): {bar(bar_value)}")

    print(f"LCOH: {res['lcoh']:.2f} €/MWh")
    print(f"Total CO2: {res['co2']:.1f} tCO2")
    print("  Dispatch [MWh/yr]:")
    dispatch_rows = [
        ("Gas boiler", res["e_gas_boiler_mwh"]),
        ("Heat pump", res["e_heat_pump_mwh"]),
        ("Storage charge", res["e_storage_in_mwh"]),
        ("Storage discharge", res["e_storage_out_mwh"]),
        ("Total heat", res["e_heat_demand_mwh"]),
    ]
    for label, value in dispatch_rows:
        print(f"    {label + ':':<19}{value:.1f}")


def solve_case(
    data,
    co2_price=0,
    use_gas_boiler=True,
    use_heat_pump=True,
    use_storage=True,
    cap_heat_pump_fixed=None,
    cap_storage_fixed=None,
    case_label=None,
):
    """Solve one design case.

    Capacities are investment variables by default. Passing
    ``cap_heat_pump_fixed`` / ``cap_storage_fixed`` pins that capacity instead,
    which turns the run into a pure dispatch problem for a prescribed design.
    """
    cop = 3.5
    spec_inv_heat_pump = 500000
    var_cost_heat_pump = 1.2

    spec_inv_gas_boiler = 60000
    var_cost_gas_boiler = 1.10
    gas_boiler_efficiency = 0.95

    spec_inv_storage = 1060
    var_cost_storage = 0.1

    co2_gas = 0.2
    co2_el = 0.15

    gas_source_cost = data["gas price"] + co2_price * co2_gas
    electricity_source_cost = data["el_spot_price"] + co2_price * co2_el
    hp_var_cost = var_cost_heat_pump
    storage_var_cost = var_cost_storage
    if case_label is None:
        case_label = f"lambda_{co2_price}"

    es = solph.EnergySystem(
        timeindex=data.index,
        infer_last_interval=False,
    )

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
                nominal_capacity=data["heat demand"].max(),
                fix=(data["heat demand"] / data["heat demand"].max()),
            )
        },
    )

    es.add(gas_source, electricity_source, waste_heat_source, heat_sink)

    # Components are only instantiated when active: constructing a component
    # registers its flows with the connected buses, so a merely-unadded
    # component would still corrupt the bus balances.
    gas_boiler = None
    heat_pump = None
    heat_storage = None

    if use_gas_boiler:
        gas_boiler = solph.components.Converter(
            label="gas boiler",
            inputs={gas_bus: solph.flows.Flow()},
            outputs={
                heat_bus: solph.flows.Flow(
                    nominal_capacity=solph.Investment(
                        ep_costs=epc(spec_inv_gas_boiler),
                        maximum=20,
                    ),
                    variable_costs=var_cost_gas_boiler,
                )
            },
            conversion_factors={gas_bus: gas_boiler_efficiency},
        )

    if use_heat_pump:
        hp_nominal_capacity = (
            cap_heat_pump_fixed
            if cap_heat_pump_fixed is not None
            else solph.Investment(ep_costs=epc(spec_inv_heat_pump), maximum=20)
        )
        heat_pump = solph.components.Converter(
            label="heat pump",
            inputs={
                electricity_bus: solph.flows.Flow(),
                waste_heat_bus: solph.flows.Flow(),
            },
            outputs={
                heat_bus: solph.flows.Flow(
                    nominal_capacity=hp_nominal_capacity,
                    variable_costs=hp_var_cost,
                )
            },
            conversion_factors={
                electricity_bus: 1 / cop,
                waste_heat_bus: (cop - 1) / cop,
            },
        )

    if use_storage:
        # invest_relation_* is only meaningful in investment mode; with a pinned
        # energy capacity the same 1/24 C-rate is applied directly to the flows.
        if cap_storage_fixed is not None:
            storage_extra = {}
            storage_nominal = cap_storage_fixed
            storage_flow_capacity = cap_storage_fixed / 24
        else:
            storage_extra = {
                "invest_relation_input_capacity": 1 / 24,
                "invest_relation_output_capacity": 1 / 24,
            }
            storage_nominal = solph.Investment(ep_costs=epc(spec_inv_storage))
            storage_flow_capacity = solph.Investment()

        heat_storage = solph.components.GenericStorage(
            label="heat storage",
            nominal_capacity=storage_nominal,
            inputs={
                heat_bus: solph.flows.Flow(
                    variable_costs=storage_var_cost,
                    nominal_capacity=storage_flow_capacity,
                )
            },
            outputs={
                heat_bus: solph.flows.Flow(
                    variable_costs=storage_var_cost,
                    nominal_capacity=storage_flow_capacity,
                )
            },
            balanced=True,
            loss_rate=0.001,
            **storage_extra,
        )

    active_components = [
        c for c in (gas_boiler, heat_pump, heat_storage) if c is not None
    ]
    es.add(*active_components)

    model = solph.Model(es)

    # Combined capacity cap only binds when both converters are present *and*
    # both capacities are investment variables; otherwise each converter is
    # already limited by its own maximum=20 or by a pinned capacity.
    if use_gas_boiler and use_heat_pump and cap_heat_pump_fixed is None:
        model.combined_cap_constraint = po.Constraint(
            expr=(
                model.InvestmentFlowBlock.invest[gas_boiler, heat_bus, 0]
                + model.InvestmentFlowBlock.invest[heat_pump, heat_bus, 0]
                <= 20
            )
        )

    model.solve(solver="cbc", solve_kwargs={"tee": False})

    results = solph.processing.results(model)

    data_gas_bus = solph.views.node(results, "gas network")["sequences"]
    data_heat_bus = solph.views.node(results, "heat network")["sequences"]
    data_el_bus = solph.views.node(results, "electricity network")["sequences"]
    data_caps = solph.views.node(results, "heat network")["scalars"]

    def flow_sum(df, key):
        """Sum a flow, returning 0 if the component/flow is absent."""
        return df[key].sum() if key in df.columns else 0.0

    def cost_sum(price, df, key):
        return (price * df[key]).sum() if key in df.columns else 0.0

    cap_gas_boiler = (
        data_caps[(("gas boiler", "heat network"), "invest")]
        if use_gas_boiler
        else 0.0
    )
    if not use_heat_pump:
        cap_heat_pump = 0.0
    elif cap_heat_pump_fixed is not None:
        cap_heat_pump = cap_heat_pump_fixed
    else:
        cap_heat_pump = data_caps[(("heat pump", "heat network"), "invest")]

    if not use_storage:
        cap_storage = 0.0
        cap_storage_out = 0.0
    elif cap_storage_fixed is not None:
        cap_storage = cap_storage_fixed
        cap_storage_out = cap_storage_fixed / 24
    else:
        cap_storage = solph.views.node(results, "heat storage")["scalars"][
            (("heat storage", "None"), "invest")
        ]
        cap_storage_out = data_caps[
            (("heat storage", "heat network"), "invest")
        ]

    invest_cost = (
        spec_inv_gas_boiler * cap_gas_boiler
        + spec_inv_heat_pump * cap_heat_pump
        + spec_inv_storage * cap_storage
    )

    operation_cost = (
        var_cost_gas_boiler
        * flow_sum(data_heat_bus, (("gas boiler", "heat network"), "flow"))
        + cost_sum(
            data["gas price"],
            data_gas_bus,
            (("gas network", "gas boiler"), "flow"),
        )
        + var_cost_heat_pump
        * flow_sum(data_heat_bus, (("heat pump", "heat network"), "flow"))
        + cost_sum(
            data["el_spot_price"],
            data_el_bus,
            (("electricity network", "heat pump"), "flow"),
        )
        + var_cost_storage
        * flow_sum(data_heat_bus, (("heat storage", "heat network"), "flow"))
        + var_cost_storage
        * flow_sum(data_heat_bus, (("heat network", "heat storage"), "flow"))
    )

    heat_produced = data_heat_bus[
        (("heat network", "heat sink"), "flow")
    ].sum()

    total_co2 = co2_gas * flow_sum(
        data_gas_bus, (("gas network", "gas boiler"), "flow")
    ) + co2_el * flow_sum(
        data_el_bus, (("electricity network", "heat pump"), "flow")
    )

    lcoh = LCOH(invest_cost, operation_cost, heat_produced)

    e_gas_boiler = flow_sum(
        data_heat_bus, (("gas boiler", "heat network"), "flow")
    )
    e_heat_pump = flow_sum(
        data_heat_bus, (("heat pump", "heat network"), "flow")
    )
    e_storage_in = flow_sum(
        data_heat_bus, (("heat network", "heat storage"), "flow")
    )
    e_storage_out = flow_sum(
        data_heat_bus, (("heat storage", "heat network"), "flow")
    )

    data_heat_storage = (
        solph.views.node(results, "heat storage")["sequences"]
        if use_storage
        else None
    )

    return {
        "case_label": case_label,
        "co2_price": co2_price,
        "cap_gas_boiler_mw": cap_gas_boiler,
        "cap_heat_pump_mw": cap_heat_pump,
        "cap_storage_mwh": cap_storage,
        "cap_storage_out_mw": cap_storage_out,
        "lcoh": lcoh,
        "co2": total_co2,
        "e_gas_boiler_mwh": e_gas_boiler,
        "e_heat_pump_mwh": e_heat_pump,
        "e_storage_in_mwh": e_storage_in,
        "e_storage_out_mwh": e_storage_out,
        "e_heat_demand_mwh": heat_produced,
        "_data_heat_bus": data_heat_bus,
        "_data_heat_storage": data_heat_storage,
        "_gas_price": data["gas price"],
    }


def slugify(label):
    """Filename-safe form of a case label, e.g. 'full λ=0' -> 'full_lambda_0'."""
    label = label.replace("λ", "lambda").replace("=", " ")
    return "_".join(
        "".join(ch for ch in part if ch.isascii() and ch.isalnum())
        for part in label.split()
    )


def plot_dispatch(res):
    unit_colors = {
        "gas boiler": "#EC6707",
        "heat pump": "#B54036",
        "heat storage (discharge)": "#BFBFBF",
        "heat storage (charge)": "#696969",
    }

    data_heat_bus = res["_data_heat_bus"]
    data_heat_storage = res["_data_heat_storage"]

    fig_dispatch, ax = plt.subplots(figsize=[10, 6])
    bottom = 0
    for unit in ["heat pump", "gas boiler", "heat storage"]:
        unit_label = f"{unit} (discharge)" if "storage" in unit else unit
        ax.bar(
            data_heat_bus.index,
            data_heat_bus[((unit, "heat network"), "flow")],
            label=unit_label,
            color=unit_colors[unit_label],
            bottom=bottom,
        )
        bottom += data_heat_bus[((unit, "heat network"), "flow")]

    ax.bar(
        data_heat_bus.index,
        -1 * data_heat_bus[(("heat network", "heat storage"), "flow")],
        label="heat storage (charge)",
        color=unit_colors["heat storage (charge)"],
    )

    ax.legend(loc="upper center", ncol=2)
    ax.grid(axis="y")
    ax.set_ylim(-22, 22)
    ax.set_ylabel("Hourly heat production in MWh")

    fig_storage, ax = plt.subplots(figsize=[10, 6])
    ax.plot(
        data_heat_storage[(("heat storage", "None"), "storage_content")],
        color="#00395B",
    )
    ax.grid(axis="y")
    ax.set_ylabel("Hourly heat storage content in MWh")

    slug = slugify(res["case_label"])
    fig_dispatch.tight_layout()
    fig_storage.tight_layout()
    fig_dispatch.savefig(PLOT_DIR / f"dispatch_{slug}.png", dpi=150)
    fig_storage.savefig(PLOT_DIR / f"storage_content_{slug}.png", dpi=150)

    dispatch_df = pd.DataFrame(
        {
            "gas_boiler": data_heat_bus[
                (("gas boiler", "heat network"), "flow")
            ],
            "heat_pump": data_heat_bus[
                (("heat pump", "heat network"), "flow")
            ],
            "storage_discharge": data_heat_bus[
                (("heat storage", "heat network"), "flow")
            ],
            "storage_charge": data_heat_bus[
                (("heat network", "heat storage"), "flow")
            ],
            "heat_demand": data_heat_bus[
                (("heat network", "heat sink"), "flow")
            ],
            "gas_price": res["_gas_price"],
        }
    )
    dispatch_df["balance"] = (
        dispatch_df["gas_boiler"]
        + dispatch_df["heat_pump"]
        + dispatch_df["storage_discharge"]
        - dispatch_df["storage_charge"]
        - dispatch_df["heat_demand"]
    )
    dispatch_df.to_csv(RESULTS_DIR / f"dispatch_{slug}.csv")


def write_combitimetable(res, filename=None):
    """Export the hourly dispatch as a Modelica CombiTimeTable file.

    Columns are heat pump / gas boiler / storage charge / storage discharge /
    heat demand, converted from MW to W. The optimiser models 8759 intervals
    for 8760 timestamps (infer_last_interval=False), so the final hour has no
    solution and is set equal to the previous step -- matching the convention
    in the existing dispatch_combitimetable.txt.
    """
    data_heat_bus = res["_data_heat_bus"]
    df = pd.DataFrame(
        {
            "heat_pump": data_heat_bus[
                (("heat pump", "heat network"), "flow")
            ],
            "gas_boiler": data_heat_bus[
                (("gas boiler", "heat network"), "flow")
            ],
            "storage_charge": data_heat_bus[
                (("heat network", "heat storage"), "flow")
            ],
            "storage_discharge": data_heat_bus[
                (("heat storage", "heat network"), "flow")
            ],
            "heat_demand": data_heat_bus[
                (("heat network", "heat sink"), "flow")
            ],
        }
    ).ffill()

    watts = df * 1e6
    seconds = (df.index - df.index[0]).total_seconds().astype("int64")

    if filename is None:
        filename = (
            RESULTS_DIR
            / f"dispatch_combitimetable_{slugify(res['case_label'])}.txt"
        )

    with open(filename, "w") as f:
        f.write("#1\n")
        f.write(
            f"# Dispatch for {res['case_label']}: heat pump"
            f" {res['cap_heat_pump_mw']:.1f} MW / storage"
            f" {res['cap_storage_mwh']:.0f} MWh / gas boiler"
            f" {res['cap_gas_boiler_mw']:.1f} MW\n"
        )
        f.write(
            f"# LCOH {res['lcoh']:.3f} EUR/MWh, total CO2 {res['co2']:.1f} tCO2\n"
        )
        f.write("# col 1: time              [s]\n")
        f.write("# col 2: heat pump         [W]\n")
        f.write("# col 3: gas boiler        [W]\n")
        f.write("# col 4: storage charge    [W]   heat into the store\n")
        f.write("# col 5: storage discharge [W]   heat out of the store\n")
        f.write(
            "# col 6: heat demand       [W]   = col2 + col3 + col5 - col4\n"
        )
        f.write(f"double dispatch({len(watts)},6)\n")
        for t, (_, row) in zip(seconds, watts.iterrows()):
            values = "  ".join(f"{v:.3f}" for v in row)
            f.write(f"  {t}  {values}\n")

    print(f"Wrote {filename} ({len(watts)} rows)")


data = pd.read_csv(
    DATA_DIR / "input_data.csv", sep=";", index_col=0, parse_dates=True
)

# 6 CO2 prices (lambda values) used for every optimization search.
co2_prices = [0, 20, 50, 100, 200, 400]

# -------------------------------------------------------------------------
# Solve all cases once, then reuse the results across the three plots.
# -------------------------------------------------------------------------

# Reference anchors: single-technology systems (CO2 price is irrelevant, only
# one configuration is possible, so it is left at 0).
print("\nSolving reference case: gas boiler only")
boiler_only = solve_case(
    data,
    co2_price=0,
    use_gas_boiler=True,
    use_heat_pump=False,
    use_storage=False,
    case_label="gas boiler only",
)
print_sizing_bar(boiler_only)

print("\nSolving reference case: heat pump only")
hp_only = solve_case(
    data,
    co2_price=0,
    use_gas_boiler=False,
    use_heat_pump=True,
    use_storage=False,
    case_label="heat pump only",
)
print_sizing_bar(hp_only)

# Design study 1: heat pump + storage (no gas boiler).
#
# A lambda sweep has no lever here: the heat demand is fixed, the COP is constant
# and the electricity emission factor is constant, so total CO2 is pinned at
# co2_el * demand / cop no matter how the system is sized. With CO2 effectively
# constant this reduces to a single-objective cost minimization, so a single
# point -- the cost-optimal design -- is all there is to show.
print("\nSolving heat pump + storage: cost-optimal design")
hp_storage_opt = solve_case(
    data,
    co2_price=0,
    use_gas_boiler=False,
    use_heat_pump=True,
    use_storage=True,
    case_label="HP+sto optimum",
)
print_sizing_bar(hp_storage_opt)

# Optimization search 2: gas boiler + heat pump + storage, 6 lambda values.
full_sweep = []
for cp in co2_prices:
    print(f"\nSolving full system, lambda = {cp} €/tCO2")
    res = solve_case(
        data,
        co2_price=cp,
        use_gas_boiler=True,
        use_heat_pump=True,
        use_storage=True,
        case_label=f"full λ={cp}",
    )
    full_sweep.append(res)
    print_sizing_bar(res)

# Dispatch for the most economic boiler + heat pump + storage design. That is the
# lambda = 0 end of the sweep: with no carbon price the optimizer is free to lean
# on the cheap gas boiler, which is exactly what makes it the low-cost design.
best_full = min(full_sweep, key=lambda c: c["lcoh"])
print(
    f"\nPlotting dispatch for the most economic full-system case: "
    f"{best_full['case_label']} (LCOH {best_full['lcoh']:.2f} €/MWh)"
)
plot_dispatch(best_full)

# The lambda = 20 design (heat pump 10.6 MW / storage 187 MWh / gas boiler
# 4.4 MW) is the one handed over to the Modelica model, so it gets its own
# dispatch figures plus a CombiTimeTable file.
case_l20 = next(c for c in full_sweep if c["co2_price"] == 20)
print(
    f"\nPlotting dispatch and writing Modelica table for: "
    f"{case_l20['case_label']} (LCOH {case_l20['lcoh']:.2f} €/MWh)"
)
plot_dispatch(case_l20)
write_combitimetable(case_l20)


# -------------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------------
# Plots 1 and 2 scale to the solved cases, as they originally did: derived from
# every case, so points do not shift between those figures as cases are added.
_all_cases = [boiler_only, hp_only, hp_storage_opt] + full_sweep
_co2 = [c["co2"] for c in _all_cases]
_lcoh = [c["lcoh"] for c in _all_cases]

XLIM = (
    min(_co2) - 0.06 * (max(_co2) - min(_co2)),
    max(_co2) + 0.06 * (max(_co2) - min(_co2)),
)
# Extra headroom below so the lowest-cost points sit clear of the axis.
YLIM = (
    min(_lcoh) - 0.18 * (max(_lcoh) - min(_lcoh)),
    max(_lcoh) + 0.12 * (max(_lcoh) - min(_lcoh)),
)

# Plot 3 instead uses limits fixed identically in step2c_adv_super.py, _el.py
# and _eff.py, so that one figure can be read straight across the scenarios.
# Wide enough to hold every case of all three: CO2 runs from 2216 t (heat pump
# only at COP 4.5) to 12632 t (gas boiler only, identical in every scenario),
# LCOH from 15.6 EUR/MWh (the cost optimum on gas reduced 20%) to 24.5 (heat
# pump only at COP 3.5). The plot-3 label positions below are tuned to this
# range.
XLIM3 = (1500, 13500)
YLIM3 = (14, 26)


def style_pareto(ax, xlim=XLIM, ylim=YLIM):
    ax.set_xlabel("Total CO2 [tCO2]")
    ax.set_ylabel("LCOH [€/MWh]")
    ax.grid(True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # Fixed corner rather than "best", so the legend does not jump between figures.
    ax.legend(loc="upper right", fontsize=9)
    # Bottom-left: with the shared limits the bottom-right corner is occupied by
    # the gas-boiler-only anchor in the half-price-gas scenario.
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


# Compact "heat pump / storage / gas boiler" sizing label. The triple is terse
# enough to sit next to tightly clustered points; SIZE_NOTE spells out the units.
SIZE_NOTE = "labels: heat pump [MW] / storage [MWh] / gas boiler [MW]"


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
        # Backing so the iso-cost line in plot 3 -- which runs through the gas
        # boiler anchor at exactly this height -- does not strike the label out.
        bbox=dict(
            boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"
        ),
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
        # The boiler anchor sits at the right edge, so its label goes to the left.
        annotate_case(ax, boiler_only, dx=-11, ha="right")
        annotate_case(ax, hp_only)


def plot_hp_storage_opt(ax, annotate=True, hollow=False):
    """Plot the single cost-optimal heat pump + storage design.

    ``hollow`` draws it as a ring: in plot 3 the lambda=400 full-system point is
    nearly coincident with this one (the boiler has all but vanished by then),
    so a filled marker would hide it.
    """
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
    """Non-dominated cases (minimising both CO2 and LCOH), ordered by CO2.

    Fed every solved design, not just the multi-objective sweep, so the line
    only ever passes through points nothing else beats on both axes -- the
    single-technology anchors drop out on their own.
    """
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
        # Past the cheapest design, spending more CO2 buys nothing: no design
        # anywhere to the right costs less. Running the front flat to the edge
        # makes that explicit, and any point above the flat stretch -- gas
        # boiler only, in plot 3 -- is visibly off the optimal set.
        xs.append(XLIM3[1])
        ys.append(ys[-1])
    ax.plot(
        xs,
        ys,
        color="black",
        linestyle="--",
        linewidth=1.3,
        zorder=2,
        label="Pareto front",
    )
    return front


def plot_sweep(
    ax, cases, label, marker, color, annotate=lambda c: f"{c['co2_price']:.0f}"
):
    ax.scatter(
        [c["co2"] for c in cases],
        [c["lcoh"] for c in cases],
        label=label,
        marker=marker,
        s=70,
        color=color,
        zorder=3,
    )
    if annotate is None:  # caller places the labels itself
        return
    for c in cases:
        ax.annotate(
            annotate(c),
            (c["co2"], c["lcoh"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )


# -------------------------------------------------------------------------
# Plot 1: reference anchors only
# -------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(9, 6))
plot_anchors(ax1, annotate=True)
style_pareto(ax1)
fig1.tight_layout()

# -------------------------------------------------------------------------
# Plot 2: anchors + heat pump + storage optimization search (6 lambda)
# -------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 6))
plot_anchors(ax2, annotate=True)
plot_hp_storage_opt(ax2)
style_pareto(ax2)
fig2.tight_layout()

# -------------------------------------------------------------------------
# Plot 3: anchors + HP+storage + full system optimization search (6 lambda)
# -------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(9, 6))
plot_anchors(ax3, annotate=True)
plot_hp_storage_opt(ax3, annotate=False, hollow=True)
plot_sweep(
    ax3,
    full_sweep,
    "heat pump + storage + gas boiler (multi-objective)",
    "D",
    "#7A9A01",
    annotate=None,
)
front = plot_pareto_line(
    ax3,
    [boiler_only, hp_only, hp_storage_opt] + list(full_sweep),
    extend_right=True,
)

# Every marker gets a sizing label. The high-lambda designs and the
# single-objective ring are packed into a few hundred tCO2, far too close for
# inline labels, so those go into an aligned column with leader lines. The
# lambda=0 point stands well clear of the rest and is labelled in place.
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
        arrowprops=dict(
            arrowstyle="-", color="0.6", linewidth=0.6, shrinkA=0, shrinkB=4
        ),
        # Opaque backing: the iso-cost line below runs straight through this
        # column and would otherwise strike a label out.
        bbox=dict(
            boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"
        ),
    )

annotate_case(ax3, next(c for c in full_sweep if c["co2_price"] == 0))

style_pareto(ax3, XLIM3, YLIM3)
fig3.tight_layout()

# -------------------------------------------------------------------------
# Summary table over every solved case
# -------------------------------------------------------------------------
for fig, name in (
    (fig1, "pareto_1_anchors"),
    (fig2, "pareto_2_hp_storage"),
    (fig3, "pareto_3_full"),
):
    fig.savefig(PLOT_DIR / f"{name}.png", dpi=150)

df = pd.DataFrame([boiler_only, hp_only, hp_storage_opt] + full_sweep)
summary_cols = [
    "case_label",
    "co2_price",
    "cap_gas_boiler_mw",
    "cap_heat_pump_mw",
    "cap_storage_mwh",
    "lcoh",
    "co2",
]
print("\nSummary:")
print(df[summary_cols].round(3).to_string(index=False))

plt.show()
