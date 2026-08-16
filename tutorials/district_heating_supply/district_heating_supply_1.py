# %%[sec_1_start]
import os

import pandas as pd
import matplotlib.pyplot as plt
import oemof.solph as solph
from helpers import LCOH

file_path = os.path.dirname(__file__)
data_dir = "data"
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

filename = "input_data.csv"
data = pd.read_csv(
    os.path.join(file_path, data_dir, filename), 
    sep=";", 
    index_col=0, 
    parse_dates=True
)

district_heating_system = solph.EnergySystem(
    timeindex=data.index,
    infer_last_interval=True,
)

heat_bus = solph.Bus(label="heat network")
gas_bus = solph.Bus(label="gas network")

district_heating_system.add(heat_bus, gas_bus)

gas_source = solph.components.Source(
    label="gas source",
    outputs={gas_bus: solph.flows.Flow(variable_costs=data["gas price"])},
)

heat_sink = solph.components.Sink(
    label="heat sink",
    inputs={
        heat_bus: solph.flows.Flow(
            nominal_capacity=data["heat demand"].max(),
            fix=data["heat demand"] / data["heat demand"].max(),
        )
    },
)

district_heating_system.add(heat_sink, gas_source)

gas_boiler = solph.components.Converter(
    label="gas boiler",
    inputs={gas_bus: solph.flows.Flow()},
    outputs={
        heat_bus: solph.flows.Flow(
            nominal_capacity=data["heat demand"].max(), variable_costs=1.10
        )
    },
    conversion_factors={heat_bus: 0.95},
)

district_heating_system.add(gas_boiler)

model = solph.Model(district_heating_system)
results = model.solve(solver="cbc", solve_kwargs={"tee": True})
flows = results["flow"]

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=[10, 6])
ax.bar(
    flows.index,
    flows[("gas boiler", "heat network")],
    label="gas boiler",
    color="#EC6707",
)
ax.legend(loc="upper right")
ax.grid(axis="y")
ax.set_ylabel("Hourly heat production in MWh")
plt.tight_layout()
filename = 'intro_tut_dhs_1_hourly_heat_production.pdf'
plt.savefig(os.path.join(plot_dir, filename))
plt.show()

spec_inv_gas_boiler = 50000
cap_gas_boiler = 20
var_cost_gas_boiler = 1.10

invest_cost = spec_inv_gas_boiler * cap_gas_boiler
operation_cost = (
    var_cost_gas_boiler * flows[("gas boiler", "heat network")].sum()
    + (data["gas price"] * flows[("gas network", "gas boiler")]).sum()
)
heat_produced = flows[("heat network", "heat sink")].sum()

lcoh = LCOH(invest_cost, operation_cost, heat_produced)
print(f"LCOH: {lcoh:.2f} €/MWh")
