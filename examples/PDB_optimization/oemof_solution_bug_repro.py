"""Minimal reproduction: oemof.solph raises "No value for uninitialized
VarData" on a successfully-solved model, using only public/documented
oemof.solph API (no direct pyomo model access at all).

Confirmed to reproduce identically on two independent environments:
    1. Linux, Python 3.12.3, oemof.solph 0.6.4, pyomo 6.10.1, CBC 2.10.11
       (built Jan 21 2024)
    2. macOS, Python 3.11.15, oemof.solph 0.6.4, pyomo 6.10.1, CBC 2.10.13
       (built May 12 2026)
Same package versions but a different OS, Python minor version, and CBC
build in each case -- same exact error, same line, same traceback shape.

Expected: model.solve() completes and results can be read.
Actual: model.solve() raises
    ValueError: No value for uninitialized VarData object flow[bus,storage,0]

EXPECTED SOLUTION (hand-derived and independently confirmed with a
completely different solver -- scipy.optimize.linprog's HiGHS backend,
unrelated to CBC/pyomo -- solving the same linear program from scratch):
    source[bus, 0]        = 15.098136200132151
    source[bus, 1]        =  8.445402
    storage charge[0]     =  1.2011362001321508
    storage discharge[0]  =  0.0
    storage charge[1]     =  0.0
    storage discharge[1]  =  7.814598
    storage_content[0]    = 11.897758000000001   (the given initial value)
    storage_content[1]    = 13.086996442132152   (exactly the min_storage_level floor)
    storage_content[2]    =  5.259311445690021
    objective              = 25.89789202014537
This is the unique optimum: there is only one cost-bearing variable
(the source), so the model charges the storage the bare minimum needed
to satisfy the floor at t=1, then discharges as much as its own flow
capacity allows in the second hour (discharging is free, so this is
always beneficial) to reduce how much the source needs to supply.

NOTABLE: SOURCE_CAPACITY as given below is 15.098135999999998, while the
solution above needs source[0] = 15.098136200132151 to satisfy the
storage floor -- i.e., the model as specified is actually INFEASIBLE by
about 2e-7 (confirmed: relaxing SOURCE_CAPACITY by 1e-6 makes it cleanly,
unambiguously feasible with the exact solution above).

Confirmed directly, via oemof's own documented `cmdline_options` pass-
through to CBC:
    model.solve(solver="cbc", cmdline_options={"primalTolerance": "1e-9"})
        -> correctly reports INFEASIBLE (matches the independent scipy
           check above -- CBC's own default tolerance is 1e-7, and 1e-9
           is tighter than the ~2e-7 gap)
    model.solve(solver="cbc", cmdline_options={"primalTolerance": "1e-5"})
        -> solves CLEANLY, no crash at all (1e-5 is looser than the gap)
    model.solve(solver="cbc")  # CBC's default tolerance, 1e-7
        -> the buggy degenerate solution shown above
The default tolerance (1e-7) sits almost exactly at the same order of
magnitude as the model's ~2e-7 infeasibility gap -- just barely enough
for CBC to accept the point as feasible, landing on a genuinely
borderline/degenerate vertex solution that its own solution output does
not fully report back through pyomo. Comfortably tighter or looser
tolerances both avoid the issue entirely, which is about as clean a
confirmation of a tolerance-boundary numerical issue as this could get.

Other trigger conditions, found by simplifying a larger real model down
to this:
    - A GenericStorage with a `min_storage_level` that is genuinely
      binding at the optimum. The identical model with a trivial (always
      satisfied) min_storage_level does not raise.
    - Extremely sensitive to the exact input values: changing only the
      FIRST demand value below by as little as 0.0001 makes the error
      disappear completely, while changing only the LAST demand value by
      as much as 1.0 has no effect. Also disappears if the capacities
      below are rounded to fewer decimal places.

Related prior reports (not the same bug, but corroborating context):
    - Pyomo issue #3855: the identical "No value for uninitialized
      VarData" error, in a completely unrelated code path (MindtPy's
      short-circuit for pure NLP/LP subproblems) -- confirms this exact
      error text is a generic pyomo symptom for "a solver plugin did not
      load a value," not something specific to this model.
    - Cbc.jl issue #211: documents CBC versions differing in how they
      propagate primalTolerance/dualTolerance for LPs, and its own CLI
      trace includes the exact phrase "Presolved model was optimal, full
      model needs cleaning up" that also appeared in our own captured
      CBC output while investigating this -- suggesting CBC's presolve/
      postsolve reconciliation step is a known, pre-existing source of
      numerically-inconsistent results in other projects too.
    - oemof-solph issue #1056 ("Results processing should check if
      optimisation succeeded") tracks a related but distinct concern:
      oemof.solph v0.6.0 added a check that raises if the solver's
      reported status is non-optimal (Model.solve(allow_nonoptimal=...)).
      That check does not catch this bug, because CBC's reported status
      genuinely is "optimal" here -- the problem is that status itself is
      wrong (see the tolerance-boundary analysis above), not that oemof
      failed to check it.
"""

import pandas as pd
import oemof.solph as solph

DEMAND = [13.897, 16.26]
SOURCE_CAPACITY = 4.419 * 1.002 + 10.649 * 1.002  # exactly tight, see above
STORAGE_CAPACITY = 187.176 * 1.002
INITIAL_STORAGE_LEVEL = 0.06343767352673378
MIN_STORAGE_LEVEL_MWH = (
    13.086996442132152  # binding; 0.0 does not reproduce it
)

n = len(DEMAND)
idx = pd.date_range("2020-01-01", periods=n + 1, freq="h")
demand = pd.Series(DEMAND, index=idx[:-1])

es = solph.EnergySystem(timeindex=idx, infer_last_interval=False)
bus = solph.Bus(label="bus")
es.add(bus)

source = solph.components.Source(
    label="source",
    outputs={
        bus: solph.flows.Flow(
            nominal_capacity=SOURCE_CAPACITY, variable_costs=1.1
        )
    },
)
sink = solph.components.Sink(
    label="sink",
    inputs={bus: solph.flows.Flow(nominal_capacity=20.0, fix=demand / 20.0)},
)
storage = solph.components.GenericStorage(
    label="storage",
    nominal_capacity=STORAGE_CAPACITY,
    inputs={bus: solph.flows.Flow(nominal_capacity=STORAGE_CAPACITY / 24)},
    outputs={bus: solph.flows.Flow(nominal_capacity=STORAGE_CAPACITY / 24)},
    balanced=False,
    loss_rate=0.001,
    initial_storage_level=INITIAL_STORAGE_LEVEL,
    min_storage_level=[0.0, MIN_STORAGE_LEVEL_MWH / STORAGE_CAPACITY, 0.0],
)
es.add(source, sink, storage)

model = solph.Model(es)
model.solve(solver="cbc", solve_kwargs={"tee": False})

results = solph.processing.results(model)
print(solph.views.node(results, "bus")["sequences"])
