"""Minimal reproduction: CBC reports "optimal" with a valid objective
value, but pyomo has no solved value for several of the model's
variables afterward. Pure pyomo -- no other package involved.

Environment: pyomo 6.10.1, CBC 2.10.11 (also confirmed on CBC 2.10.13).

Expected: after a successful solve, every variable has a value.
Actual: `termination_condition` is "optimal", CBC's own log reports a
valid objective, but every variable in this model comes back with
`.value is None` -- as if no solution had been loaded at all.

Confirmed via scipy.optimize.linprog (an unrelated LP solver, HiGHS
backend): the constraint `content[1] >= MIN_STORAGE_LEVEL` is genuinely
infeasible by about 2e-7 given the exact capacities below -- CBC's
default feasibility tolerance is 1e-7, so this sits almost exactly on
that boundary. Tightening the tolerance below the gap makes CBC correctly
report infeasible; loosening it comfortably above the gap makes it solve
cleanly with no missing values. Only the default tolerance, which just
barely accepts the point, produces this failure:

    solver.solve(m, options={"presolve": "off", "primalTolerance": "1e-9"})
        -> correctly reports infeasible
    solver.solve(m, options={"presolve": "off", "primalTolerance": "1e-5"})
        -> solves cleanly, every variable has a value
    solver.solve(m, options={"presolve": "off"})  # CBC's default, 1e-7
        -> the failure below

Originally found via oemof.solph (an energy-system modelling package
built on pyomo); this script reproduces it with no other package
involved, isolating it to pyomo's CBC solver interface (or CBC itself).

Trigger conditions, found by simplifying the original model:
    - A recursive state variable (`content`) chained across timesteps by
      equality constraints, together with a separate auxiliary variable
      (`storage_losses`) also defined by its own equality constraint.
      Folding the loss directly into the recursion (one constraint
      instead of two) does NOT reproduce it -- the extra variable/
      constraint appears to be necessary.
    - The floor constraint on `content[1]` must be genuinely binding.
      A trivial floor (e.g. 0) does not reproduce it.
    - Extremely sensitive to the exact input values: changing only
      DEMAND[0] by as little as 0.0001 makes it disappear completely,
      while changing DEMAND[1] by as much as 1.0 has no effect. Also
      disappears if the capacities below are rounded to fewer decimal
      places.
"""

import pyomo.environ as po

DEMAND = [13.897, 16.26]
SOURCE_CAPACITY = 4.419 * 1.002 + 10.649 * 1.002
STORAGE_CAPACITY = 187.176 * 1.002
INITIAL_STORAGE_LEVEL = 0.06343767352673378
MIN_STORAGE_LEVEL = 13.086996442132152  # binding; 0.0 does not reproduce it
LOSS_RATE = 0.001
COST = 1.1

storage_flow_cap = STORAGE_CAPACITY / 24
content0 = INITIAL_STORAGE_LEVEL * STORAGE_CAPACITY

m = po.ConcreteModel()
m.T = po.RangeSet(0, 1)   # intervals
m.TP = po.RangeSet(0, 2)  # timepoints (interval boundaries)

m.source = po.Var(m.T, within=po.NonNegativeReals, bounds=(0, SOURCE_CAPACITY))
m.charge = po.Var(m.T, within=po.NonNegativeReals, bounds=(0, storage_flow_cap))
m.discharge = po.Var(m.T, within=po.NonNegativeReals, bounds=(0, storage_flow_cap))
m.content = po.Var(m.TP, within=po.NonNegativeReals, bounds=(0, STORAGE_CAPACITY))
m.storage_losses = po.Var(m.T, within=po.NonNegativeReals)

m.balance = po.Constraint(
    m.T, rule=lambda m, t: m.source[t] - m.charge[t] + m.discharge[t] == DEMAND[t]
)
m.losses_def = po.Constraint(
    m.T, rule=lambda m, t: m.storage_losses[t] == m.content[t] * LOSS_RATE
)
m.content_recursion = po.Constraint(
    m.T,
    rule=lambda m, t: m.content[t + 1]
    == m.content[t] - m.storage_losses[t] + m.charge[t] - m.discharge[t],
)
m.initial_content = po.Constraint(expr=m.content[0] == content0)
m.min_level = po.Constraint(expr=m.content[1] >= MIN_STORAGE_LEVEL)

m.obj = po.Objective(expr=COST * sum(m.source[t] for t in m.T), sense=po.minimize)

solver = po.SolverFactory("cbc", solver_io="lp")
#results = solver.solve(m, tee=False)  # CBC's default, 1e-7
# -> optimal, but 11/11 variables have no value

# results = solver.solve(m, options={"primalTolerance": "1e-9"})
# -> correctly reports infeasible

results = solver.solve(m, options={"primalTolerance": "1e-5"})  # looser than the gap
# -> solves cleanly, every variable has a value

print(f"termination_condition: {results.solver.termination_condition}")
all_vars = list(m.component_data_objects(po.Var, active=True))
none_vars = [v for v in all_vars if v.value is None]
print(f"variables with no value: {len(none_vars)} / {len(all_vars)}")
