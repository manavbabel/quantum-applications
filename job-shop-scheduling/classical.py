# all classical solvers

from itertools import combinations, product
from math import prod

import gurobipy as gp
from gurobipy import GRB

from base import Solver, SolverUnavailable, build_formulation
from formulation import decode

# set a maximum level of in-window combos above which exhaustive search is not practical
MAX_COMBINATIONS = 5_000_000

# wrap the problem's own greedy solver
# always feasible
# optimal when the critical path dominates
class GreedySolver(Solver):
    name = "greedy"
    uses_qubo = False

    def _solve(self, problem):
        return problem.greedy(), 1, {"calls_unit": "passes"}

# exhausative search
# exponential, but gets true optimum
class ExactSolver(Solver):

    name = "exact"
    uses_qubo = False

    def _solve(self, problem):
        windows = problem.windows()

        # check size is below threshold
        if (total:=prod([len(w) for w in windows])) > MAX_COMBINATIONS:
            raise SolverUnavailable(
                f"{total:,} combinations exceeds MAX_COMBINATIONS={MAX_COMBINATIONS:,}"
            )

        best = None
        checked = 0
        for candidate in product(*windows):
            checked += 1
            if problem.is_feasible(list(candidate)):
                value = problem.makespan(list(candidate))
                if best is None or value < best[0]:
                    best = (value, list(candidate))
        metadata = {
            "calls_unit": "candidates",
            "combinations": total,
            "variables": sum(len(w) for w in windows),
            "couplings": 0,
        }
        return (best[1] if best else None), checked, metadata

# gurobi solver
# best heuristic solver
# can either solve the MILP directly
# or solve the QUBO to see how the formulation affects things
class GurobiSolver(Solver):

    def __init__(self, model="milp", time_limit=None, threads=None):
        if model not in ("milp", "qubo"):
            raise ValueError("model must be 'milp' or 'qubo'")

        self.model = model
        self.time_limit = time_limit
        self.threads = threads

        self.name = f"gurobi/{model}"
        self.uses_qubo = self.model=="qubo"

    def _apply_model_params(self, model):
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        if self.threads is not None:
            model.Params.Threads = self.threads

    def _solve(self, problem):
        if self.model == "qubo":
            return self._solve_qubo(problem)
        else:
            return self._solve_milp(problem)

    # disjunctive MILP: integer start times, and a binary per same-machine pair choosing which goes first
    def _solve_milp(self, problem):
        windows = problem.windows()
        horizon = problem.horizon

        model = gp.Model("job_shop_milp")
        self._apply_model_params(model)

        # start times, bounded by each task's window
        start = [
            model.addVar(lb=w.start, ub=w.stop - 1, vtype=GRB.INTEGER, name=f"s{i}")
            for i, w in enumerate(windows)
        ]
        makespan = model.addVar(lb=problem.lower_bound, ub=horizon, vtype=GRB.INTEGER, name="makespan")

        # the makespan is at least every task's finish time
        for i, (_, duration, _) in enumerate(problem.tasks):
            model.addConstr(makespan >= start[i] + duration)

        # constraint: every parent finishes before its child starts
        for i, dependencies in enumerate(problem.dependencies):
            for p in dependencies:
                model.addConstr(start[i] >= start[p] + problem.tasks[p][1])

        # constraint: one task per machine at a time
        # order[i, j] = 1 means i finishes before j starts, 0 means j finishes before i starts
        # every task finishes by the horizon, so the horizon is a big enough M to switch off the unused side
        order = {}
        for i, j in combinations(range(len(problem)), 2):
            if problem.tasks[i][0] == problem.tasks[j][0]:
                y = order[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y{i}_{j}")
                model.addConstr(start[i] + problem.tasks[i][1] <= start[j] + horizon * (1 - y))
                model.addConstr(start[j] + problem.tasks[j][1] <= start[i] + horizon * y)

        model.setObjective(makespan, GRB.MINIMIZE)
        model.optimize()

        metadata = {
            "calls_unit": "model solves",
            "status": model.Status
        }

        # if failure
        if model.SolCount == 0:
            print("solve failed")
            return None, 1, metadata

        metadata.update({
            "model": "milp",
            "proven_optimal": model.Status == GRB.OPTIMAL,
            "objective": model.ObjVal,
            "nodes": int(model.NodeCount),
            "variables": model.NumVars,
            "order_variables": len(order),
        })
        return [round(s.X) for s in start], 1, metadata

    def _solve_qubo(self, problem):
        formulation = build_formulation(problem)

        model = gp.Model("job_shop_qubo")
        self._apply_model_params(model)

        # extract the variables
        x = [model.addVar(vtype=GRB.BINARY, name=f"x{n}") for n in range(formulation.num_variables)]

        # add the objective
        # which has the constraints included as penalty terms
        objective = gp.quicksum(coefficient * x[a] * x[b] for (a, b), coefficient in formulation.Q.items())
        model.setObjective(objective + formulation.offset, GRB.MINIMIZE)

        model.optimize()

        metadata = {
            "calls_unit": "model solves",
            "status": model.Status
        }

        # if failure
        if model.SolCount == 0:
            print("solve failed")
            return None, 1, metadata

        bits = [round(variable.X) for variable in x]
        metadata.update({
            "model": "qubo",
            "proven_optimal": model.Status == GRB.OPTIMAL,
            "energy": model.ObjVal,
            "nodes": int(model.NodeCount),
        })
        return decode(bits, formulation), 1, metadata

