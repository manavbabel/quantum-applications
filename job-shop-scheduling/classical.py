import math
import time
from bisect import bisect_right, insort
from collections import defaultdict
from graphlib import TopologicalSorter
from itertools import combinations, islice, product

import gurobipy as gp
from gurobipy import GRB

from formulation import build_formulation, decode
from problem import Problem
from solver import Solver, SolverUnavailable


class GreedySolver(Solver):
    name = "greedy"

    def _solve(self, problem):
        t0 = time.perf_counter()

        # each task's priority: the longest chain of durations from its start to the end, itself included
        # worked out children first, so each task is visited once
        chain = [0] * len(problem)
        for i in TopologicalSorter(dict(enumerate(problem.children))).static_order():
            chain[i] = problem.tasks[i][1] + max((chain[c] for c in problem.children[i]), default=0)

        order = TopologicalSorter(dict(enumerate(problem.dependencies)))
        order.prepare()
        start_times = [0] * len(problem)
        # each machine's busy intervals, sorted; as they never overlap, their ends are sorted too
        busy = defaultdict(list)
        while order.is_active():
            ready = sorted(order.get_ready(), key=lambda i: chain[i], reverse=True)
            for i in ready:
                machine, duration, dependencies = problem.tasks[i]
                start = max(
                    (start_times[p] + problem.tasks[p][1] for p in dependencies), default=0
                )
                # skip the intervals over by the time the parents finish, then step past each one
                # in the way until the task fits in a gap
                intervals = busy[machine]
                k = bisect_right(intervals, start, key=lambda interval: interval[1])
                while k < len(intervals) and intervals[k][0] < start + duration:
                    start = max(start, intervals[k][1])
                    k += 1
                insort(intervals, (start, start + duration))
                start_times[i] = start
            order.done(*ready)
        t1 = time.perf_counter()
        return start_times, t1 - t0, 1, {}, None

    def estimate(self, problem):
        return {"time": "very fast"}

class ExactSolver(Solver):
    name="exact"
    MAX_COMBINATIONS = 5_000_000
    def _solve(self, problem):
        windows = problem.windows()

        # check size is below threshold
        if (total:=math.prod([len(w) for w in windows])) > self.MAX_COMBINATIONS:
            raise SolverUnavailable(
                f"{total:,} combinations exceeds MAX_COMBINATIONS={self.MAX_COMBINATIONS:,}"
            )

        t0 = time.perf_counter()
        candidates = map(list, product(*windows))
        best = min(filter(problem.is_feasible, candidates), key=problem.makespan, default=None)
        t1 = time.perf_counter()

        metadata = {
            "calls_unit": "combinations",
            "combinations": total,
            "variables": sum(len(w) for w in windows),
            "couplings": 0,
            "proven_optimal": True,  # every combination was checked
        }

        return best, t1-t0, total, metadata, None

    # every combination of start times in the windows, priced by timing the first few of the same search
    def estimate(self, problem, trials=1000):
        windows = problem.windows()
        total = math.prod(len(w) for w in windows)

        checked = 0
        t0 = time.perf_counter()
        for candidate in islice(product(*windows), trials):
            checked += 1
            if problem.is_feasible(list(candidate)):
                problem.makespan(list(candidate))
        per_combination = (time.perf_counter() - t0) / checked

        return {
            "combinations": total,
            "within_limit": total <= self.MAX_COMBINATIONS,
            "expected_time": total * per_combination,  # seconds
        }

# solves either the MILP directly, or the QUBO to see how the formulation affects things
class GurobiSolver(Solver):
    name = "gurobi"

    def __init__(self, time_limit=None, threads=None):
        self.time_limit = time_limit
        self.threads = threads

    def _solve(self, problem:Problem, method="milp"):
        if method == "milp":
            return self._solve_milp(problem)
        elif method == "qubo":
            return self._solve_qubo(problem)
        else:
            raise ValueError(f"method must be 'milp' or 'qubo', got {method!r}")

    def estimate(self, problem:Problem, method="milp"):
        return {"time": "very fast"}

    # a model with this solver's limits; Gurobi rejects None, so unset limits keep its defaults
    def _model(self, name):
        model = gp.Model(name)
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        if self.threads is not None:
            model.Params.Threads = self.threads
        return model

    # optimise and return the time taken
    # the size-limited pip licence refuses big models, which is a limit of the set-up rather than a bug
    def _optimize(self, model):
        t0 = time.perf_counter()
        try:
            model.optimize()
        except gp.GurobiError as exc:
            if exc.errno == GRB.Error.SIZE_LIMIT_EXCEEDED:
                raise SolverUnavailable(f"gurobi: {exc}") from exc
            raise
        return time.perf_counter() - t0

    def _solve_milp(self, problem):
        windows = problem.windows()
        horizon = problem.horizon

        model = self._model("job_shop_milp")

        # start times, bounded by each task's window
        start = [
            model.addVar(lb=w.start, ub=w.stop - 1, vtype=GRB.INTEGER, name=f"s{i}")
            for i, w in enumerate(windows)
        ]
        makespan = model.addVar(lb=problem.lower_bound, ub=horizon, vtype=GRB.INTEGER, name="makespan")

        # the makespan is at least every childless task's finish time
        # every other task finishes before one of those starts, so needs no constraint of its own
        for i, (_, duration, _) in enumerate(problem.tasks):
            if not problem.children[i]:
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

        tts = self._optimize(model)

        metadata = {
            "model status": model.Status
        }

        # stopped (e.g. by the time limit) before finding anything
        if model.SolCount == 0:
            return None, tts, 1, metadata, None

        metadata.update({
            "model": "milp",
            "proven_optimal": model.Status == GRB.OPTIMAL,
            "nodes": int(model.NodeCount),
            "variables": model.NumVars,
            "order_variables": len(order),
        })
        return [round(s.X) for s in start], tts, 1, metadata, None

    def _solve_qubo(self, problem):
        formulation = build_formulation(problem)

        model = self._model("job_shop_qubo")

        # extract the variables
        x = [model.addVar(vtype=GRB.BINARY, name=f"x{n}") for n in range(formulation.num_variables)]

        # add the objective
        # which has the constraints included as penalty terms
        objective = gp.quicksum(coefficient * x[a] * x[b] for (a, b), coefficient in formulation.Q.items())
        model.setObjective(objective + formulation.offset, GRB.MINIMIZE)

        tts = self._optimize(model)

        metadata = {
            "model status": model.Status
        }

        # stopped (e.g. by the time limit) before finding anything
        if model.SolCount == 0:
            return None, tts, 1, metadata, None

        bits = [round(variable.X) for variable in x]
        metadata.update({
            "model": "qubo",
            "proven_optimal": model.Status == GRB.OPTIMAL,
            "nodes": int(model.NodeCount),
            "energy": model.ObjVal,
        })
        return decode(bits, formulation), tts, 1, metadata, None
