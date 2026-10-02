import math
import time
from itertools import combinations, islice, product

import gurobipy as gp
from formulation import build_formulation, decode
from gurobipy import GRB
from results import SolverUnavailable, make_result

MAX_COMBINATIONS = 5_000_000


def solve_greedy(problem):
    t0 = time.perf_counter()
    start_times = problem.greedy_schedule()
    return make_result(problem, "greedy", start_times, time.perf_counter() - t0, calls=1)


# try every combination of start times in the windows, and keep the best feasible one
def solve_exact(problem, max_combinations=MAX_COMBINATIONS):
    windows = problem.windows()
    total = math.prod(len(w) for w in windows)
    if total > max_combinations:
        raise SolverUnavailable(f"{total:,} combinations exceeds max_combinations={max_combinations:,}")

    t0 = time.perf_counter()
    candidates = map(list, product(*windows))
    best = min(filter(problem.is_feasible, candidates), key=problem.makespan, default=None)
    tts = time.perf_counter() - t0

    metadata = {"calls_unit": "combinations", "proven_optimal": True}
    return make_result(problem, "exact", best, tts, calls=total, metadata=metadata)


# the number of combinations, and the expected time from timing the first `trials` of them
def estimate_exact(problem, trials=1000):
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
        "within_limit": total <= MAX_COMBINATIONS,
        "expected_time": total * per_combination,  # seconds
    }


# solves either the MILP directly, or the QUBO to see how the formulation affects things
def solve_gurobi(problem, method="milp", time_limit=None):
    model = gp.Model(f"job_shop_{method}")
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if method == "milp":
        start = add_milp(model, problem)
    elif method == "qubo":
        formulation = build_formulation(problem)
        x = add_qubo(model, formulation)
    else:
        raise ValueError(f"method must be 'milp' or 'qubo', got {method!r}")

    # the size-limited pip licence refuses big models, so report those as unavailable
    t0 = time.perf_counter()
    try:
        model.optimize()
    except gp.GurobiError as exc:
        if exc.errno == GRB.Error.SIZE_LIMIT_EXCEEDED:
            raise SolverUnavailable(f"gurobi: {exc}") from exc
        raise
    tts = time.perf_counter() - t0

    # stopped (e.g. by the time limit) before finding anything
    if model.SolCount == 0:
        start_times = None
    elif method == "milp":
        start_times = [round(s.X) for s in start]
    else:
        start_times = decode([round(v.X) for v in x], formulation)

    metadata = {"proven_optimal": model.Status == GRB.OPTIMAL, "nodes": int(model.NodeCount)}
    if method == "qubo" and model.SolCount:
        metadata["energy"] = model.ObjVal  # the makespan task's start time
    return make_result(problem, f"gurobi/{method}", start_times, tts, calls=1, metadata=metadata)


# the problem as a MILP: integer start times, and a binary order variable for each pair sharing a machine
# returns the start-time variables
def add_milp(model, problem):
    windows = problem.windows()
    horizon = problem.horizon

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
    # y = 1 means i finishes before j starts, 0 means j finishes before i starts
    # every task finishes by the horizon, so the horizon is a big enough M to switch off the unused side
    for i, j in combinations(range(len(problem)), 2):
        if problem.tasks[i][0] == problem.tasks[j][0]:
            y = model.addVar(vtype=GRB.BINARY, name=f"y{i}_{j}")
            model.addConstr(start[i] + problem.tasks[i][1] <= start[j] + horizon * (1 - y))
            model.addConstr(start[j] + problem.tasks[j][1] <= start[i] + horizon * y)

    model.setObjective(makespan, GRB.MINIMIZE)
    return start


# the QUBO as a binary quadratic program, with the constraints as penalty terms
# returns the binary variables
def add_qubo(model, formulation):
    x = [model.addVar(vtype=GRB.BINARY, name=f"x{n}") for n in range(formulation.num_variables)]
    objective = gp.quicksum(coefficient * x[a] * x[b] for (a, b), coefficient in formulation.Q.items())
    model.setObjective(objective + formulation.offset, GRB.MINIMIZE)
    return x
