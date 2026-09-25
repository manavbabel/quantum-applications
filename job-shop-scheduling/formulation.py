from collections import Counter, defaultdict
from itertools import combinations
from typing import NamedTuple

from qiskit_addon_opt_mapper import OptimizationProblem

from problem import Problem


# QUBO formulation
# energy is sum over a,b of Q[a,b]x_ax_b + offset
# for a feasible solution, energy is makespan
class Formulation(NamedTuple):
    horizon: int  # the latest start time considered
    penalty: int  # the cost of breaking any one constraint
    variables: list  # variables[n] is the (task, time) pair that index n stands for
    blocks: list  # blocks[i] is the time-ordered variable indices for task i
    Q: dict  # maps (a, b) variable-index pairs to coefficients, with a == b for linear terms
    offset: int  # constant energy term, so a feasible energy is exactly the makespan

    @property
    def num_variables(self):
        return len(self.variables)

    def describe(self):
        print("QUBO formulation")
        print(f" * variables       : {self.num_variables}")
        print(f" * linear terms    : {sum(a == b for a, b in self.Q)}")
        print(f" * quadratic terms : {sum(a != b for a, b in self.Q)}")
        print(f" * horizon         : {self.horizon}")
        print(f" * penalty         : {self.penalty}")
        print(f" * offset          : {self.offset}")

def build_formulation(problem:Problem, for_qaoa:bool=False):
    problem = add_makespan_task(problem)
    penalty = problem.horizon + 1

    # get the list of possible start times for each task
    windows = problem.windows()

    # variables indexed by task and time
    variables = [(i, k) for i in range(len(problem)) for k in windows[i]]

    # map each variable back to its index
    index = {variable: n for n, variable in enumerate(variables)}

    # group the variable indices by task
    blocks = [[index[(i, k)] for k in windows[i]] for i in range(len(problem))]

    Q = defaultdict(int)

    # the objective is the makespan task's start time
    # give each of its variables a linear cost equal to the time it stands for
    # so the one that is set, at time k, contributes exactly k to the energy
    for n in blocks[-1]:  # the makespan task is the last one
        Q[(n, n)] += variables[n][1]

    # constraint: each task starts exactly once
    # penalty * (sum_k x_ik - 1)^2, expanded using x^2 = x for binary x
    # the +1 from each square is constant, and is collected in the offset below
    # ignored if using QAOA since this is accomplished via a hamming-weight-preserving mixer
    if not for_qaoa:
        for block in blocks:
            for n in block:
                Q[(n, n)] -= penalty
            for a, b in combinations(block, 2):
                Q[(a, b)] += 2 * penalty

    # constraints: parents finish before their children start, and one task per machine at a time
    # penalise each pair of start times that would break either
    for a, b in clashes(problem, windows):
        Q[(index[a], index[b])] += penalty

    # the offset is the +1 per task left over from the one-hot squares
    return Formulation(
        problem.horizon,
        penalty,
        variables,
        blocks,
        dict(Q),
        0 if for_qaoa else penalty * len(problem),
    )

# the pairs of (task, time) variables that cannot both be set
# a pair that breaks both constraints comes up twice, so is penalised twice
def clashes(problem:Problem, windows):
    # every parent finishes before its child starts: pair each child start before the parent finishes
    for i, (_, _, dependencies) in enumerate(problem.tasks):
        for p in dependencies:
            for kp in windows[p]:
                for ki in windows[i]:
                    if ki < kp + problem.tasks[p][1]:  # i starts before p finishes
                        yield (p, kp), (i, ki)

    # one task per machine at a time: pair each two start times whose intervals overlap
    for i, j in combinations(range(len(problem)), 2):
        if problem.tasks[i][0] == problem.tasks[j][0]:
            for ki in windows[i]:
                for kj in windows[j]:
                    if ki < kj + problem.tasks[j][1] and kj < ki + problem.tasks[i][1]:
                        yield (i, ki), (j, kj)

# the same problem as a constrained model in qiskit-addon-opt-mapper, Qiskit's supported modelling layer,
# for its converters, translators (e.g. to docplex, to write an LP file) and classical reference solvers
# the variables come in the same order as build_formulation's, variable n named x{task}_{time}
# OptimizationProblemToQubo(penalty=formulation.penalty) turns it into exactly formulation.Q and offset,
# but takes seconds on ft06 where build_formulation takes milliseconds, so the solvers use that instead
def build_model(problem:Problem, for_qaoa:bool=False):
    problem = add_makespan_task(problem)
    windows = problem.windows()
    model = OptimizationProblem(name="job_shop")
    for i, window in enumerate(windows):
        for k in window:
            model.binary_var(f"x{i}_{k}")

    # the objective is the makespan task's start time
    model.minimize(linear={f"x{len(problem) - 1}_{k}": k for k in windows[-1]})

    # each task starts exactly once, unless the QAOA mixer sees to that
    if not for_qaoa:
        for i, window in enumerate(windows):
            model.linear_constraint({f"x{i}_{k}": 1 for k in window}, "==", 1)

    # no two clashing start times are both set
    for (i, ki), (j, kj) in clashes(problem, windows):
        model.linear_constraint({f"x{i}_{ki}": 1, f"x{j}_{kj}": 1}, "<=", 1)
    return model

# add zero-duration makespan task whose parents are every childless task, on its own machine
def add_makespan_task(problem:Problem):
    sinks = tuple(i for i in range(len(problem)) if not problem.children[i])
    return Problem(list(problem.tasks) + [(problem.num_machines, 0, sinks)])

# substitute x = (1-s)/2 to get the Ising energy, up to a constant
# energy = sum h[i] s_i + sum J[i, j] s_i s_j + constant
def qubo_to_ising(Q):
    h, J = defaultdict(float), defaultdict(float)
    for (a, b), coefficient in Q.items():
        if a == b:
            h[a] -= coefficient / 2
        else:
            J[(a, b)] += coefficient / 4
            h[a] -= coefficient / 4
            h[b] -= coefficient / 4
    return dict(h), dict(J)

# convert a bitstring into a set of start times
# note we assume qubit 0 is leftmost (opposite to Qiskit)
# a task whose block is not exactly one-hot has no start time, so decodes to None, which is infeasible
def decode(bits, formulation):
    start_times = []
    for block in formulation.blocks:
        on = [n for n in block if int(bits[n])]
        start_times.append(formulation.variables[on[0]][1] if len(on) == 1 else None)
    return start_times


# from a set of samples (bits in variable order), find the feasible schedule with the lowest makespan
# ranked by the real tasks' own makespan, not by energy: the makespan task only steers the
# sampler towards short schedules, so a sample that leaves it unset or late still holds a valid schedule
# weights say how often each sample came up (reads, shot counts or probabilities)
# returns the best start times (None if nothing was feasible) and a tally of weight by makespan,
# with None for the infeasible samples
def best_feasible(samples, weights, formulation, problem):
    best, best_makespan, makespans = None, None, Counter()
    for bits, weight in zip(samples, weights):
        start_times = decode(bits, formulation)[: len(problem)]
        makespan = problem.makespan(start_times) if problem.is_feasible(start_times) else None
        makespans[makespan] += float(weight)
        if makespan is not None and (best is None or makespan < best_makespan):
            best, best_makespan = start_times, makespan
    return best, dict(makespans)
