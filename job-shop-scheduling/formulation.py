# given a problem, formulate it into a format the solvers can take
# specifically a QUBO

from collections import defaultdict
from itertools import combinations
from typing import NamedTuple

from problem import Problem


# one-hot QUBO formulation
# energy(x) = sum over (a, b) of Q[a, b] x_a x_b  +  offset
# which gives the makespan for a feasible solution
class Formulation(NamedTuple):
    tasks: list  # tasks including a zero-duration makespan task
    horizon: int  # the latest start time considered
    penalty: int  # the cost of breaking any one constraint
    windows: list  # windows[i] is the possible start times for task i
    variables: list  # variables[n] is the (task, time) pair that index n stands for
    index: dict  # the reverse of variables: (task, time) -> index
    blocks: list  # blocks[i] is the time-ordered variable indices for task i
    Q: dict  # maps (a, b) variable-index pairs to coefficients, with a == b for linear terms
    offset: int  # constant energy term, so a feasible energy is exactly the makespan

    @property
    def num_variables(self):
        return len(self.variables)

    @property
    def num_couplings(self):
        return sum(1 for a, b in self.Q if a != b)


# add a zero-duration makespan task
# depends on every childless task
# on its own machine
# the start time of this task is the makespan which we want to minimise
# keeps the objective linear
def add_makespan_task(problem):
    sinks = tuple(i for i in range(len(problem)) if not problem.children[i])
    return Problem(list(problem.tasks) + [(problem.num_machines, 0, sinks)])


# compute the QUBO and return a Formulation
# objective spans [0, horizon] and penalty is horizon + 1
# so all infeasible solutions are worse than all feasible ones
def build_formulation(problem):
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
    for block in blocks:
        for n in block:
            Q[(n, n)] -= penalty
        for a, b in combinations(block, 2):
            Q[(a, b)] += 2 * penalty

    # constraint: every parent finishes before its child starts
    # penalise each pair of start times that would break this
    for i, (_, _, dependencies) in enumerate(problem.tasks):
        for p in dependencies:
            for kp in windows[p]:
                for ki in windows[i]:
                    if ki < kp + problem.tasks[p][1]:  # i starts before p finishes
                        Q[(index[(p, kp)], index[(i, ki)])] += penalty

    # constraint: one task per machine at a time
    # penalise each pair of start times whose intervals overlap
    for i, j in combinations(range(len(problem)), 2):
        if problem.tasks[i][0] == problem.tasks[j][0]:
            for ki in windows[i]:
                for kj in windows[j]:
                    if ki < kj + problem.tasks[j][1] and kj < ki + problem.tasks[i][1]:
                        Q[(index[(i, ki)], index[(j, kj)])] += penalty

    # the offset is the +1 per task left over from the one-hot squares
    return Formulation(
        problem.tasks,
        problem.horizon,
        penalty,
        windows,
        variables,
        index,
        blocks,
        dict(Q),
        penalty * len(problem),
    )


# substitute x = (1-s)/2 to get the Ising energy
# energy = sum h[i] s_i + sum J[i, j] s_i s_j + offset
def qubo_to_ising(Q, offset=0):
    h, J = defaultdict(float), defaultdict(float)
    for (a, b), coefficient in Q.items():
        if a == b:
            h[a] -= coefficient / 2
            offset += coefficient / 2
        else:
            J[(a, b)] += coefficient / 4
            h[a] -= coefficient / 4
            h[b] -= coefficient / 4
            offset += coefficient / 4
    return dict(h), dict(J), offset


# calculate the QUBO energy
def energy(bits, formulation):
    return (
        sum(
            coefficient * int(bits[a]) * int(bits[b])
            for (a, b), coefficient in formulation.Q.items()
        )
        + formulation.offset
    )


# convert a Qiskit counts key into a list of bits
# Qiskit keys are little-endian: qubit 0 is the rightmost character, so reverse it
def counts_to_bits(key):
    return [int(bit) for bit in reversed(key)]


# convert a bitstring into a set of start times
# note we assume qubit 0 is leftmost (opposite to Qiskit)
# a task whose block is not one-hot decodes to None, or to its last set bit
def decode(bits, formulation):
    start_times = [None] * len(formulation.tasks)
    for n, (i, k) in enumerate(formulation.variables):
        if int(bits[n]):
            start_times[i] = k
    return start_times


# convert a set of start times into bits
# assume qubit 0 is leftmost
# the makespan task may be left off, in which case we work out when it starts
# useful for warm starts, and for checking the energy of a known schedule
def encode(start_times, formulation):
    if len(start_times) < len(formulation.tasks):
        # the makespan task starts once every other task has finished
        start_times = list(start_times) + [
            max(k + duration for k, (_, duration, _) in zip(start_times, formulation.tasks))
        ]
    bits = [0] * len(formulation.variables)
    for i, start in enumerate(start_times):
        bits[formulation.index[(i, start)]] = 1
    return bits
