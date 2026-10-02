from collections import Counter, defaultdict
from itertools import combinations
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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


def build_formulation(problem: Problem, for_qaoa: bool = False):
    problem = add_makespan_task(problem)

    # get the list of possible start times for each task
    windows = problem.windows()

    # the makespan task can't start before `earliest`, so we measure the objective from there
    # it then runs from 0 to horizon - earliest, so breaking a constraint only has to cost more than that
    earliest = windows[-1].start
    penalty = problem.horizon - earliest + 1

    # variables indexed by task and time
    variables = [(i, k) for i in range(len(problem)) for k in windows[i]]

    # map each variable back to its index
    index = {variable: n for n, variable in enumerate(variables)}

    # group the variable indices by task
    blocks = [[index[(i, k)] for k in windows[i]] for i in range(len(problem))]

    Q = defaultdict(int)

    # the objective is the makespan task's start time, less `earliest`
    # each of its variables gets a linear cost equal to its time, so the one set at time k adds k - earliest
    for n in blocks[-1]:  # the makespan task is the last one
        Q[(n, n)] += variables[n][1] - earliest

    # constraint: each task starts exactly once
    # penalty * (sum_k x_ik - 1)^2, expanded using x^2 = x for binary x
    # the +1 from each square is constant, and is collected in the offset below
    # left out for QAOA, where a Hamming-weight-preserving mixer enforces it
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

    # the offset adds back `earliest`, and the +1 per task left over from the one-hot squares
    offset = earliest + (0 if for_qaoa else penalty * len(problem))
    return Formulation(problem.horizon, penalty, variables, blocks, dict(Q), offset)


# the pairs of (task, time) variables that cannot both be set
# a pair that breaks both constraints comes up twice, so is penalised twice
def clashes(problem: Problem, windows):
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


# add zero-duration makespan task whose parents are every childless task, on its own machine
def add_makespan_task(problem: Problem):
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


# convert bits (bits[n] is variable n) into start times
# a task whose block isn't one-hot gets None, which is infeasible
def decode(bits, formulation):
    start_times = []
    for block in formulation.blocks:
        on = [n for n in block if int(bits[n])]
        start_times.append(formulation.variables[on[0]][1] if len(on) == 1 else None)
    return start_times


# the best feasible schedule among samples (bits in variable order), weighted by how often each came up
# samples are ranked by the real tasks' makespan, so one with the makespan task unset or late still counts
# returns the best start times (None if none were feasible) and the weight at each makespan (None for
# infeasible)
def best_feasible(samples, weights, formulation, problem):
    best, best_makespan, makespans = None, None, Counter()
    for bits, weight in zip(samples, weights):
        start_times = decode(bits, formulation)[: len(problem)]
        makespan = problem.makespan(start_times) if problem.is_feasible(start_times) else None
        makespans[makespan] += float(weight)
        if makespan is not None and (best is None or makespan < best_makespan):
            best, best_makespan = start_times, makespan
    return best, dict(makespans)


# the QUBO as a heatmap, one row and column per variable, grouped into each task's block of start times
# a pair's colour is its total coupling; the one-hot terms show as the blocks on the diagonal
def draw_qubo(formulation):
    matrix = np.zeros((formulation.num_variables, formulation.num_variables))
    for (a, b), coefficient in formulation.Q.items():
        matrix[a, b] += coefficient
        if a != b:
            matrix[b, a] += coefficient
    limit = np.abs(matrix).max()
    colours = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])

    figure, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    image = ax.imshow(matrix, cmap=colours, vmin=-limit, vmax=limit)
    figure.colorbar(image, ax=ax, label="coefficient", shrink=0.8)
    for block in formulation.blocks[1:]:
        ax.axhline(block[0] - 0.5, color="white", linewidth=1)
        ax.axvline(block[0] - 0.5, color="white", linewidth=1)
    centres = [(block[0] + block[-1]) / 2 for block in formulation.blocks]
    labels = [f"T{i}" for i in range(len(formulation.blocks) - 1)] + ["makespan"]
    ax.set_xticks(centres, labels, rotation=90)
    ax.set_yticks(centres, labels)
    ax.set_title(f"QUBO coefficients (penalty {formulation.penalty})", loc="left")
