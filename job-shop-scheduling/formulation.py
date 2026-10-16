from collections import Counter, defaultdict
from itertools import combinations
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from problem import Problem


class Formulation(NamedTuple):
    """A QUBO whose energy, sum of Q[a, b] x_a x_b plus the offset, is the makespan of a feasible schedule."""

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
    """The problem's QUBO; for_qaoa leaves out the one-hot penalty, which QAOA's mixer enforces instead."""
    problem = add_makespan_task(problem)
    windows = problem.windows()

    # the makespan task can't start before `earliest`, so we measure the objective from there
    # it then runs from 0 to horizon - earliest, so breaking a constraint only has to cost more than that
    earliest = windows[-1].start
    penalty = problem.horizon - earliest + 1

    # one variable per task and start time
    variables = [(i, k) for i in range(len(problem)) for k in windows[i]]
    index = {variable: n for n, variable in enumerate(variables)}
    blocks = [[index[(i, k)] for k in windows[i]] for i in range(len(problem))]

    Q = defaultdict(int)

    # the objective is the makespan task's start time, less `earliest`
    # each of its variables gets a linear cost equal to its time, so the one set at time k adds k - earliest
    for n in blocks[-1]:  # the makespan task is the last one
        Q[(n, n)] += variables[n][1] - earliest

    # each task starts exactly once: penalty * (sum_k x_ik - 1)^2, expanded using x^2 = x,
    # with the constant +1 from each square collected in the offset
    if not for_qaoa:
        for block in blocks:
            for n in block:
                Q[(n, n)] -= penalty
            for a, b in combinations(block, 2):
                Q[(a, b)] += 2 * penalty

    # parents finish before their children start, and one task per machine at a time
    for a, b in clashes(problem, windows):
        Q[(index[a], index[b])] += penalty

    # the offset adds back `earliest`, and the +1 per task left over from the one-hot squares
    offset = earliest + (0 if for_qaoa else penalty * len(problem))
    return Formulation(problem.horizon, penalty, variables, blocks, dict(Q), offset)


def clashes(problem: Problem, windows):
    """The pairs of (task, time) variables that can't both be set; one breaking both comes up twice."""
    for i, (_, _, dependencies) in enumerate(problem.tasks):
        for p in dependencies:
            for kp in windows[p]:
                for ki in windows[i]:
                    if ki < kp + problem.tasks[p][1]:  # i starts before p finishes
                        yield (p, kp), (i, ki)

    # two tasks on one machine whose intervals overlap
    for i, j in combinations(range(len(problem)), 2):
        if problem.tasks[i][0] == problem.tasks[j][0]:
            for ki in windows[i]:
                for kj in windows[j]:
                    if ki < kj + problem.tasks[j][1] and kj < ki + problem.tasks[i][1]:
                        yield (i, ki), (j, kj)


def add_makespan_task(problem: Problem):
    """Add a zero-duration task on its own machine that follows every childless task."""
    sinks = tuple(i for i in range(len(problem)) if not problem.children[i])
    return Problem(list(problem.tasks) + [(problem.num_machines, 0, sinks)])


def qubo_to_ising(Q):
    """h and J of the Ising energy sum h_i s_i + sum J_ij s_i s_j, up to a constant, from x = (1 - s) / 2."""
    h, J = defaultdict(float), defaultdict(float)
    for (a, b), coefficient in Q.items():
        if a == b:
            h[a] -= coefficient / 2
        else:
            J[(a, b)] += coefficient / 4
            h[a] -= coefficient / 4
            h[b] -= coefficient / 4
    return dict(h), dict(J)


def decode(bits, formulation):
    """Start times from bits in variable order; a task whose block isn't one-hot gets None."""
    start_times = []
    for block in formulation.blocks:
        on = [n for n in block if int(bits[n])]
        start_times.append(formulation.variables[on[0]][1] if len(on) == 1 else None)
    return start_times


def best_feasible(samples, weights, formulation, problem):
    """The best feasible start times among samples, and the weight at each makespan (None for infeasible)."""
    # judged on the real tasks alone, so a sample with the makespan task unset or late still counts
    best, best_makespan, makespans = None, None, Counter()
    for bits, weight in zip(samples, weights):
        start_times = decode(bits, formulation)[: len(problem)]
        makespan = problem.makespan(start_times) if problem.is_feasible(start_times) else None
        makespans[makespan] += float(weight)
        if makespan is not None and (best is None or makespan < best_makespan):
            best, best_makespan = start_times, makespan
    return best, dict(makespans)


def draw_qubo(formulation):
    """The QUBO as a heatmap, one row and column per variable, grouped into each task's start times."""
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
