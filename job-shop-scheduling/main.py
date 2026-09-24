import json
import math
import random
import time
from bisect import bisect_right, insort
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from graphlib import TopologicalSorter
from itertools import combinations, islice, pairwise, product
from math import prod
from pathlib import Path
from typing import NamedTuple

import dimod
import gurobipy as gp
import matplotlib.pyplot as plt
import minorminer
import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from gurobipy import GRB
from matplotlib import patches
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit.library import qaoa_ansatz
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import IBMBackend, SamplerV2
from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.optimizers import COBYLA


# a random dependency graph with no redundant dependencies
# tasks are placed one at a time; each draws how many parents it wants, evenly between min_parents and
# max_parents (none making it a starting task), then picks them one by one, uniformly from the tasks placed
# before it that neither depend on nor are depended on by the parents it already has
# so no parent is implied by another, and as a new task has no children yet, no earlier dependency becomes
# implied either
# a task that runs out of such tasks keeps what it found, unless that is fewer than min_parents: then, as for
# the first task, it starts with none
# any graph meeting these rules can come up, though not all equally often
# the placement order is shuffled into the task indices, so a parent may have a higher index than its child
def random_dependencies(num_tasks, min_parents, max_parents, rng):
    # ancestors[i] is a bitmask of the earlier-placed tasks that task i depends on, directly or not
    ancestors, parents = [], []
    for i in range(num_tasks):
        wanted = rng.randint(min_parents, max_parents)
        # chosen is a bitmask of the picked parents; excluded adds everything they depend on
        picked, chosen, excluded = [], 0, 0
        while len(picked) < wanted:
            # skip the picked parents, what they depend on, and what depends on them
            options = [j for j in range(i) if not excluded >> j & 1 and not ancestors[j] & chosen]
            if not options:
                break
            j = rng.choice(options)
            picked.append(j)
            chosen |= 1 << j
            excluded |= ancestors[j] | 1 << j
        if len(picked) < max(min_parents, 1):
            picked, excluded = [], 0
        parents.append(picked)
        ancestors.append(excluded)

    order = list(range(num_tasks))
    rng.shuffle(order)
    dependencies = [()] * num_tasks
    for i, picked in enumerate(parents):
        dependencies[order[i]] = tuple(sorted(order[j] for j in picked))
    return dependencies


# a uniformly random assignment of tasks to machines, among those that give every machine at least one task
# ways[i][j] is how many ways the last i tasks can be assigned so that the j machines still unused all get one
def random_machines(num_tasks, num_machines, rng):
    ways = [[1] + [0] * num_machines]
    for i in range(1, num_tasks + 1):
        ways.append(
            [
                (num_machines - j) * ways[i - 1][j] + (j * ways[i - 1][j - 1] if j else 0)
                for j in range(num_machines + 1)
            ]
        )

    unused, used, machines = list(range(num_machines)), [], []
    for i in range(num_tasks, 0, -1):
        j = len(unused)
        # an unused machine, or one already in use, as often as the ways to finish from each
        if j and rng.randrange(ways[i][j]) < j * ways[i - 1][j - 1]:
            used.append(unused.pop(rng.randrange(j)))
            machines.append(used[-1])
        else:
            machines.append(rng.choice(used))
    return machines


class Problem:
    def __init__(self, tasks, name: str = None, optimum: int | None = None):
        # each task is a tuple of (machine, duration, dependencies), where dependencies is a tuple of task indices, 0-indexed

        # validate tasks
        for task in tasks:
            machine, duration, dependencies = task

            if machine < 0:
                raise ValueError(f"Machine index must be non-negative, got {machine}")
            if duration < 0:
                raise ValueError(f"Duration must be non-negative, got {duration}")
            for dependency in dependencies:
                if dependency < 0 or dependency >= len(tasks):
                    raise ValueError(
                        f"Dependency index {dependency} out of bounds for tasks of length {len(tasks)}"
                    )

        self.tasks = tasks
        self.name = name
        self.optimum = optimum

    @classmethod
    def load(cls, instance_name):
        """Loads an instance from benchmark_instances.json. Note these are special classes of our general problem where each task has at most one parent."""
        with open("benchmark_instances.json", "r") as f:
            instance_dict = json.load(f)[instance_name]

        durations = instance_dict["duration_matrix"]
        machines = instance_dict["machines_matrix"]

        tasks = []

        for j in range(len(durations)):
            for i in range(len(durations[j])):
                if i == 0:
                    parent = ()
                else:
                    # the previous task in this job, which was the last one added
                    parent = (len(tasks) - 1,)

                tasks.append((machines[j][i], durations[j][i], parent))

        # the benchmark set records the optimal makespan where it is known
        return cls(name=instance_dict["name"], tasks=tasks, optimum=instance_dict["metadata"].get("optimum"))

    @classmethod
    def random(
        cls,
        num_tasks,
        num_machines,
        min_parents=0,
        max_parents=2,
        min_duration=1,
        max_duration=10,
        seed=None,
    ):
        # a seeded random problem with
        # num_tasks tasks
        # num_machines machines, each given at least one task, the assignment uniform among those that do
        # between min and max parents for each task: each aims for a count drawn evenly across that range, and
        # gets fewer when the tasks before it cannot supply enough independent ones; one that cannot get
        # min_parents (the first placed, at least) gets none, and min_parents=0 lets any task be a starting task
        # no dependency implied by others (A before B before C, and A before C), as it would only add
        # QUBO couplings
        # between min and max durations (integers), uniformly
        if num_tasks < 1:
            raise ValueError(f"need at least one task, got {num_tasks}")
        if not 1 <= num_machines <= num_tasks:
            raise ValueError(
                f"need 1 <= num_machines <= num_tasks, got {num_machines} and {num_tasks}"
            )
        if not 0 <= min_parents <= max_parents:
            raise ValueError(
                f"need 0 <= min_parents <= max_parents, got {min_parents} and {max_parents}"
            )
        if not 0 <= min_duration <= max_duration:
            raise ValueError(
                f"need 0 <= min_duration <= max_duration, got {min_duration} and {max_duration}"
            )

        rng = random.Random(seed)
        dependencies = random_dependencies(num_tasks, min_parents, max_parents, rng)
        machines = random_machines(num_tasks, num_machines, rng)
        durations = [rng.randint(min_duration, max_duration) for _ in range(num_tasks)]
        return cls(list(zip(machines, durations, dependencies)), name=f"random-{seed}")

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return self.tasks.__iter__()

    def describe(self):
        print(f"{self.name or 'unnamed'} problem")
        print(f" * tasks        : {len(self)}")
        print(f" * machines     : {self.num_machines}")
        print(f" * dependencies : {sum(len(d) for d in self.dependencies)}")
        print(f" * makespan     : between {self.lower_bound} and {self.horizon}")
        print(f" * optimum      : {'unknown' if self.optimum is None else self.optimum}")

    # the optimal makespan, if known: from the benchmark set, set by hand, recorded by a solver that
    # proves it, or implied when the lower bound already meets the greedy schedule
    # success probabilities and reps99 are measured against it
    @property
    def optimum(self):
        if self._optimum is None and self.lower_bound == self.horizon:
            return self.lower_bound
        return self._optimum

    @optimum.setter
    def optimum(self, value):
        self._optimum = value

    @property
    def num_machines(self):
        return max(machine for machine, _, _ in self.tasks) + 1

    # the tasks that directly precede each task
    @cached_property
    def dependencies(self):
        return [dependencies for _, _, dependencies in self.tasks]

    # the tasks that directly follow each task
    @cached_property
    def children(self):
        children = [[] for _ in self.tasks]
        for i, dependencies in enumerate(self.dependencies):
            for p in dependencies:
                children[p].append(i)
        return children

    # given an adjacency list (parents or children) and a task
    # find all tasks reachable from this task, not including the task itself
    # a loop rather than recursion, so a long chain cannot hit the recursion limit
    def reachable(self, task_index, adjacency):
        found, stack = set(), list(adjacency[task_index])
        while stack:
            j = stack.pop()
            if j not in found:
                found.add(j)
                stack.extend(adjacency[j])
        return found

    # what is the total duration on the busiest machine for a given set of tasks?
    def machine_load(self, task_indices):
        load = defaultdict(int)
        for i in task_indices:
            machine, duration, _ = self.tasks[i]
            load[machine] += duration
        return max(load.values(), default=0)

    # the earliest each task can start: the later of every parent finishing at its own earliest,
    # and the busiest machine working through everything before it
    # worked out parents first, so a task builds on its parents' bounds rather than on raw durations,
    # and a parent held back by a busy machine holds back everything after it too
    @cached_property
    def heads(self):
        heads = [0] * len(self)
        for i in TopologicalSorter(dict(enumerate(self.dependencies))).static_order():
            heads[i] = max(
                max((heads[p] + self.tasks[p][1] for p in self.dependencies[i]), default=0),
                self.machine_load(self.reachable(i, self.dependencies)),
            )
        return heads

    # the least time from each task starting to everything after it finishing, by the same two
    # measures over the task and everything after it, worked out children first
    @cached_property
    def tails(self):
        tails = [0] * len(self)
        for i in TopologicalSorter(dict(enumerate(self.children))).static_order():
            tails[i] = max(
                self.tasks[i][1] + max((tails[c] for c in self.children[i]), default=0),
                self.machine_load(self.reachable(i, self.children) | {i}),
            )
        return tails

    # the makespan is at least any task's earliest start plus the least time after it,
    # and at least the busiest machine's total load
    @cached_property
    def lower_bound(self):
        return max(
            max(head + tail for head, tail in zip(self.heads, self.tails)),
            self.machine_load(range(len(self))),
        )

    # an upper bound on the makespan: whatever the greedy schedule manages
    # straight from _solve, since solve measures against the optimum, which needs the horizon
    @cached_property
    def horizon(self):
        start_times, *_ = GreedySolver()._solve(self)
        return self.makespan(start_times)

    # for a given task, what are the possible start times?
    # from its earliest start, to the horizon less the least time after it
    def windows(self):
        return [range(head, self.horizon - tail + 1) for head, tail in zip(self.heads, self.tails)]

    # given a solution (start times for each task), what is the makespan?
    def makespan(self, start_times):
        return max(
            start + duration for start, (_, duration, _) in zip(start_times, self.tasks)
        )

    # check the dependency and machine constraints hold (start times are assumed to be non-negative)
    def is_feasible(self, start_times):
        if start_times is None or any(start is None for start in start_times):
            return False

        # check that all dependencies finish by the time this task starts
        for i, dependencies in enumerate(self.dependencies):
            if any(
                start_times[p] + self.tasks[p][1] > start_times[i] for p in dependencies
            ):
                return False

        # check that no two tasks on the same machine overlap
        busy = defaultdict(list)
        for i, (machine, duration, _) in enumerate(self.tasks):
            busy[machine].append((start_times[i], start_times[i] + duration))
        return all(
            end <= next_start
            for intervals in busy.values()
            for (_, end), (next_start, _) in pairwise(sorted(intervals))
        )

    # draw a schedule as a Gantt chart, by default the greedy one
    # zero-duration tasks, such as the makespan task, get no bar
    def draw(self, start_times=None, title=None):
        if start_times is None:
            start_times = GreedySolver().solve(self).start_times
            title = title or "greedy"

        # a decoded schedule may still carry the makespan task at the end
        if len(start_times) == len(self) + 1:
            start_times = start_times[:-1]
        elif len(start_times) != len(self):
            raise ValueError(f"expected {len(self)} start times, got {len(start_times)}")

        bars = [
            (i, machine, duration)
            for i, (machine, duration, _) in enumerate(self.tasks)
            if duration
        ]
        num_machines = self.num_machines
        total = self.makespan(start_times)

        figure, ax = plt.subplots(figsize=(10, 6))
        for i, machine, duration in bars:
            ax.add_patch(
                patches.Rectangle(
                    (start_times[i], machine - 0.5),
                    duration,
                    1,
                    edgecolor="black",
                    facecolor="skyblue",
                )
            )
            ax.text(
                start_times[i] + duration / 2,
                machine,
                f"T{i}",
                ha="center",
                va="center",
            )

        ax.set_xlim(0, total)
        ax.set_ylim(-1, num_machines)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machines")
        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f"M{i}" for i in range(num_machines)])
        ax.grid(True)
        ax.set_title(f"{title or self.name or 'schedule'} (makespan {total})")
        figure.tight_layout()
        plt.show()


# QUBO formulation
# energy is sum over a,b of Q[a,b]x_ax_b + offset
# for a feasible solution, energy is makespan
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
    def num_linear_terms(self):
        return sum(1 for a, b in self.Q if a == b)

    @property
    def num_quadratic_terms(self):
        return sum(1 for a, b in self.Q if a != b)

    def describe(self):
        print("QUBO formulation")
        print(f" * variables       : {self.num_variables}")
        print(f" * linear terms    : {self.num_linear_terms}")
        print(f" * quadratic terms : {self.num_quadratic_terms}")
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
        0 if for_qaoa else penalty * len(problem),
    )

# add zero-duration makespan task whose parents are every childless task, on its own machine
def add_makespan_task(problem:Problem):
    sinks = tuple(i for i in range(len(problem)) if not problem.children[i])
    return Problem(list(problem.tasks) + [(problem.num_machines, 0, sinks)])

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
# a task whose block is not exactly one-hot has no start time, so decodes to None, which is infeasible
def decode(bits, formulation):
    start_times = []
    for block in formulation.blocks:
        on = [n for n in block if int(bits[n])]
        start_times.append(formulation.variables[on[0]][1] if len(on) == 1 else None)
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


# what a sampling solver drew, for its success probability and reps99
# each sample (a read or a shot) counts as a run of its own
class Samples(NamedTuple):
    makespans: dict  # makespan -> weight, as tallied by best_feasible

    def fraction(self, makespan):
        return self.makespans.get(makespan, 0) / sum(self.makespans.values())


# how many runs (reads or shots) it takes to see the optimum at least once with 99% confidence
# a run that finds it with probability p needs ln(0.01) / ln(1 - p) of them, but always at least one
# set against the runs actually drawn, it says whether the shot count could come down, or needs to go up
def reps99(p):
    if p is None:
        return None
    if p == 0:
        return math.inf
    return 1.0 if p >= 0.99 else math.log(0.01) / math.log(1 - p)


class Solver:
    name = "abstract solver"

    # subclasses implement _solve(problem, **kwargs), returning (start_times, tts, calls, metadata, samples)
    # and build a formulation from the problem themselves if they need one
    # start_times may carry the makespan task on the end, which is trimmed off here
    # samples is None for a deterministic solver, whose single run either finds the optimum or never will
    def solve(self, problem:Problem, **kwargs):
        start_times, tts, calls, metadata, samples = self._solve(problem, **kwargs)

        if start_times is not None:
            start_times = list(start_times)[: len(problem)]
        feasible = problem.is_feasible(start_times)
        makespan = problem.makespan(start_times) if feasible else None

        # a proven optimum is a fact about the problem, so later solvers can be measured against it
        if feasible and metadata.get("proven_optimal") and problem.optimum is None:
            problem.optimum = makespan

        # the chance one run finds the optimum
        if problem.optimum is None:
            p = None
        elif samples is None:
            p = float(makespan == problem.optimum)
        else:
            p = samples.fraction(problem.optimum)

        return SolverResult(
            makespan=makespan,
            feasible=feasible,
            tts=tts,
            success_probability=p,
            reps99=reps99(p),
            start_times=start_times,
            solver=self.name,
            calls=calls,
            metadata=metadata,
        )

    # what a solve would need, without running it; takes the same keywords as solve
    def estimate(self, problem:Problem, **kwargs):
        raise NotImplementedError

class SolverUnavailable(RuntimeError):
    # can catch this to ensure a run doesn't crash
    pass

@dataclass
class SolverResult:
    makespan: int | None
    feasible: bool
    tts: float  # wall-clock seconds for this solve
    success_probability: float | None  # the chance one run finds the optimum
    reps99: float | None  # runs needed to see the optimum with 99% confidence; None if the optimum is unknown
    start_times: list | None
    solver: str
    calls: int
    metadata: dict = field(default_factory=dict)

    def describe(self):
        print(f"{self.solver} result {'FEASIBLE' if self.feasible else 'NOT FEASIBLE'}")
        print(f" * makespan     : {self.makespan}")
        print(f" * TTS          : {self.tts}")
        print(f" * p(optimum)   : {self.success_probability}")
        print(f" * reps99       : {self.reps99}")
        print(f" * calls        : {self.calls}")
        print(f" * metadata     : {json.dumps(self.metadata, indent=4, default=str)}")

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
        if (total:=prod([len(w) for w in windows])) > self.MAX_COMBINATIONS:
            raise SolverUnavailable(
                f"{total:,} combinations exceeds MAX_COMBINATIONS={self.MAX_COMBINATIONS:,}"
            )

        best = None
        calls = 0
        t0 = time.perf_counter()
        for candidate in product(*windows):
            calls += 1
            if problem.is_feasible(list(candidate)):
                value = problem.makespan(list(candidate))
                if best is None or value < best[0]:
                    best = (value, list(candidate))
        t1 = time.perf_counter()

        metadata = {
            "calls_unit": "combinations",
            "combinations": total,
            "variables": sum(len(w) for w in windows),
            "couplings": 0,
            "proven_optimal": True,  # every combination was checked
        }

        return (best[1] if best else None), t1-t0, calls, metadata, None

    # every combination of start times in the windows, priced by timing the first few of the same search
    def estimate(self, problem, trials=1000):
        windows = problem.windows()
        total = prod(len(w) for w in windows)

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

# samples the QUBO with any dimod sampler
# by default a D-Wave QPU, found through the local dwave config
# locally, a simulated quantum annealer such as dwave.samplers.PathIntegralAnnealingSampler
# a QPU (or a dwave.system.testing.MockDWaveSampler) only has certain couplers, so there the QUBO
# is minor-embedded first
class AnnealingSolver(Solver):

    def __init__(self, sampler=None):
        if sampler is None:
            try:
                sampler = DWaveSampler()
            except Exception as exc:  # no token, no QPU online, network down
                raise SolverUnavailable(f"cannot reach a D-Wave QPU: {exc}") from exc
        self.sampler = sampler
        self.name = f"annealing/{sampler.properties.get('chip_id', type(sampler).__name__)}"

    # any other keyword goes straight to the sampler, e.g. seed for a simulator or annealing_time for a QPU
    def _solve(self, problem, num_reads=1000, **parameters):
        formulation = build_formulation(problem)
        bqm = dimod.BinaryQuadraticModel.from_qubo(formulation.Q, offset=formulation.offset)

        sampler = self.sampler
        structured = isinstance(sampler, dimod.Structured)
        # a QPU builds its coupler list on first use, which is a fact about the device, not part of solving
        target = sampler.edgelist if structured else None

        t0 = time.perf_counter()
        if structured:
            # finding the embedding is a one-off cost ahead of the reads, so it is timed on its own
            # each variable gets a self-loop, so one with no couplings is still embedded
            source = list(bqm.quadratic) + [(v, v) for v in bqm.variables]
            embedding = minorminer.find_embedding(source, target)
            embedding_time = time.perf_counter() - t0
            # minorminer gives up by returning an empty embedding, rather than raising
            if not embedding:
                raise SolverUnavailable(f"cannot embed {bqm.num_variables} variables on {self.name}")
            sampler = FixedEmbeddingComposite(sampler, embedding)
            parameters["return_embedding"] = True  # to report the chain strength

        sampleset = sampler.sample(bqm, num_reads=num_reads, **parameters)
        sampleset.resolve()  # a QPU answers asynchronously, so wait for it inside the timing
        t1 = time.perf_counter()

        # sampleset columns follow the BQM's own variable order, so pick them out by label
        columns = [sampleset.variables.index(n) for n in range(formulation.num_variables)]
        start_times, makespans = best_feasible(
            sampleset.record.sample[:, columns],
            sampleset.record.num_occurrences,
            formulation,
            problem,
        )
        # each read is a run of its own
        samples = Samples(makespans)

        metadata = {
            "calls_unit": "reads",
            "variables": formulation.num_variables,
            "lowest_energy": sampleset.first.energy,
            "feasible_fraction": 1 - samples.fraction(None),
        }
        if structured:
            chains = [len(chain) for chain in embedding.values()]
            qpu_access_time = sampleset.info.get("timing", {}).get("qpu_access_time")  # microseconds
            metadata.update({
                "embedding_time": embedding_time,  # seconds, within the TTS
                "qpu_access_time": qpu_access_time and qpu_access_time / 1e6,  # seconds
                "physical_qubits": sum(chains),
                "max_chain_length": max(chains),
                "chain_strength": sampleset.info["embedding_context"]["chain_strength"],
                "chain_break_fraction": float(
                    np.average(
                        sampleset.record.chain_break_fraction,
                        weights=sampleset.record.num_occurrences,
                    )
                ),
            })
        return start_times, t1 - t0, num_reads, metadata, samples

    # the logical problem that has to be embedded, and how much QPU time the reads should take
    # only a QPU can price its time, and it is quoted for one qubit per variable, which is a lower
    # bound: chains in the embedding take more
    def estimate(self, problem, num_reads=1000, **parameters):
        formulation = build_formulation(problem)
        bqm = dimod.BinaryQuadraticModel.from_qubo(formulation.Q, offset=formulation.offset)

        estimate = {
            "variables": bqm.num_variables,
            "couplings": bqm.num_interactions,
            "num_reads": num_reads,
            "qpu_access_time": None,
        }
        if isinstance(self.sampler, dimod.Structured):
            estimate["hardware_qubits"] = len(self.sampler.nodelist)
        solver = getattr(self.sampler, "solver", None)
        if hasattr(solver, "estimate_qpu_access_time"):
            microseconds = solver.estimate_qpu_access_time(
                bqm.num_variables, num_reads=num_reads, **parameters
            )
            estimate["qpu_access_time"] = float(microseconds) / 1e6  # seconds
        return estimate


# the QAOA solvers below work on the formulation without its one-hot penalty
# each task's block of variables starts in an equal superposition of its start times, and an
# XY mixer within each block keeps exactly one of them set, so the constraint holds by construction
# bit order: qubit n stands for variable n; Qiskit reads its registers little-endian, so qubit n is
# bit n of an integer outcome and the rightmost character of a counts key (counts_to_bits reverses those)
# every sample below is turned into a list of bits in variable order

# the cost Hamiltonian: the QUBO's Ising form, as Z and ZZ terms
# scaled so its largest coefficient is 1, which keeps the QAOA angles meaning the same thing from one
# instance to the next; the constant only adds a global phase, so it is dropped
# built directly, as qiskit-optimization's QuadraticProgram.to_ising takes minutes on ft06
def cost_operator(formulation):
    h, J, _ = qubo_to_ising(formulation.Q)
    scale = max(map(abs, [*h.values(), *J.values()]), default=0) or 1
    terms = [("Z", [a], c / scale) for a, c in h.items()]
    terms += [("ZZ", [a, b], c / scale) for (a, b), c in J.items()]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=formulation.num_variables).simplify()

# put each block into an equal superposition of its start times, the W state
# built as a cascade, so it costs O(len(block)) gates rather than a state preparation's O(2^len(block))
def w_states(formulation):
    circuit = QuantumCircuit(formulation.num_variables)
    for block in formulation.blocks:
        circuit.x(block[0])
        for i, (a, b) in enumerate(pairwise(block)):
            circuit.cry(2 * np.arccos(np.sqrt(1 / (len(block) - i))), a, b)
            circuit.cx(b, a)
    return circuit

# the XY mixer, -(XX + YY)/2 around a ring within each block
# it moves a task between start times without ever setting two at once, and its ground state is the
# W state above, which is where an annealing-like ramp has to start
# one start time needs no mixer, and two share a single pair, which a ring would couple twice
def xy_mixer(formulation):
    terms = []
    for block in formulation.blocks:
        pairs = list(pairwise(block))
        if len(block) > 2:
            pairs.append((block[-1], block[0]))
        for a, b in pairs:
            terms += [("XX", [a, b], -0.5), ("YY", [a, b], -0.5)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=formulation.num_variables)

# the linear ramps from arXiv:2405.09169 eq. 4
# over layers i = 0 .. p-1 gamma ramps up and beta ramps down, neither reaching zero
# ordered betas then gammas, as qaoa_ansatz orders its parameters
def linear_ramp(p, delta_gamma, delta_beta):
    layers = np.arange(p)
    return np.concatenate([(1 - layers / p) * delta_beta, (layers + 1) / p * delta_gamma])


# a generic gate set to count logical depth in
LOGICAL_GATES = ["cx", "rz", "sx", "x"]


# shared set-up for the gate-based solvers
# backend is an AerSimulator (or fake backend) for local runs, or an IBM backend for a real device
# both go through Qiskit Runtime's V2 sampler, the local ones in its local testing mode
# seed fixes the transpiler and, on a simulator, the shots; the hardware itself is not reproducible
class _QAOASolver(Solver):

    def __init__(self, backend, shots=4096, seed=None):
        self.backend = backend
        self.shots = shots
        self.seed = seed
        self.name = f"{self.name}/{backend.name}"

        options = {"default_shots": shots}
        if seed is not None and not isinstance(backend, IBMBackend):
            options["simulator"] = {"seed_simulator": seed}
        self.sampler = SamplerV2(mode=backend, options=options)
        self.pass_manager = generate_preset_pass_manager(backend=backend, seed_transpiler=seed)

    # one qubit per variable, which has to fit on the backend
    def _formulation(self, problem):
        formulation = build_formulation(problem, for_qaoa=True)
        if formulation.num_variables > self.backend.num_qubits:
            raise SolverUnavailable(
                f"{formulation.num_variables} qubits needed, {self.backend.name} has {self.backend.num_qubits}"
            )
        return formulation

    # the logical circuit, with its angles left free
    # built as qiskit-optimization's QAOA builds its own, so both solvers run the same ansatz
    def _ansatz(self, formulation, p):
        circuit = qaoa_ansatz(
            cost_operator(formulation),
            reps=p,
            initial_state=w_states(formulation),
            mixer_operator=xy_mixer(formulation),
        )
        circuit.measure_all()
        return circuit

    def _metadata(self, formulation, p, samples):
        return {
            "calls_unit": "circuit executions",
            "backend": self.backend.name,
            "qubits": formulation.num_variables,
            "p": p,
            "shots": self.shots,
            "feasible_fraction": 1 - samples.fraction(None),
        }

    # what a solve would need; nothing here has to fit on the backend, except the device figures
    # the logical depth is counted in a generic gate set, with every qubit free to talk to every other
    # on the backend itself, routing adds depth, and a device that publishes its gate durations
    # (IBM hardware, a fake backend, but not a simulator) also prices one shot
    def _estimate(self, problem, p, iterations, executions):
        formulation = build_formulation(problem, for_qaoa=True)
        circuit = self._ansatz(formulation, p)
        logical = generate_preset_pass_manager(optimization_level=1, basis_gates=LOGICAL_GATES).run(circuit)

        estimate = {
            "qubits": formulation.num_variables,
            "fits_backend": formulation.num_variables <= self.backend.num_qubits,
            "optimiser_iterations": iterations,
            "circuit_executions": executions,
            "shots_per_execution": self.shots,
            "total_shots": executions * self.shots,
            "logical_depth": logical.depth(),
            "logical_two_qubit_gates": logical.num_nonlocal_gates(),
        }
        if estimate["fits_backend"]:
            device = self.pass_manager.run(circuit)
            estimate["device_depth"] = device.depth()
            estimate["device_two_qubit_gates"] = device.num_nonlocal_gates()
            try:
                estimate["shot_duration"] = device.estimate_duration(self.backend.target)  # seconds
            except QiskitError:  # no gate durations to go on
                estimate["shot_duration"] = None
        return estimate

# QAOA, with the angles found by a classical optimiser through qiskit-optimization
# they start on the linear ramp LR-QAOA uses below, which the optimiser then refines
# alpha < 1 minimises the CVaR (the mean energy of the best alpha fraction of shots) instead of the mean
class QAOASolver(_QAOASolver):
    name = "qaoa"

    @staticmethod
    def _optimizer(optimizer):
        return optimizer or COBYLA(maxiter=100)

    def _solve(self, problem, p=1, alpha=1.0, optimizer=None, delta_gamma=0.6, delta_beta=0.3):
        formulation = self._formulation(problem)
        evaluated = []  # when each objective evaluation finished
        qaoa = QAOA(
            self.sampler,
            self._optimizer(optimizer),
            reps=p,
            initial_state=w_states(formulation),
            mixer=xy_mixer(formulation),
            initial_point=linear_ramp(p, delta_gamma, delta_beta),
            aggregation=alpha,
            callback=lambda *_: evaluated.append(time.perf_counter()),
            pass_manager=self.pass_manager,
        )

        cost = cost_operator(formulation)
        t0 = time.perf_counter()
        eigen = qaoa.compute_minimum_eigenvalue(cost)
        t1 = time.perf_counter()

        # the samples are the distribution at the optimised angles, each shot a run of its own
        # the distribution is keyed by integer outcome, with qubit n as bit n
        distribution = eigen.eigenstate
        start_times, makespans = best_feasible(
            [[(state >> n) & 1 for n in range(formulation.num_variables)] for state in distribution],
            list(distribution.values()),
            formulation,
            problem,
        )
        samples = Samples(makespans)

        evaluations = int(eigen.cost_function_evals)
        metadata = self._metadata(formulation, p, samples)
        metadata.update({
            "alpha": alpha,
            "optimizer_evaluations": evaluations,
            "optimisation_time": evaluated[-1] - t0,
            "optimal_point": eigen.optimal_point.tolist(),
        })
        # one circuit execution per objective evaluation, and one more for the final distribution
        return start_times, t1 - t0, evaluations + 1, metadata, samples

    # the optimiser stops by maxiter evaluations at the latest, one circuit execution each (as for
    # COBYLA; SPSA takes two), then samples the final distribution once more
    # scipy's COBYLA raises maxiter to at least one more than its initial simplex, i.e. the 2p angles + 2
    def estimate(self, problem, p=1, alpha=1.0, optimizer=None, delta_gamma=0.6, delta_beta=0.3):
        iterations = max(self._optimizer(optimizer).settings["maxiter"], 2 * p + 2)
        return self._estimate(problem, p, iterations, iterations + 1)

# LR-QAOA: the angles follow a fixed annealing-like schedule, so there is no optimiser and one execution
# the ramps and the default slopes are from arXiv:2405.09169
class LRQAOASolver(_QAOASolver):
    name = "lr-qaoa"

    def _solve(self, problem, p=10, delta_gamma=0.6, delta_beta=0.3):
        formulation = self._formulation(problem)
        circuit = self.pass_manager.run(self._ansatz(formulation, p))

        t0 = time.perf_counter()
        job = self.sampler.run([(circuit, linear_ramp(p, delta_gamma, delta_beta))])
        counts = job.result()[0].data.meas.get_counts()
        t1 = time.perf_counter()

        # each shot is a run of its own
        start_times, makespans = best_feasible(
            [counts_to_bits(key) for key in counts],
            list(counts.values()),
            formulation,
            problem,
        )
        samples = Samples(makespans)

        metadata = self._metadata(formulation, p, samples)
        metadata.update({
            "delta_gamma": delta_gamma,
            "delta_beta": delta_beta,
            "depth": circuit.depth(),
            "two_qubit_gates": circuit.num_nonlocal_gates(),
        })
        return start_times, t1 - t0, 1, metadata, samples

    # one pass through the ramp, so one execution
    def estimate(self, problem, p=10, delta_gamma=0.6, delta_beta=0.3):
        return self._estimate(problem, p, 1, 1)
