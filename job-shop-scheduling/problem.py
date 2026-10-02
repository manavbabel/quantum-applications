import json
import random
from bisect import bisect_right, insort
from collections import defaultdict
from functools import cached_property
from graphlib import TopologicalSorter
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

INSTANCES = Path(__file__).parent / "benchmark_instances.json"


# a random dependency graph with no redundant edges (never A -> C as well as A -> B -> C)
# tasks are placed one at a time, each picking between min_parents and max_parents parents from the
# earlier tasks unrelated to the parents it already has; a task that can't find min_parents gets none
# the order is then shuffled, so a parent may have a higher index than its child
def random_dependencies(num_tasks, min_parents, max_parents, rng):
    # ancestors[i] is a bitmask of every task that task i depends on, directly or not
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
        # a new machine or a used one, weighted by the number of ways to finish from each
        if j and rng.randrange(ways[i][j]) < j * ways[i - 1][j - 1]:
            used.append(unused.pop(rng.randrange(j)))
            machines.append(used[-1])
        else:
            machines.append(rng.choice(used))
    return machines


class Problem:
    def __init__(self, tasks, name: str | None = None, optimum: int | None = None):
        # each task is a tuple of (machine, duration, dependencies), where dependencies is a tuple of task indices, 0-indexed
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
        """Load a named instance from benchmark_instances.json. These are classic job shops, where each job is a chain of tasks, so every task has at most one parent."""
        with open(INSTANCES) as f:
            instance_dict = json.load(f)[instance_name]

        durations = instance_dict["duration_matrix"]
        machines = instance_dict["machines_matrix"]

        tasks = []

        for j in range(len(durations)):
            for i in range(len(durations[j])):
                if i == 0:
                    parent = ()
                else:
                    # the previous task in the same job
                    parent = (len(tasks) - 1,)

                tasks.append((machines[j][i], durations[j][i], parent))

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
        # a random problem where every machine gets at least one task, each task gets between min_parents
        # and max_parents parents (or none, if too few are available), and durations are uniform integers
        # implied dependencies are left out, as they would only add QUBO couplings
        if num_tasks < 1:
            raise ValueError(f"need at least one task, got {num_tasks}")
        if not 1 <= num_machines <= num_tasks:
            raise ValueError(f"need 1 <= num_machines <= num_tasks, got {num_machines} and {num_tasks}")
        if not 0 <= min_parents <= max_parents:
            raise ValueError(f"need 0 <= min_parents <= max_parents, got {min_parents} and {max_parents}")
        if not 0 <= min_duration <= max_duration:
            raise ValueError(f"need 0 <= min_duration <= max_duration, got {min_duration} and {max_duration}")

        rng = random.Random(seed)
        dependencies = random_dependencies(num_tasks, min_parents, max_parents, rng)
        machines = random_machines(num_tasks, num_machines, rng)
        durations = [rng.randint(min_duration, max_duration) for _ in range(num_tasks)]
        return cls(list(zip(machines, durations, dependencies)), name=f"random-{seed}")

    def __len__(self):
        return len(self.tasks)

    def describe(self):
        print(f"{self.name or 'unnamed'} problem")
        print(f" * tasks        : {len(self)}")
        print(f" * machines     : {self.num_machines}")
        print(f" * dependencies : {sum(len(d) for d in self.dependencies)}")
        print(f" * makespan     : between {self.lower_bound} and {self.horizon}")
        print(f" * optimum      : {'unknown' if self.optimum is None else self.optimum}")

    # the optimal makespan if known: given (by the benchmark set, or set after an exact solve), or implied
    # when the lower bound meets the greedy makespan; the solvers measure success against it
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

    # every task reachable from task_index through an adjacency list (parents or children), excluding itself
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

    # the earliest each task can start: after every parent has finished, and after the busiest machine
    # has worked through all of the task's ancestors on it
    # computed parents first, so each task builds on its parents' heads
    @cached_property
    def heads(self):
        heads = [0] * len(self)
        for i in TopologicalSorter(dict(enumerate(self.dependencies))).static_order():
            heads[i] = max(
                max((heads[p] + self.tasks[p][1] for p in self.dependencies[i]), default=0),
                self.machine_load(self.reachable(i, self.dependencies)),
            )
        return heads

    # the least time from each task starting to the end of the schedule, by the same two bounds,
    # computed children first
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

    # a quick schedule: tasks are placed once their parents are done, longest remaining chain first,
    # each in the first gap on its machine that fits
    def greedy_schedule(self):
        # each task's priority: the longest chain of durations from it to the end, itself included
        chain = [0] * len(self)
        for i in TopologicalSorter(dict(enumerate(self.children))).static_order():
            chain[i] = self.tasks[i][1] + max((chain[c] for c in self.children[i]), default=0)

        order = TopologicalSorter(dict(enumerate(self.dependencies)))
        order.prepare()
        start_times = [0] * len(self)
        # each machine's busy intervals, sorted; as they never overlap, their ends are sorted too
        busy = defaultdict(list)
        while order.is_active():
            ready = sorted(order.get_ready(), key=lambda i: chain[i], reverse=True)
            for i in ready:
                machine, duration, dependencies = self.tasks[i]
                start = max((start_times[p] + self.tasks[p][1] for p in dependencies), default=0)
                # skip the intervals over by the time the parents finish, then move past any in the way
                # until the task fits in a gap
                intervals = busy[machine]
                k = bisect_right(intervals, start, key=lambda interval: interval[1])
                while k < len(intervals) and intervals[k][0] < start + duration:
                    start = max(start, intervals[k][1])
                    k += 1
                insort(intervals, (start, start + duration))
                start_times[i] = start
            order.done(*ready)
        return start_times

    # an upper bound on the makespan: the greedy schedule's
    @cached_property
    def horizon(self):
        return self.makespan(self.greedy_schedule())

    # for a given task, what are the possible start times?
    # from its earliest start, to the horizon less the least time after it
    def windows(self):
        return [range(head, self.horizon - tail + 1) for head, tail in zip(self.heads, self.tails)]

    # given a solution (start times for each task), what is the makespan?
    def makespan(self, start_times):
        return max(start + duration for start, (_, duration, _) in zip(start_times, self.tasks))

    # check the dependency and machine constraints hold (start times are assumed to be non-negative)
    def is_feasible(self, start_times):
        if start_times is None or any(start is None for start in start_times):
            return False

        # check that all dependencies finish by the time this task starts
        for i, dependencies in enumerate(self.dependencies):
            if any(start_times[p] + self.tasks[p][1] > start_times[i] for p in dependencies):
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

    # draw a schedule as a Gantt chart (by default the greedy one), with an arrow from each parent's end
    # to its child's start; zero-duration tasks, such as the makespan task, get no bar
    def draw(self, start_times=None, title=None, ax=None):
        if start_times is None:
            start_times = self.greedy_schedule()
            title = title or "greedy"
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 1.2 + 0.6 * self.num_machines), layout="constrained")
        total = self.makespan(start_times)

        for i, (machine, duration, dependencies) in enumerate(self.tasks):
            if duration:
                bar = patches.Rectangle(
                    (start_times[i], machine - 0.4),
                    duration,
                    0.8,
                    facecolor="#86b6ef",
                    edgecolor="white",
                    linewidth=2,
                )
                ax.add_patch(bar)
                ax.text(start_times[i] + duration / 2, machine, f"T{i}", ha="center", va="center")
            for p in dependencies:
                ax.annotate(
                    "",
                    xy=(start_times[i], machine),
                    xytext=(start_times[p] + self.tasks[p][1], self.tasks[p][0]),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#52514e",
                        "linewidth": 1,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                )

        ax.set_xlim(0, total)
        ax.set_ylim(-0.6, self.num_machines - 0.4)
        ax.set_xlabel("time")
        ax.set_yticks(range(self.num_machines), [f"M{m}" for m in range(self.num_machines)])
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{title or self.name or 'schedule'} (makespan {total})", loc="left")
