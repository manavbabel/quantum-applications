import json
import random
from collections import defaultdict
from functools import cached_property
from graphlib import TopologicalSorter
from itertools import pairwise

import matplotlib.pyplot as plt
from matplotlib import patches


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
    # the solvers are imported here rather than at the top, as they import Problem themselves
    @cached_property
    def horizon(self):
        from classical import GreedySolver

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
            from classical import GreedySolver

            start_times = GreedySolver().solve(self).start_times
            title = title or "greedy"

        total = self.makespan(start_times)

        figure, ax = plt.subplots(figsize=(10, 6))
        for i, (machine, duration, _) in enumerate(self.tasks):
            if not duration:
                continue
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
        ax.set_ylim(-1, self.num_machines)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machines")
        ax.set_yticks(range(self.num_machines))
        ax.set_yticklabels([f"M{i}" for i in range(self.num_machines)])
        ax.grid(True)
        ax.set_title(f"{title or self.name or 'schedule'} (makespan {total})")
        figure.tight_layout()
        plt.show()
