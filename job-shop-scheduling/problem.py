# stores the problem itself with some useful methods

import random
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from graphlib import TopologicalSorter
from itertools import pairwise

# each task is a tuple of (machine, duration, dependencies), where dependencies is a tuple of task indices
Task = tuple[int, int, tuple[int, ...]]

class Problem:

    def __init__(self, tasks:list[Task], name:str=None):

        # validate
        for task in tasks:
            machine, duration, dependencies = task
            if machine < 0:
                raise ValueError(f"Machine index must be non-negative, got {machine}")
            if duration < 0:
                raise ValueError(f"Duration must be positive, got {duration}")
            for dependency in dependencies:
                if dependency < 0 or dependency >= len(tasks):
                    raise ValueError(f"Dependency index {dependency} out of bounds for tasks of length {len(tasks)}")

        self.name = name
        self.tasks = tasks

    # a seeded random problem whose QUBO has between min_variables and max_variables variables,
    # counting the makespan task's as build_formulation does
    # the family is drawn too, so problems spread across the space rather than sharing one shape:
    # one machine up to max_machines, unit tasks up to max_duration long ones, and dependencies
    # anywhere from none at all to a single chain
    # tasks only depend on earlier ones, so the dependencies form a DAG and a feasible schedule
    # (the greedy one) always exists
    @classmethod
    def random(cls, seed=None, max_variables=500, min_variables=2, max_machines=10, max_duration=10):
        from formulation import add_makespan_task  # formulation imports this module

        # the smallest problem is one task, with one start time, plus the makespan task
        if not 2 <= min_variables <= max_variables:
            raise ValueError(f"need 2 <= min_variables <= max_variables, got {min_variables} and {max_variables}")

        def num_variables(tasks):
            return sum(len(w) for w in add_makespan_task(cls(tasks)).windows())

        rng = random.Random(seed)
        while True:
            num_machines = rng.randint(1, max_machines)
            longest = rng.randint(1, max_duration)
            # squared, to favour sparse dependencies: once there are a few dozen tasks, even a
            # modest density chains most of them together
            density = rng.random() ** 2
            target = rng.randint(min_variables, max_variables)

            # every task costs at least one variable, as does the makespan task, so this many tasks
            # always overshoots
            # each depends on each earlier task with probability density, less any already implied
            # through another of its dependencies, which would only add redundant couplings
            tasks, ancestors = [], []
            for i in range(max_variables):
                parents = {j for j in range(i) if rng.random() < density}
                implied = set().union(*(ancestors[j] for j in parents))
                ancestors.append(parents | implied)
                tasks.append((rng.randrange(num_machines), rng.randint(1, longest), tuple(sorted(parents - implied))))

            # the longest run of leading tasks that fits the target, by bisection
            # any such run is a problem in its own right, since tasks only depend on earlier ones
            fits, size, overshoots = 0, None, len(tasks)
            while overshoots - fits > 1:
                middle = (fits + overshoots) // 2
                count = num_variables(tasks[:middle])
                if count <= target:
                    fits, size = middle, count
                else:
                    overshoots = middle
            if size is not None and size >= min_variables:
                return cls(tasks[:fits], name=f"random-{seed}")

    def __len__(self):
        return len(self.tasks)

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
    # find the total duration of the longest chain, not counting the task itself
    # i.e. giving parents finds how long before this task can start
    # giving children finds how long after this task finishes
    # worked out furthest task first, so each task is visited once rather than once per path to it,
    # which on a densely dependent problem is exponentially many
    def longest_chain(self, task_index, adjacency):
        chain = {}
        reach = self.reachable(task_index, adjacency) | {task_index}
        for i in TopologicalSorter({i: adjacency[i] for i in reach}).static_order():
            chain[i] = max((self.tasks[j][1] + chain[j] for j in adjacency[i]), default=0)
        return chain[task_index]

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

    # the makespan is at least the max of the critical path and the busiest machine's total load
    @cached_property
    def lower_bound(self):
        critical_path = max(duration + self.longest_chain(i, self.children) for i, (_, duration, _) in enumerate(self.tasks))
        return max(critical_path, self.machine_load(range(len(self))))

    # an upper bound on the makespan: whatever the greedy schedule manages
    @cached_property
    def horizon(self):
        return self.makespan(self.greedy())

    # for a given task, what are the possible start times?
    # the earliest start time is the later of the longest dependency chain, and the busiest machine's load over everything before it
    # the latest start time works back from the horizon by the same two measures over the task and everything after it
    def windows(self):
        windows = []
        for i, (_, duration, _) in enumerate(self.tasks):
            before = max(self.longest_chain(i, self.dependencies), self.machine_load(self.reachable(i, self.dependencies)))
            after = max(duration + self.longest_chain(i, self.children), self.machine_load(self.reachable(i, self.children) | {i}))
            windows.append(range(before, self.horizon - after + 1))
        return windows

    # greedily solve: assign tasks in topological order, each in the earliest gap on its machine once its dependencies are done,
    # taking the ready task with the longest path from it first (critical path priority rule)
    def greedy(self):
        order = TopologicalSorter(dict(enumerate(self.dependencies)))
        order.prepare()
        start_times = [0] * len(self)
        busy = defaultdict(list)
        while order.is_active():
            ready = sorted(order.get_ready(), key=lambda i: self.tasks[i][1] + self.longest_chain(i, self.children), reverse=True)
            for i in ready:
                machine, duration, dependencies = self.tasks[i]
                start = max((start_times[p] + self.tasks[p][1] for p in dependencies), default=0)
                for busy_start, busy_end in sorted(busy[machine]):
                    if busy_start < start + duration and start < busy_end:
                        start = busy_end
                busy[machine].append((start, start + duration))
                start_times[i] = start
            order.done(*ready)
        return start_times

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
        return all(end <= next_start for intervals in busy.values() for (_, end), (next_start, _) in pairwise(sorted(intervals)))
