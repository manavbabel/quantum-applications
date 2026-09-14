# stores the problem itself with some useful methods

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from graphlib import TopologicalSorter
from itertools import pairwise

# each task is a tuple of (machine, duration, dependencies), where dependencies is a tuple of task indices
Task = tuple[int, int, tuple[int, ...]]


@dataclass
class Problem:
    tasks: list[Task]

    def __post_init__(self):
        # validate
        for task in self.tasks:
            machine, duration, dependencies = task
            if machine < 0:
                raise ValueError(f"Machine index must be non-negative, got {machine}")
            if duration < 0:
                raise ValueError(f"Duration must be positive, got {duration}")
            for dependency in dependencies:
                if dependency < 0 or dependency >= len(self.tasks):
                    raise ValueError(f"Dependency index {dependency} out of bounds for tasks of length {len(self.tasks)}")

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
    def longest_chain(self, task_index, adjacency):
        return max((self.tasks[j][1] + self.longest_chain(j, adjacency) for j in adjacency[task_index]), default=0)

    # given an adjacency list (parents or children) and a task
    # find all tasks reachable from this task, not including the task itself
    def reachable(self, task_index, adjacency):
        return set(adjacency[task_index]).union(*(self.reachable(j, adjacency) for j in adjacency[task_index]))

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
