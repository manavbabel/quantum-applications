# solver base class
# solvers take a Problem and return a SolverResult
# solvers requiring a QUBO take a Formulation
# other solvers just work with problem.tasks

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from formulation import Formulation, build_formulation, decode


@dataclass
class SolverResult:
    # result from a single solve call
    # calls counts shots, samples etc.
    # named in metadata['calls_unit']
    solver: str
    start_times: list | None  # the solution
    makespan: int | None  # solution metric
    feasible: bool
    seconds: float  # TTS
    calls: int
    formulation: Formulation
    metadata: dict = field(default_factory=dict)

    def pprint(self):
        print(f"{self.solver} result:")
        print(f" * makespan:    {self.makespan}")
        print(f" * feasible:    {self.feasible}")
        print(f" * start times: {self.start_times}")
        print(f" * time taken:  {self.seconds}")
        print(f" * metadata:    {json.dumps(self.metadata, indent=4)}")


class Solver(ABC):
    # subclass this by implementing _solve
    # solve handles timing, decoding, validation

    name = "solver"
    # whether this solver consumes the one-hot QUBO rather than modelling the tasks directly
    uses_qubo = True
    _formulation = None

    @abstractmethod
    def _solve(self, problem):
        # Return `(start_times, calls, metadata)`. `start_times` covers the problem's own tasks; any trailing makespan task is trimmed by `solve`
        pass

    # the problem's QUBO, built once per solve and shared with _solve
    def formulation(self, problem):
        if self._formulation is None:
            self._formulation = build_formulation(problem)
        return self._formulation

    def solve(self, problem):

        self._formulation = None  # a solver can be reused on a different problem
        formulation = self.formulation(problem) if self.uses_qubo else None

        started = time.perf_counter()
        start_times, calls, metadata = self._solve(problem)
        seconds = time.perf_counter() - started

        if start_times is not None:
            start_times = list(start_times[: len(problem)])
        feasible = problem.is_feasible(start_times)

        return SolverResult(
            solver=self.name,
            start_times=start_times,
            makespan=problem.makespan(start_times) if feasible else None,
            feasible=feasible,
            seconds=seconds,
            calls=calls,
            formulation=formulation,
            metadata=metadata,
        )


class SolverUnavailable(RuntimeError):
    # can catch this to ensure a run doesn't crash
    pass


def best_feasible(sample_bits, formulation, problem, best=None):
    # from a set of bitstrings, find the feasible schedule with the lowest makespan
    # returns (makespan, start_times)
    # if best is provided, returns that if nothing beats it
    # so it can be kept over many optimisations

    # only the real tasks are decoded, and they are ranked by their own makespan, not by energy
    # the makespan task is bookkeeping that only steers the sampler towards short schedules,
    # so a sample that leaves it unset or late still holds a valid schedule
    for bits in sample_bits:
        start_times = decode(bits, formulation)[: len(problem)]
        if not problem.is_feasible(start_times):
            continue
        makespan = problem.makespan(start_times)
        if best is None or makespan < best[0]:
            best = (makespan, start_times)
    return best
