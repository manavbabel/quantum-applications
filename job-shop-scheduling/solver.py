import json
import math
from dataclasses import dataclass, field

from problem import Problem


# the share of what a sampling solver drew that came out at a given makespan, None for infeasible
# samples is the makespan -> weight tally from best_feasible; each sample (a read or a shot) is a run of its own
def fraction(samples, makespan):
    return samples.get(makespan, 0) / sum(samples.values())


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
            p = fraction(samples, problem.optimum)

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
