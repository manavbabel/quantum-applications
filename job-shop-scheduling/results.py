import json
import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt


# raised when a solver can't take a problem (too big, no licence, no device), so a loop can skip it
class SolverUnavailable(RuntimeError):
    pass


@dataclass
class SolverResult:
    solver: str
    makespan: int | None
    feasible: bool
    tts: float  # wall-clock seconds for this solve
    success_probability: float | None  # the chance one run (a shot or a read) finds the optimum
    reps99: float | None  # runs needed to see the optimum with 99% confidence; None if the optimum is unknown
    start_times: list | None
    calls: int
    samples: dict | None = None  # share of runs at each makespan, None for infeasible; None if deterministic
    metadata: dict = field(default_factory=dict)

    def describe(self):
        print(f"{self.solver} result {'FEASIBLE' if self.feasible else 'NOT FEASIBLE'}")
        print(f" * makespan     : {self.makespan}")
        print(f" * TTS          : {self.tts}")
        print(f" * p(optimum)   : {self.success_probability}")
        print(f" * reps99       : {self.reps99}")
        print(f" * calls        : {self.calls}")
        print(f" * metadata     : {json.dumps(self.metadata, indent=4, default=str)}")


# the share of samples (makespan -> weight) at a given makespan, None for infeasible
def fraction(samples, makespan):
    return samples.get(makespan, 0) / sum(samples.values())


# the runs (reads or shots) needed to see the optimum at least once with 99% confidence,
# ln(0.01) / ln(1 - p) when each run finds it with probability p, and at least one
def reps99(p):
    if p is None:
        return None
    if p == 0:
        return math.inf
    return 1.0 if p >= 0.99 else math.log(0.01) / math.log(1 - p)


# turn a solver's output into a SolverResult
# start_times may include the makespan task at the end, which is dropped
# samples (makespan -> weight) is None for deterministic solvers, whose one run finds the optimum or doesn't
# success is measured against problem.optimum, so set that first if the bounds don't already give it
def make_result(problem, solver, start_times, tts, calls, metadata=None, samples=None):
    if start_times is not None:
        start_times = list(start_times)[: len(problem)]
    feasible = problem.is_feasible(start_times)
    makespan = problem.makespan(start_times) if feasible else None
    metadata = dict(metadata or {})

    if samples is not None:
        samples = {key: fraction(samples, key) for key in samples}
        metadata["feasible_fraction"] = 1 - samples.get(None, 0)

    if problem.optimum is None:
        p = None
    elif samples is None:
        p = float(makespan == problem.optimum)
    else:
        p = samples.get(problem.optimum, 0)

    return SolverResult(
        solver=solver,
        makespan=makespan,
        feasible=feasible,
        tts=tts,
        success_probability=p,
        reps99=reps99(p),
        start_times=start_times,
        calls=calls,
        samples=samples,
        metadata=metadata,
    )


# what each sampler's runs came out as: optimal, feasible but longer, or infeasible
def draw_outcomes(results):
    names = list(results)
    optimal = [result.success_probability for result in results.values()]
    infeasible = [result.samples.get(None, 0) for result in results.values()]
    longer = [1 - o - i for o, i in zip(optimal, infeasible)]

    figure, ax = plt.subplots(figsize=(9, 0.45 * len(names) + 1.2))
    left = [0] * len(names)
    for shares, label, colour in [
        (optimal, "optimal", "#0ca30c"),
        (longer, "feasible, longer than optimal", "#fab219"),
        (infeasible, "infeasible", "#d03b3b"),
    ]:
        ax.barh(names, shares, left=left, color=colour, edgecolor="white", linewidth=2, label=label)
        left = [a + b for a, b in zip(left, shares)]
    for y, p in enumerate(optimal):
        ax.text(1.01, y, f"{100 * p:.3g}% optimal", va="center", color="#52514e")

    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of shots or reads")
    ax.legend(ncols=3, loc="lower left", bbox_to_anchor=(0, 1), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
