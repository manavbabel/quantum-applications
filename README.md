# Quantum applications

[![CI](https://github.com/manavbabel/quantum-applications/actions/workflows/ci.yml/badge.svg)](https://github.com/manavbabel/quantum-applications/actions/workflows/ci.yml)

This repo contains a series of optimisation problems, from industrial scheduling to sudoku, solved with quantum algorithms and benchmarked against classical solutions. They are written as educational resources for those exploring quantum computing.

Results from real QPUs (IBM and D-Wave) are included, but you will need to supply your own API keys to get more results.

![Job shop scheduling: the share of each method's shots or reads that came out optimal, feasible but longer, or infeasible](job-shop-scheduling/figures/outcomes.png)

*Job shop scheduling on a 21-qubit example: the share of each method's shots or reads that came out optimal, feasible but longer, or infeasible. Random guessing is the baseline.*

## Examples

| folder | problem | methods | status |
|---|---|---|---|
| [`job-shop-scheduling`](job-shop-scheduling) | schedule tasks on machines to finish as soon as possible | greedy, brute force, Gurobi, QAOA and LR-QAOA (simulated and IBM), annealing (simulated and D-Wave) | done |
| [`sudoku`](sudoku) | fill in a sudoku grid | exact simulation, simulated quantum annealing, D-Wave, and an [interactive demo](sudoku/demo) | done |
| unit commitment | decide which power stations to run | MAX-LINSAT with decoded quantum interferometry (DQI) | planned |
| finance | a problem still to be chosen | quantum-accelerated Monte Carlo (QCMC) | planned |

## What we've found so far

At sizes which can be simulated or run on real quantum devices, there is no evidence of any quantum advantage. Nevertheless, the quantum algorithms are shown to work, consistently getting optimal solutions in noiseless simulation, and approaching that on real devices for smaller problems.

- Job shop scheduling: Gurobi proves every optimum in under a millisecond. D-Wave's Advantage2 finds the optimum on 70 to 100% of random problems of up to 8 tasks, and half of its reads on the 21-qubit example are optimal. IBM's circuits, at over 2,000 two-qubit gates, do worse than random guessing.
- Sudoku: D-Wave solves the 3x3 to 6x6 puzzles on nearly every read, but the easy 9x9 on only a handful of reads in 1,000. A hard 9x9 with 21 clues defeats every method here, simulated or real.

## New to quantum?

Both examples turn the problem into the same form, a QUBO, and then hand it to different solvers. A few terms come up throughout:

| term | meaning |
|---|---|
| qubit | a quantum bit; measured, it gives 0 or 1, but before that it can hold a superposition of both |
| QUBO | a quadratic unconstrained binary optimisation: minimise a sum of terms in binary variables and their pairs; constraints become penalties that raise the energy when broken |
| quantum annealing | start the qubits in an easy superposition, then slowly change the energy landscape into the problem's, so they end up in a low-energy state; D-Wave builds machines that do this |
| QAOA | the quantum approximate optimisation algorithm, which does something similar on gate-based machines such as IBM's, in a fixed number of layers |
| minor embedding | the hardware only links neighbouring qubits, so one problem variable can need a chain of physical qubits |
| shot or read | one run of the quantum computer, giving one candidate answer; we take thousands and keep the best |

The sudoku notebook is the gentler start: one encoding, one algorithm run three ways. The job shop notebook adds gate-based methods and a benchmark against classical solvers.

## Setup

With [uv](https://docs.astral.sh/uv/) installed, clone and run:

```
git clone https://github.com/manavbabel/quantum-applications.git
cd quantum-applications
uv sync
uv run pytest
```

Then open a notebook from its folder, for example:

```
cd sudoku
uv run jupyter lab sudoku.ipynb
```

- Gurobi: `gurobipy` installs with a size-limited licence, which is enough for these examples.
- D-Wave (optional): a [Leap](https://cloud.dwavesys.com/leap/) account, then `uv run dwave setup` to save your API token.
- IBM Quantum (optional): an [IBM Quantum Platform](https://quantum.cloud.ibm.com/) account, then save your API key once with `QiskitRuntimeService.save_account(token=..., instance=...)`.

Without the optional accounts, everything runs on local simulators; each example's README says which cells to skip.

Each example folder has the same layout: a README with the results, a notebook that walks through the method, the plain-Python modules the notebook imports, and tests.

## Plans

I'm planning on completing the unit commitment and finance examples next. After that, the plan is to profile the Python and rewrite the slowest part in Rust with PyO3. Working notes are in [TODO.md](TODO.md).

## AI usage

I've found that AI very much follows the jagged frontier when it comes to quantum computing. The pace of the field, the rapidly changing SDKs, and the relative underspecification of a lot of algorithms mean that simply telling Claude to "solve job-shop scheduling with annealing" won't work. A fair amount of manual work is needed beforehand: defining the problem (as well as what is out of scope), formulating it properly, and designing the pipeline, before you can hand over the more boilerplate tasks.

In this repository, the problem definitions, formulations and pipelines were done manually. AI helped with boilerplate code (figures, basic data analysis, cloud access), the documentation and tidying up.

## Licence

[MIT](LICENSE)
