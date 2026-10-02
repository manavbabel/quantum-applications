# Sudoku with quantum annealing

Solves n x n sudoku (Latin squares for prime n; boxes of br x bc with
br * bc = n otherwise: 4 -> 2x2, 6 -> 2x3, 9 -> 3x3, 16 -> 4x4) by encoding it as
a QUBO whose zero-energy states are exactly the solutions, then running
quantum annealing on one of three backends:

| backend | what it is | size limit |
|---|---|---|
| `anneal_sqa` | Simulated quantum annealing: path-integral Monte Carlo of the transverse-field Ising model | none (polynomial cost) |
| `anneal_statevector` | Exact Schrodinger evolution of H(s) = (1-s)(-sum X) + s H_problem | 20 qubits |
| `anneal_qpu` | D-Wave Advantage via `EmbeddingComposite(DWaveSampler())`; one job per solve | solves up to ~150 logical qubits; embeds ~250 |

## Files

- `sudoku.py`: parsing grids, the QUBO encoding (`build_bqm`) and decoding samples back into grids
- `annealers.py`: the three backends; each takes a dimod BQM and returns a dimod SampleSet
- `puzzles/`: example puzzles
- `test_sudoku.py`: checks the encoding on every state of a small puzzle

## Usage

From this folder:

```python
import numpy as np
from annealers import anneal_sqa
from sudoku import build_bqm, decode, fmt, is_solution, parse

np.random.seed(1)  # the simulated backends use numpy's global RNG
puzzle = parse(open("puzzles/6x6.txt").read())
sampleset = anneal_sqa(build_bqm(puzzle), stop_energy=0)
grid = decode(sampleset.first.sample, puzzle)
print(fmt(grid))
print(is_solution(grid, puzzle))
```

`anneal_statevector` takes the same BQM up to 20 qubits (`puzzles/3x3.txt`),
and `anneal_qpu` sends it to D-Wave as one job, which needs a D-Wave account
(see the top-level README). For a 9x9 on the QPU, pass a longer anneal:
`anneal_qpu(bqm, annealing_time=200)`.

Puzzle files have one row per line (or rows separated by `/`), with `.`, `0` or
`_` for blanks, and whitespace-separated tokens when n > 9.

Run the test with `uv run pytest sudoku`.

## Choosing the algorithm

Resource numbers are for a 4x4 puzzle with 9 blanks (an earlier version of
`puzzles/4x4.txt`) and the Wikipedia 9x9 (`puzzles/9x9_easy.txt`, 51 blanks).

| approach / hardware | resources (4x4 / 9x9) | feasibility today | verdict |
|---|---|---|---|
| **Grover search** (gate model) | 18 / 204 data qubits (binary digits) + one ancilla per "!=" check (18 / 313); ~10^11 oracle calls for 9x9 even after pruning | 4x4 already exceeds 20 simulated qubits; oracle depth (multi-controlled gates) is thousands of 2-qubit gates, far beyond NISQ fidelity | only a toy; quadratic speedup over brute force, which is the wrong baseline |
| **QAOA / VQE** (gate model) | 15 / 153 qubits, 22 / 650 ZZ terms per layer (same QUBO) | heavy-hex routing multiplies depth; ground-state probability decays exponentially with size for constrained problems; > 20 qubits cannot be simulated here | poor likelihood of success beyond 3x3 |
| **Gaussian boson sampling** (photonic) | solution = independent set of size #blanks in the 15 / 153-node conflict graph; GBS samples dense subgraphs as seeds for a classical max-clique search | simulation needs hafnians of ~51-photon states; no public programmable device | heuristic at best, classical post-processing does the work |
| **Rydberg atoms** (analog MIS) | same conflict graph, but it is not unit-disk: needs gadget embedding with quadratic overhead | ~256 atoms; no access here | promising natively, but too small once gadgets are added |
| **Quantum backtracking** (fault-tolerant) | near-quadratic speedup over classical backtracking | needs error-corrected hardware | future work |
| **Quantum annealing** (D-Wave) | 15 / 153 logical qubits, 16 / 628 physical after embedding | 5,627-qubit Advantage available; QUBO is native | selected |

We chose annealing because the QUBO runs on the hardware as it is, the qubit
count grows only with the number of *candidate* values, hardware of the needed
size exists, and a ground-state energy of 0 certifies a solution immediately.

## Encoding

One binary variable x[r, c, v] per blank cell and candidate value v. Candidates
exclude values already given in the cell's row, column or box, which is the same
as substituting the clues into the full one-hot QUBO. There is no further
constraint propagation, so the classical preprocessing does not solve the
puzzle. Each "exactly one" constraint (one value per cell, and each missing
value once per row, column and box) adds (sum(group) - 1)^2, so the valid
solutions are the energy-0 states, and every violation costs at least 1.
`test_sudoku.py` checks this on every state of a small puzzle with
`dimod.ExactSolver`.

## Results

Measured on 2026-09-25 with an earlier version of this code. The 4x4 and 6x6
puzzles in `puzzles/` have changed since: they now need 26 and 46 qubits,
against the 15 and 52 below. SQA uses its default settings (16 Trotter slices,
beta=160, Gamma 0.5 -> 0.001 over 2000 sweeps); "solved" counts independent
reads that reached energy 0. Random puzzles are cleared at random from a valid
grid (they may have several solutions).

**SQA**

| puzzle | blanks | qubits | couplers | solved | s/read |
|---|---|---|---|---|---|
| 3x3 | 8 | 20 | 48 | 10/10 | 0.06 |
| 4x4 | 9 | 15 | 22 | 10/10 | 0.04 |
| 6x6 | 22 | 52 | 154 | 10/10 | 0.18 |
| 9x9 easy (Wikipedia) | 51 | 153 | 650 | 10/10 | 0.63 |
| random 9x9 | 61 | 288 | 2109 | 10/10 | 1.6 |
| random 12x12 | 72 | 196 | 792 | 10/10 | 0.75 |
| random 16x16 | 128 | 459 | 2661 | 10/10 | 2.4 |
| random 25x25 | 250 | 846 | 4669 | 5/5 | 4.6 |
| random 16x16 | 154 | 763 | 6096 | 0/5 (1/3 with 10k sweeps) | 5.6 (29) |
| random 25x25 | 312 | 1555 | 13223 | 0/5 | 8.1 |
| 9x9 hard (Inkala, 21 clues) | 60 | 254 | 1570 | 0/10 (another 0/20 with 10k-30k sweeps; always stuck at E=4) | 1.3 |

**Statevector** (exact quantum dynamics, total time 100, 1000 Trotter steps):
4x4 (15 qubits) ground-state probability 0.996 in ~7 s; 3x3 (20 qubits,
4 degenerate solutions) 0.734 in ~130 s (wall clock, including JIT compile).

**D-Wave Advantage_system4** (1000 reads per job, 6 jobs total)

| puzzle | logical | physical | max chain | solved reads |
|---|---|---|---|---|
| 3x3 | 20 | 28 | 2 | 993/1000 |
| 4x4 | 15 | 16 | 2 | 998/1000 |
| 6x6 | 52 | 106 | 4 | 783/1000 |
| 9x9 easy, 20 us anneal | 153 | 628 | 8 | 1/1000 |
| 9x9 easy, 200 us anneal | 153 | 624 | 9 | 5/1000 |
| 9x9 hard (Inkala) | 254 | 2196 | 19 | 0/1000 (best E=46) |

## Caveats

- SQA is a classical Monte Carlo method that samples the annealer's quantum
  thermal state along the schedule; it does not simulate unitary dynamics, and
  16 slices is a coarse Trotterisation. It is the usual stand-in for a quantum
  annealer at sizes a statevector cannot reach, and says nothing about speedup.
- Difficulty depends more on the puzzle's structure than its size: sparse 25x25
  grids solve, but a minimal-clue 9x9 with a unique solution defeats every
  backend. Hard puzzles like that need hybrid or classical solvers.
- On the QPU, 9x9 is the practical ceiling: chains of 8-19 qubits and analog
  noise make success rare above ~150 logical qubits.
