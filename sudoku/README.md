# Sudoku with quantum annealing

Sudoku is a great game, and I thought it would be quite a good educational tool.

We write an n x n sudoku as a QUBO, whose solutions are the zero energy states, and then solve it by quantum annealing.

There's also an [interactive demo](demo) which I built and used at NewScientist Live 2026, where users are able to solve a sudoku, and then see a visualisation of how the quantum computer does it.

![The share of each puzzle's reads that solved it, for SQA and for D-Wave with a 20 µs and a 200 µs anneal](figures/solved.png)

## Results

From the notebook's run on 16 October 2026:

| puzzle | blanks | qubits | SQA | D-Wave, 20 µs | D-Wave, 200 µs | qubits on the chip | longest chain |
|---|---|---|---|---|---|---|---|
| 3x3 | 8 | 20 | 10/10 | 999/1000 | | 26 | 2 |
| 4x4 | 12 | 26 | 10/10 | 994/1000 | | 34 | 2 |
| 6x6 | 22 | 46 | 10/10 | 1000/1000 | | 69 | 3 |
| 9x9 easy | 51 | 153 | 10/10 | 6/1000 | 2/1000 | 596 | 9 |
| 9x9 hard | 60 | 254 | 0/10 | 0/1000 | 0/1000 | 2,078 | 19 |
| 16x16 random | 128 | 445 | 5/5 | | | | |
| 25x25 random | 250 | 789 | 5/5 | | | | |

Some findings:

- the difficulty seems to depend more on puzzle structure over size, with the hard 9x9 unsolvable with all these methods
- on the real device, small puzzles are solved on almost all reads, but the easy 9x9 is where we start seeing some failures. Results can also be inconsistent between runs

## Running it

From this folder:

```
cd sudoku
uv run jupyter lab sudoku.ipynb
```

The D-Wave cell needs an account (see the [top-level README](../README.md)) and uses about 1.7 s of QPU time; set `use_dwave = False` to skip it. 

Or use the modules directly:

```python
import numpy as np
from annealers import anneal_sqa
from sudoku import build_bqm, decode, fmt, is_solution, parse

np.random.seed(1)  # the simulators use numpy's global RNG
puzzle = parse(open("puzzles/6x6.txt").read())
sampleset = anneal_sqa(build_bqm(puzzle), stop_energy=0)
grid = decode(sampleset.first.sample, puzzle)
print(fmt(grid))
print(is_solution(grid, puzzle))
```

`anneal_statevector` takes the same BQM up to 20 qubits, and `anneal_qpu` sends it to D-Wave as one job of 1,000 reads.

Puzzle files have one row per line (or rows separated by `/`), with `.`, `0` or `_` for blanks, and whitespace-separated tokens when n > 9. Boxes are as square as possible (6x6 has 2x3 boxes), and a prime n has none, which makes a Latin square.

## How it works

We do a quick preprocessing pass, where we eliminate values in the same box, row, or column. Then, we say that each candidate value in each blank cell gets its own binary variable, and we add quadratic terms which penalise forbidden combinations (one value per cell, and exactly one value per row, column, and box). We set the penalty to be $(\text{group sum} - 1)^2$. 

## Why annealing

Resources are for the easy 9x9.

| approach | resources | feasibility today | verdict |
|---|---|---|---|
| Grover search (gate model) | 204 data qubits (binary digits) plus an ancilla for each of 313 "not equal" checks, and about 10^11 oracle calls even after pruning | the oracle's multi-controlled gates need thousands of two-qubit gates, far beyond today's devices | a toy only; its quadratic speed-up is over brute force, but that's not a good baseline |
| QAOA or VQE (gate model) | 153 qubits and 650 ZZ terms per layer, from the same QUBO | routing on IBM's heavy-hex chips multiplies the depth, success falls exponentially with size on constrained problems, and over 20 qubits can't be simulated easily | unlikely to work beyond the 3x3 |
| Gaussian boson sampling (photonic) | resources vary with the algorithm; the BBS would need as many modes as there are possible values | insufficient data to gauge feasibility, and no public devices available | not suitable for this repo |
| Rydberg atoms (analogue) | the same conflict graph, which isn't unit-disk, so it needs gadgets with quadratic overhead | about 256 atoms, and no public access | promising natively, but too small once gadgets are added |
| quantum annealing (D-Wave) | 153 logical qubits, about 600 once embedded | 4,575-qubit Advantage2 available, and the QUBO runs natively | chosen |

We chose annealing because the QUBOs are native, the qubit count grows only with the number of possible values, and suitable hardware exists and is available publicly.

## Files

| file | contents |
|---|---|
| `sudoku.ipynb` | the walkthrough and results |
| `sudoku.py` | parsing grids, the QUBO (`build_bqm`) and decoding samples back into grids |
| `annealers.py` | the three backends; each takes a dimod BQM and returns a dimod SampleSet |
| `puzzles/` | the example puzzles |
| `test_sudoku.py` | checks the encoding on every state of a small puzzle |
| `figures/` | the chart above, written by the notebook |
| `demo/` | the interactive demo; see [its README](demo/README.md) |

## Limits

- SQA is a classical Monte Carlo method that samples the annealer's quantum thermal state along the schedule. It doesn't simulate the real dynamics, and 16 Trotter slices is a coarse approximation. It's the usual stand-in for a quantum annealer at sizes a statevector can't reach, and says nothing about speed-up.
- On the QPU, the 9x9 is where solutions start getting worse: long chains and noise make success rare above about 150 logical qubits. Puzzles as hard as the hard 9x9 need hybrid or classical solvers.
- The random puzzles are cleared at random from a valid grid, so they may have more than one solution.
