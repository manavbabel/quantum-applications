"""Sudoku grids, as n x n arrays with 0 for a blank, and their QUBO encoding.

One variable x[r, c, v] per blank cell and candidate value, with energy 0 exactly at the solutions.
"""

import itertools
import math

import dimod
import numpy as np

BLANKS = ".0_"


def parse(text):
    """Parse rows separated by newlines or '/', with '.', '0' or '_' for blanks.

    Cells are single characters, or whitespace-separated tokens when n > 9.
    """
    rows = [
        ln.split() if len(ln.split()) > 1 else list(ln.strip())
        for ln in text.replace("/", "\n").splitlines()
        if ln.strip()
    ]
    grid = np.array([[0 if t in BLANKS else int(t) for t in row] for row in rows])
    n = len(grid)
    if grid.shape != (n, n) or grid.min() < 0 or grid.max() > n:
        raise ValueError(f"expected an n x n grid with values 0..n, got shape {grid.shape}")
    return grid


def fmt(grid):
    """The grid as text, with dots for blanks."""
    w = len(str(len(grid)))
    return "\n".join(" ".join(f"{v if v else '.':>{w}}" for v in row) for row in grid)


def box_shape(n):
    """Box rows and columns, as square as possible (6 -> 2x3); None for prime n, a Latin square."""
    br = max(d for d in range(1, math.isqrt(n) + 1) if n % d == 0)
    return (br, n // br) if br > 1 else None


def units(n):
    """Every row, column and box as a list of (r, c) cells."""
    rows = [[(r, c) for c in range(n)] for r in range(n)]
    cols = [[(r, c) for r in range(n)] for c in range(n)]
    boxes = []
    if box := box_shape(n):
        br, bc = box
        boxes = [
            [(r0 + r, c0 + c) for r in range(br) for c in range(bc)]
            for r0 in range(0, n, br)
            for c0 in range(0, n, bc)
        ]
    return rows + cols + boxes


def is_solution(grid, puzzle=None):
    """True if grid is a complete valid sudoku (agreeing with puzzle's clues)."""
    grid = np.asarray(grid)
    n = len(grid)
    valid = all(sorted(grid[r, c] for r, c in u) == list(range(1, n + 1)) for u in units(n))
    return valid and (puzzle is None or bool(np.all((puzzle == 0) | (puzzle == grid))))


def candidates(puzzle):
    """{(r, c): [values]} for each blank cell, excluding values given in its units.

    This is the same as substituting the clues into the full QUBO; nothing further is propagated.
    """
    n = len(puzzle)
    seen = {}
    for u in units(n):
        given = {puzzle[r, c] for r, c in u} - {0}
        for cell in u:
            seen.setdefault(cell, set()).update(given)
    return {
        (r, c): [v for v in range(1, n + 1) if v not in seen[r, c]]
        for r in range(n)
        for c in range(n)
        if puzzle[r, c] == 0
    }


def constraint_groups(puzzle):
    """The "exactly one" groups of variables (r, c, v)."""
    n = len(puzzle)
    cand = candidates(puzzle)
    groups = [[(r, c, v) for v in vs] for (r, c), vs in cand.items()]  # one value per blank cell
    for u in units(n):
        given = {puzzle[r, c] for r, c in u}
        for v in set(range(1, n + 1)) - given:  # each missing value exactly once per unit
            groups.append([(r, c, v) for r, c in u if v in cand.get((r, c), ())])
    return groups


def build_bqm(puzzle):
    """QUBO over variables (r, c, v); ground-state energy 0 iff solvable."""
    bqm = dimod.BinaryQuadraticModel("BINARY")
    for g in constraint_groups(puzzle):  # (sum x - 1)^2 = 1 - sum x_i + 2 sum_{i<j} x_i x_j, using x^2 = x
        bqm.offset += 1
        for x in g:
            bqm.add_linear(x, -1)
        for a, b in itertools.combinations(g, 2):
            bqm.add_quadratic(a, b, 2)
    return bqm


def decode(sample, puzzle):
    """The puzzle with each set variable (r, c, v) written into its cell."""
    grid = np.array(puzzle)
    for (r, c, v), bit in sample.items():
        if bit:
            grid[r, c] = v
    return grid


def random_puzzle(n, blanks):
    """A random valid n x n grid with `blanks` cells cleared, and its solution, from the global numpy RNG.

    It shuffles the rows, columns and symbols of the pattern (bc*(r%br) + r//br + c) % n.
    """
    br, bc = box_shape(n) or (1, n)
    r = np.concatenate([b * br + np.random.permutation(br) for b in np.random.permutation(n // br)])
    c = np.concatenate([s * bc + np.random.permutation(bc) for s in np.random.permutation(n // bc)])
    pattern = (bc * (r[:, None] % br) + r[:, None] // br + c[None, :]) % n
    solution = np.random.permutation(n)[pattern] + 1
    puzzle = solution.copy()
    puzzle.flat[np.random.choice(n * n, blanks, replace=False)] = 0
    return puzzle, solution
