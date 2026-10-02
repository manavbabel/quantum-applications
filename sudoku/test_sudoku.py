from collections import Counter

import dimod
import numpy as np

from sudoku import build_bqm, decode, is_solution, random_puzzle


# a valid assignment sets exactly one value in every blank cell, and the grid it gives is solved
def is_valid(sample, puzzle):
    values_set = Counter((r, c) for (r, c, _), bit in sample.items() if bit)
    blanks = int((puzzle == 0).sum())
    one_each = len(values_set) == blanks and all(n == 1 for n in values_set.values())
    return one_each and is_solution(decode(sample, puzzle), puzzle)


# the encoding's claim: the zero-energy states are exactly the valid assignments, and the rest cost at least 1
# checked on every state of a small puzzle (14 variables)
def test_zero_energy_states_are_the_solutions():
    np.random.seed(0)
    puzzle, _ = random_puzzle(4, blanks=8)
    sampleset = dimod.ExactSolver().sample(build_bqm(puzzle))
    for sample, energy in sampleset.data(["sample", "energy"]):
        assert (energy == 0) == is_valid(sample, puzzle)
        assert energy == 0 or energy >= 1
