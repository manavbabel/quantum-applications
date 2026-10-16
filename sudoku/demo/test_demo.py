"""Checks on data.js: the demo's puzzles, recordings and chip mappings."""

import os
import sys

import networkx as nx
import numpy as np
import pytest
from dwave.embedding import verify_embedding

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from precompute import count_solutions
from serve import energy, load_demo
from sudoku import build_bqm, is_solution

DATA = load_demo(os.path.join(HERE, "data.js"))
PUZZLES = [p for entries in DATA["puzzles"].values() for p in entries]
IDS = [f"{p['n']}x{p['n']}-{i}" for entries in DATA["puzzles"].values() for i, p in enumerate(entries)]


def one_hot(p):
    solution = np.array(p["solution"])
    return [int(solution[r, c] == v) for r, c, v in p["variables"]]


# otherwise "exactly one solution" below proves nothing
def test_count_solutions_stops_at_two():
    assert count_solutions(np.zeros((4, 4), dtype=int)) == 2


@pytest.mark.parametrize("p", PUZZLES, ids=IDS)
def test_puzzle(p):
    grid = np.array(p["grid"])
    assert count_solutions(grid.copy()) == 1
    assert is_solution(np.array(p["solution"]), grid)
    # the recording and the fallback both end on the solution
    assert p["frames"][-1]["p"] == one_hot(p)
    assert p["fallback"]["best"] == one_hot(p)


# the server scores D-Wave's reads from the groups, so this must be the QUBO's energy
@pytest.mark.parametrize("p", PUZZLES, ids=IDS)
def test_server_energy_is_the_qubo_energy(p):
    np.random.seed(0)
    bqm = build_bqm(np.array(p["grid"]))
    variables = [tuple(x) for x in p["variables"]]
    assert variables == list(bqm.variables)
    bits = np.vstack([np.random.randint(2, size=(20, len(variables))), one_hot(p)])
    assert [energy(row, p["groups"]) for row in bits] == list(bqm.energies((bits, variables)))


@pytest.mark.parametrize("p", PUZZLES, ids=IDS)
def test_embedding(p):
    chip = nx.Graph(DATA["chip"]["couplers"])
    chip.add_nodes_from(DATA["chip"]["qubits"])
    index = {tuple(x): i for i, x in enumerate(p["variables"])}
    source = nx.Graph([(index[a], index[b]) for a, b in build_bqm(np.array(p["grid"])).quadratic])
    source.add_nodes_from(range(len(index)))
    assert verify_embedding(dict(enumerate(p["qpu"]["chains"])), source, chip)
    # the physical problem uses working couplers only
    assert all(chip.has_edge(a, b) for a, b, _ in p["qpu"]["J"])
