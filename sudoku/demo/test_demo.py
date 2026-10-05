import json
import os
import sys

import networkx as nx
import numpy as np
import pytest
from dwave.embedding import verify_embedding

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from precompute import count_solutions  # noqa: E402
from sudoku import box_shape, build_bqm, constraint_groups, decode, is_solution  # noqa: E402

with open(os.path.join(HERE, "data.js")) as f:
    text = f.read()
DATA = json.loads(text[text.index("=") + 1:text.rindex(";")])
PUZZLES = [p for entries in DATA["puzzles"].values() for p in entries]
IDS = [f"{p['n']}x{p['n']}-{i}" for entries in DATA["puzzles"].values() for i, p in enumerate(entries)]


def group_energy(bits, groups):
    return sum((bits[:, g].sum(axis=1) - 1) ** 2 for g in groups)


def one_hot(p):
    solution = np.array(p["solution"])
    return [int(solution[r, c] == v) for r, c, v in p["variables"]]


# the page and the server score reads from groups, so this must be the QUBO's energy
def test_group_energy_matches_bqm():
    np.random.seed(0)
    for p in PUZZLES:
        variables = [tuple(x) for x in p["variables"]]
        bits = np.vstack([np.random.randint(2, size=(50, len(variables))), one_hot(p)])
        bqm = build_bqm(np.array(p["grid"]))
        assert np.array_equal(group_energy(bits, p["groups"]), bqm.energies((bits, variables)))


# otherwise "exactly one solution" below proves nothing
def test_count_solutions_stops_at_two():
    assert count_solutions(np.zeros((4, 4), dtype=int)) == 2


def test_chip_drawing():
    chip = DATA["chip"]
    assert len(chip["xy"]) == len(chip["qubits"])
    assert all(0 <= x <= 1 for xy in chip["xy"] for x in xy)


@pytest.mark.parametrize("p", PUZZLES, ids=IDS)
def test_puzzle(p):
    grid, solution = np.array(p["grid"]), np.array(p["solution"])
    bqm = build_bqm(grid)
    variables = [tuple(x) for x in p["variables"]]
    index = {x: i for i, x in enumerate(variables)}
    assert variables == list(bqm.variables)
    assert p["groups"] == [[index[x] for x in g] for g in constraint_groups(grid)]
    assert p["num_couplers"] == len(bqm.quadratic)
    assert p["box"] == list(box_shape(p["n"]))
    assert p["cells"] == np.argwhere(grid == 0).tolist()

    assert count_solutions(grid.copy()) == 1
    assert is_solution(solution, grid)

    assert p["frames"][-1]["p"] == one_hot(p)
    # the recording starts from the random state and every slice agrees before the read-out
    assert max(p["frames"][0]["heat"]) > 0.4
    assert p["frames"][-2]["p"] == p["frames"][-1]["p"]
    for frame in p["frames"]:
        assert len(frame["p"]) == len(variables) and len(frame["heat"]) == len(p["cells"])
        assert all(0 <= x <= 1 for x in frame["p"] + frame["heat"])

    fallback = p["fallback"]
    assert len(fallback["energies"]) == 100 and min(fallback["energies"]) == 0
    assert np.array_equal(decode(dict(zip(variables, fallback["best"])), grid), solution)
    assert fallback["best"] == one_hot(p)


@pytest.mark.parametrize("p", PUZZLES, ids=IDS)
def test_embedding(p):
    chip = nx.Graph(DATA["chip"]["couplers"])
    chip.add_nodes_from(DATA["chip"]["qubits"])
    variables = [tuple(x) for x in p["variables"]]
    index = {x: i for i, x in enumerate(variables)}
    source = nx.Graph([(index[a], index[b]) for a, b in build_bqm(np.array(p["grid"])).quadratic])
    source.add_nodes_from(range(len(variables)))
    chains = p["qpu"]["chains"]
    assert verify_embedding(dict(enumerate(chains)), source, chip)

    # the physical problem sits on the chains' qubits and working couplers only
    assert {q for q, _ in p["qpu"]["h"]} == {q for chain in chains for q in chain}
    assert all(chip.has_edge(q1, q2) for q1, q2, _ in p["qpu"]["J"])
