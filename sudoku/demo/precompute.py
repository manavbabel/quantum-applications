"""Builds data.js for the booth demo: puzzles, annealing recordings, chip mappings and fallback results.

    uv run python sudoku/demo/precompute.py [--solver NAME] [--qpu-check]

Fetching the solver's properties is free. --qpu-check instead sends each problem stored in data.js to
the QPU once (15 jobs of 100 reads) and prints the share of reads that solved it.
"""

import argparse
import json
import os
import sys
import time

import minorminer
import numpy as np
from dwave.embedding import embed_bqm
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.graphs import pegasus_graph, pegasus_layout, zephyr_graph, zephyr_layout
from dwave.system import DWaveSampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from annealers import _sqa_inputs, _sqa_read, anneal_sqa
from sudoku import box_shape, build_bqm, candidates, constraint_groups, random_puzzle

DATA = os.path.join(HERE, "data.js")
SEED = 2026
EMBED_SEED = 1
SIZES = {4: (10, False), 6: (22, False), 9: (40, True)}  # blanks, and whether naked singles must solve it
PER_SIZE = 5
# Recording settings, chosen for the display: each slice starts hot (beta / slices = 0.125) and the
# slices lock together as Gamma falls, ending at beta = 16. The fallback uses anneal_sqa's defaults.
REC_SLICES, REC_BETA, REC_GAMMA, REC_SWEEPS = 128, 16.0, (16.0, 0.007), 312
RECORD_EVERY = 4  # frames: the random start, 78 snapshots and the read-out
MAX_SEEDS = 50
ANNEALING_TIME = {4: 20.0, 6: 20.0, 9: 200.0}  # microseconds
NUM_READS = 100


def count_solutions(grid, limit=2):
    """Number of solutions of grid, counting no further than limit."""
    cand = candidates(grid)
    if not cand:
        return 1
    cell = min(cand, key=lambda k: len(cand[k]))  # branch on the most constrained cell
    total = 0
    for v in cand[cell]:
        grid[cell] = v
        total += count_solutions(grid, limit - total)
        grid[cell] = 0
        if total >= limit:
            break
    return total


def solves_by_singles(grid):
    """True if repeatedly filling the cells that have one candidate completes the grid."""
    grid = grid.copy()
    while cand := candidates(grid):
        singles = {cell: vs[0] for cell, vs in cand.items() if len(vs) == 1}
        if not singles:
            return False
        for cell, v in singles.items():
            grid[cell] = v
    return True


def make_puzzles(n, blanks, singles):
    puzzles = []
    while len(puzzles) < PER_SIZE:
        grid, solution = random_puzzle(n, blanks)
        if (not singles or solves_by_singles(grid)) and count_solutions(grid.copy()) == 1:
            puzzles.append((grid, solution))
    return puzzles


def display_gammas():
    """Gamma from REC_GAMMA[0] to REC_GAMMA[1], spaced so that the slice coupling
    Jp = 1/2 ln coth(beta Gamma / P) rises linearly. A linear Gamma leaves the slices almost
    uncoupled until the last few frames, so every puzzle would settle at the very end."""
    bp = REC_BETA / REC_SLICES
    jp = np.linspace(*(0.5 * np.log(1 / np.tanh(bp * g)) for g in REC_GAMMA), REC_SWEEPS)
    return np.arctanh(np.exp(-2 * jp)) / bp


def record(bqm, variables, groups, cells):
    """Frames of one SQA read that ends at energy 0 with every slice agreeing, trying seeds from the
    global RNG.

    Each frame has p, the share of Trotter slices with each variable set, and heat, each blank
    cell's broken share of the groups it is in, averaged over the slices. The first frame is the
    random start and the last is the read-out.
    """
    h, indptr, indices, data = _sqa_inputs(bqm, variables)
    gammas = display_gammas()
    cell_index = {cell: k for k, cell in enumerate(cells)}
    in_group = np.zeros((len(groups), len(variables)))
    cell_in_group = np.zeros((len(cells), len(groups)))
    for k, g in enumerate(groups):
        in_group[k, g] = 1
        cell_in_group[[cell_index[variables[i][:2]] for i in g], k] = 1
    cell_in_group /= cell_in_group.sum(axis=1, keepdims=True)

    def frame(bits):
        broken = (bits @ in_group.T) != 1
        heat = (broken @ cell_in_group.T).mean(axis=0)
        return {"p": np.round(bits.mean(axis=0), 3).tolist(), "heat": np.round(heat, 3).tolist()}

    for tries in range(1, MAX_SEEDS + 1):
        spins, history = _sqa_read(
            h, indptr, indices, data, REC_SLICES, REC_BETA, gammas, np.random.randint(2**31), RECORD_EVERY
        )
        bits = ((spins + 1) // 2).astype(np.int8)
        energies = bqm.energies((bits, variables))
        best = int(np.argmin(energies))
        if energies[best] <= 1e-9 and np.all(bits == bits[best]):
            frames = [frame((s + 1) / 2) for s in history] + [frame(bits[best : best + 1])]
            return frames, tries
    raise SystemExit(f"[FAIL] no read ended at energy 0 with all slices agreeing in {MAX_SEEDS} seeds")


def chip(sampler):
    """Working qubits with drawing coordinates in [0, 1] (y down), and working couplers."""
    topology = sampler.properties["topology"]
    if topology["type"] == "zephyr":
        m, t = topology["shape"]
        pos = zephyr_layout(zephyr_graph(m, t, node_list=sampler.nodelist, edge_list=sampler.edgelist))
    else:
        m = topology["shape"][0]
        pos = pegasus_layout(pegasus_graph(m, node_list=sampler.nodelist, edge_list=sampler.edgelist))
    xy = np.array([pos[q] for q in sampler.nodelist]) * [1, -1]  # the layouts put y up
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0)).max()
    return {
        "solver": sampler.solver.name,
        "region": sampler.client.config.region,
        "topology": topology["type"],
        "qubits": [int(q) for q in sampler.nodelist],
        "xy": np.round(xy, 4).tolist(),
        "couplers": [[int(p), int(q)] for p, q in sampler.edgelist],
    }


def qpu_problem(bqm, variables, sampler, n):
    """Minor embedding and the physical Ising problem, as EmbeddingComposite would build them."""
    source = list(bqm.quadratic) + [(v, v) for v in variables]  # self-loops embed variables with no couplers
    embedding = minorminer.find_embedding(source, sampler.edgelist, random_seed=EMBED_SEED)
    if not embedding:
        raise SystemExit(f"[FAIL] no embedding found for a {n}x{n} puzzle")
    spin = bqm.spin
    strength = uniform_torque_compensation(spin)
    physical = embed_bqm(spin, embedding, sampler.adjacency, chain_strength=strength)
    low, high = sampler.properties["annealing_time_range"]
    return {
        "chains": [[int(q) for q in embedding[v]] for v in variables],
        "h": [[int(q), round(float(b), 6)] for q, b in physical.linear.items()],
        "J": [[int(p), int(q), round(float(b), 6)] for (p, q), b in physical.quadratic.items()],
        "chain_strength": float(strength),
        "num_reads": NUM_READS,
        "annealing_time": float(min(max(ANNEALING_TIME[n], low), high)),
    }


def puzzle_entry(grid, solution, sampler):
    n = len(grid)
    bqm = build_bqm(grid)
    variables = list(bqm.variables)
    index = {x: i for i, x in enumerate(variables)}
    groups = [[index[x] for x in g] for g in constraint_groups(grid)]
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(grid == 0))]

    frames, tries = record(bqm, variables, groups, cells)
    sampleset = anneal_sqa(bqm, num_reads=NUM_READS)
    if sampleset.first.energy > 1e-9:
        raise SystemExit(f"[FAIL] no fallback read of a {n}x{n} puzzle reached energy 0")

    entry = {
        "n": n,
        "box": list(box_shape(n)),
        "grid": grid.tolist(),
        "solution": solution.tolist(),
        "variables": [[int(r), int(c), int(v)] for r, c, v in variables],
        "groups": groups,
        "num_couplers": len(bqm.quadratic),
        "cells": [list(cell) for cell in cells],
        "frames": frames,
        "qpu": qpu_problem(bqm, variables, sampler, n),
        "fallback": {
            "energies": [round(float(e), 3) for e in sampleset.record.energy],
            "best": [int(sampleset.first.sample[v]) for v in variables],
        },
    }
    chains = [len(chain) for chain in entry["qpu"]["chains"]]
    solved = int(np.sum(sampleset.record.energy <= 1e-9))
    print(
        f"  {len(variables)} qubits, {entry['num_couplers']} couplers, recorded on seed {tries}, "
        f"fallback {solved}/{NUM_READS} solved, {sum(chains)} physical qubits, longest chain {max(chains)}"
    )
    return entry


def qpu_check():
    """Send each stored problem once and print the share of reads that broke no rules."""
    with open(DATA) as f:
        text = f.read()
    data = json.loads(text[text.index("=") + 1 : text.rindex(";")])
    sampler = DWaveSampler(solver=data["chip"]["solver"])
    for size, entries in data["puzzles"].items():
        for i, p in enumerate(entries):
            qpu = p["qpu"]
            sampleset = sampler.sample_ising(
                {q: b for q, b in qpu["h"]},
                {(q1, q2): b for q1, q2, b in qpu["J"]},
                num_reads=qpu["num_reads"],
                annealing_time=qpu["annealing_time"],
                label="nqcc-sudoku-demo-check",
            )
            spins = np.repeat(sampleset.record.sample, sampleset.record.num_occurrences, axis=0)
            column = {q: k for k, q in enumerate(sampleset.variables)}
            # majority vote over each chain; a tie gives 0
            bits = np.stack(
                [spins[:, [column[q] for q in chain]].sum(axis=1) > 0 for chain in qpu["chains"]], axis=1
            )
            energy = sum((bits[:, g].sum(axis=1) - 1) ** 2 for g in p["groups"])
            seconds = sampleset.info["timing"]["qpu_access_time"] / 1e6
            print(
                f"{size}x{size} #{i}: {np.mean(energy == 0):.0%} of {len(bits)} reads solved, "
                f"qpu_access_time {seconds:.3f} s"
            )


def main():
    parser = argparse.ArgumentParser(description="Build data.js for the sudoku annealing demo.")
    parser.add_argument(
        "--solver", default="Advantage2_system1", help="D-Wave solver to map the puzzles onto"
    )
    parser.add_argument(
        "--qpu-check",
        action="store_true",
        help="send each problem in data.js to the QPU once (costs QPU time)",
    )
    args = parser.parse_args()
    if args.qpu_check:
        qpu_check()
        return

    start = time.perf_counter()
    np.random.seed(SEED)
    # every puzzle first, so the puzzles do not depend on how many seeds the recordings take
    puzzles = {n: make_puzzles(n, blanks, singles) for n, (blanks, singles) in SIZES.items()}
    sampler = DWaveSampler(solver=args.solver)
    data = {"chip": chip(sampler), "puzzles": {}}
    print(f"{args.solver} ({data['chip']['region']}): {len(data['chip']['qubits'])} working qubits")
    for n, entries in puzzles.items():
        data["puzzles"][str(n)] = []
        for i, (grid, solution) in enumerate(entries):
            print(f"{n}x{n} #{i}, {SIZES[n][0]} blanks:")
            data["puzzles"][str(n)].append(puzzle_entry(grid, solution, sampler))

    with open(DATA, "w", newline="\n") as f:
        f.write("window.DEMO = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"[OK] wrote {DATA} ({os.path.getsize(DATA) / 1e6:.2f} MB) in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
