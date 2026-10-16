"""Quantum annealing of a dimod BQM, exactly, by simulated quantum annealing (SQA), or on a D-Wave QPU.

The simulators use numpy's global RNG, so np.random.seed() makes runs reproducible.
"""

import dimod
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
from numba import njit

MAX_STATEVECTOR_QUBITS = 20


def anneal_statevector(bqm, num_reads=100, total_time=100.0, steps=1000):
    """Exact Schrodinger evolution under H(s) = (1 - s)(-sum X) + s H_problem, Trotterised.

    Basis state k sets variable i to bit i of k, so the qubit order is little-endian.
    """
    variables = list(bqm.variables)
    n = len(variables)
    if n > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"{n} qubits exceeds the statevector limit of {MAX_STATEVECTOR_QUBITS}; use sqa")

    bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.int8)  # (2^n, n)
    energy = bqm.energies((bits, variables))
    # scaled so that total_time is in units of the inverse problem energy scale
    scale = max(np.abs(list(bqm.linear.values()) + list(bqm.quadratic.values())), default=1.0)
    # H_problem is diagonal with few distinct levels; shifting it only changes a global phase
    levels, level_of = np.unique((energy - energy.min()) / scale, return_inverse=True)

    psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)  # |+>^n, ground state of -sum X
    _evolve(psi, n, levels, level_of, total_time, steps)

    probs = np.abs(psi) ** 2
    probs /= probs.sum()
    picks = np.random.choice(2**n, size=num_reads, p=probs)
    ss = dimod.SampleSet.from_samples_bqm((bits[picks], variables), bqm)
    ss.info["ground_state_probability"] = float(probs[energy <= energy.min() + 1e-9].sum())
    return ss


@njit(cache=True)
def _evolve(psi, n, levels, level_of, total_time, steps):
    """In place, psi <- prod_k exp(-i dt s_k H_problem) exp(-i dt (1 - s_k)(-sum X))."""
    dt = total_time / steps
    for k in range(steps):
        s = (k + 0.5) / steps
        c, si = np.cos((1 - s) * dt), 1j * np.sin((1 - s) * dt)  # exp(i theta X) = c + i sin X
        for q in range(n):
            stride = 1 << q
            for base in range(0, psi.size, 2 * stride):
                for j in range(base, base + stride):
                    a0, a1 = psi[j], psi[j + stride]
                    psi[j] = c * a0 + si * a1
                    psi[j + stride] = si * a0 + c * a1
        phase = np.exp(-1j * s * dt * levels)
        for j in range(psi.size):
            psi[j] *= phase[level_of[j]]


@njit(cache=True)
def _sqa_read(h, indptr, indices, data, slices, beta, gammas, seed, record_every):
    """One SQA run on sum h_i s_i + sum J_ij s_i s_j, J in symmetric CSR (Martonak et al., PRB 66, 094203).

    Returns the final slices, and snapshots of them every record_every sweeps if that is above 0.
    """
    np.random.seed(seed)
    n = h.size
    s = np.where(np.random.random((slices, n)) < 0.5, -1.0, 1.0)
    bp = beta / slices
    frames = 1 + (gammas.size + record_every - 1) // record_every if record_every > 0 else 0
    history = np.empty((frames, slices, n))
    if record_every > 0:
        history[0] = s
    for t in range(gammas.size):
        gamma = gammas[t]
        # neighbouring Trotter slices are coupled ferromagnetically, more strongly as Gamma falls
        jp = 0.5 * np.log(1.0 / np.tanh(beta * gamma / slices))
        # single-spin flips within each slice
        for k in range(slices):
            up, dn = (k + 1) % slices, (k + slices - 1) % slices
            for i in range(n):
                f = h[i]
                for p in range(indptr[i], indptr[i + 1]):
                    f += data[p] * s[k, indices[p]]
                dx = 2.0 * s[k, i] * (bp * f - jp * (s[up, i] + s[dn, i]))
                if dx >= 0.0 or np.random.random() < np.exp(dx):
                    s[k, i] = -s[k, i]
        # global moves: flip spin i in every slice, which leaves the slice coupling unchanged
        for i in range(n):
            dx = 0.0
            for k in range(slices):
                f = h[i]
                for p in range(indptr[i], indptr[i + 1]):
                    f += data[p] * s[k, indices[p]]
                dx += 2.0 * bp * s[k, i] * f
            if dx >= 0.0 or np.random.random() < np.exp(dx):
                for k in range(slices):
                    s[k, i] = -s[k, i]
        if record_every > 0 and ((t + 1) % record_every == 0 or t == gammas.size - 1):
            history[1 + t // record_every] = s
    return s, history


def _sqa_inputs(bqm, variables):
    """h and symmetric CSR J (indptr, indices, data) for _sqa_read, divided by the largest |coefficient|."""
    h, (row, col, j), _ = bqm.spin.to_numpy_vectors(variables)
    scale = max(np.abs(h).max(initial=0), np.abs(j).max(initial=0)) or 1.0
    rows, cols = np.concatenate([row, col]), np.concatenate([col, row])
    order = np.argsort(rows, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=len(h)))])
    indices = cols[order].astype(np.int64)
    data = np.concatenate([j, j])[order] / scale
    return h / scale, indptr, indices, data


def anneal_sqa(bqm, num_reads=20, sweeps=2000, slices=16, beta=160.0, gamma=(0.5, 1e-3), stop_energy=None):
    """SQA with Gamma falling linearly over the sweeps, stopping early once a read reaches stop_energy.

    Tuned on 6x6 and 9x9 puzzles: beta / slices from 5 to 20 works best, and below 1 or at 40 fails.
    """
    variables = list(bqm.variables)
    h, indptr, indices, data = _sqa_inputs(bqm, variables)
    gammas = np.linspace(gamma[0], gamma[1], sweeps)

    samples, energies = [], []
    for _ in range(num_reads):
        spins, _ = _sqa_read(h, indptr, indices, data, slices, beta, gammas, np.random.randint(2**31), 0)
        bits = ((spins + 1) // 2).astype(np.int8)
        slice_e = bqm.energies((bits, variables))
        best = int(np.argmin(slice_e))
        samples.append(bits[best])
        energies.append(slice_e[best])
        if stop_energy is not None and slice_e[best] <= stop_energy + 1e-9:
            break
    return dimod.SampleSet.from_samples((np.array(samples), variables), dimod.BINARY, energies)


def anneal_qpu(bqm, num_reads=1000, solver="Advantage2_system1", **kwargs):
    """Submit one job to a D-Wave QPU, by default the latest generation, embedded by EmbeddingComposite."""
    sampler = EmbeddingComposite(DWaveSampler(solver=solver))
    return sampler.sample(bqm, num_reads=num_reads, label="quantum-sudoku", return_embedding=True, **kwargs)
