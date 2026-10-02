"""Quantum-annealing backends. Each takes a dimod BQM and returns a dimod SampleSet.

All three run the same algorithm, H(s) = A(s) * (-sum_i X_i) + B(s) * H_problem,
with the transverse-field driver switched off as s goes 0 -> 1:

- statevector: exact Schrodinger evolution of the full wavefunction (<= 20 qubits).
- sqa: simulated quantum annealing, i.e. path-integral Monte Carlo of the
  transverse-field Ising model (Martonak, Santoro & Tosatti, PRB 66, 094203).
  It samples the quantum system's thermal state along the anneal rather than
  its unitary dynamics, and scales polynomially, so large grids are feasible.
- qpu: a D-Wave annealer via minor embedding.

Randomness comes from the global numpy RNG so np.random.seed() makes runs
reproducible.
"""

import dimod
import numpy as np
from numba import njit

MAX_STATEVECTOR_QUBITS = 20


def anneal_statevector(bqm, num_reads=100, total_time=100.0, steps=1000):
    """Exact adiabatic evolution, first-order Trotterised.

    Qubit i is variables[i] and is bit i of the basis-state index
    (little-endian), so basis state k assigns variables[i] = (k >> i) & 1.
    Energies are divided by the largest |coefficient| so that total_time is in
    units of the inverse problem energy scale.
    """
    variables = list(bqm.variables)
    n = len(variables)
    if n > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"{n} qubits exceeds the statevector limit of {MAX_STATEVECTOR_QUBITS}; use sqa")

    bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.int8)  # (2^n, n)
    energy = bqm.energies((bits, variables))
    scale = max(np.abs(list(bqm.linear.values()) + list(bqm.quadratic.values())), default=1.0)
    # H_problem is diagonal with few distinct levels; shifting it only changes a global phase.
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
    """In place: psi <- prod_k exp(-i dt B(s_k) H_problem) exp(-i dt A(s_k) (-sum X)),
    with A(s) = 1 - s, B(s) = s and H_problem[j] = levels[level_of[j]]."""
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
def _sqa_read(h, indptr, indices, data, slices, beta, gammas, seed):
    """One SQA run on the Ising model sum h_i s_i + sum_{i<j} J_ij s_i s_j.

    J is symmetric CSR (indptr, indices, data). Trotter slices are coupled
    ferromagnetically with Jp = 1/2 ln coth(beta Gamma / P), so Boltzmann
    weight is exp(-(beta/P) sum_k E(s_k) + Jp sum_k sum_i s_i^k s_i^(k+1)).
    Returns all slices; the caller keeps the best.
    """
    np.random.seed(seed)
    n = h.size
    s = np.where(np.random.random((slices, n)) < 0.5, -1.0, 1.0)
    bp = beta / slices
    for gamma in gammas:
        jp = 0.5 * np.log(1.0 / np.tanh(beta * gamma / slices))
        # Local single-spin flips in each slice.
        for k in range(slices):
            up, dn = (k + 1) % slices, (k + slices - 1) % slices
            for i in range(n):
                f = h[i]
                for p in range(indptr[i], indptr[i + 1]):
                    f += data[p] * s[k, indices[p]]
                dx = 2.0 * s[k, i] * (bp * f - jp * (s[up, i] + s[dn, i]))
                if dx >= 0.0 or np.random.random() < np.exp(dx):
                    s[k, i] = -s[k, i]
        # Global moves: flip spin i in every slice (leaves the slice coupling unchanged).
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
    return s


def anneal_sqa(bqm, num_reads=20, sweeps=2000, slices=16, beta=160.0, gamma=(0.5, 1e-3),
               stop_energy=None):
    """Simulated quantum annealing; Gamma decreases linearly over `sweeps`.

    Couplings are divided by the largest |coefficient|, so beta and gamma are in
    units of the problem energy scale. Defaults were tuned on 6x6 and 9x9
    puzzles: what matters most is the per-slice inverse temperature beta/slices
    (5-20 worked best; below 1 or at 40 it failed). If stop_energy is given, reads stop as
    soon as one reaches it (sudoku ground states have energy 0).
    """
    variables = list(bqm.variables)
    h, (row, col, j), _ = bqm.spin.to_numpy_vectors(variables)
    scale = max(np.abs(h).max(initial=0), np.abs(j).max(initial=0)) or 1.0
    rows, cols = np.concatenate([row, col]), np.concatenate([col, row])
    order = np.argsort(rows, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=len(h)))])
    indices = cols[order].astype(np.int64)
    data = np.concatenate([j, j])[order] / scale
    gammas = np.linspace(gamma[0], gamma[1], sweeps)

    samples, energies = [], []
    for _ in range(num_reads):
        spins = _sqa_read(h / scale, indptr, indices, data, slices, beta, gammas,
                          np.random.randint(2**31))
        bits = ((spins + 1) // 2).astype(np.int8)
        slice_e = bqm.energies((bits, variables))
        best = int(np.argmin(slice_e))
        samples.append(bits[best])
        energies.append(slice_e[best])
        if stop_energy is not None and slice_e[best] <= stop_energy + 1e-9:
            break
    return dimod.SampleSet.from_samples((np.array(samples), variables),
                                        dimod.BINARY, energies)


def anneal_qpu(bqm, num_reads=1000, **kwargs):
    """Submit one job to the D-Wave QPU (EmbeddingComposite finds the minor embedding)."""
    from dwave.system import DWaveSampler, EmbeddingComposite

    sampler = EmbeddingComposite(DWaveSampler())
    return sampler.sample(bqm, num_reads=num_reads, label="quantum-sudoku",
                          return_embedding=True, **kwargs)


BACKENDS = {"statevector": anneal_statevector, "sqa": anneal_sqa, "qpu": anneal_qpu}
