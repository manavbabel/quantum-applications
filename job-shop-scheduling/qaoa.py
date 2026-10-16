"""QAOA without the one-hot penalty: each task's qubits start in a W state, and an XY mixer keeps one set.

Qubit n is variable n, and every sample is turned into a list of bits in variable order.
"""

import time
from itertools import pairwise

import numpy as np
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit.library import qaoa_ansatz
from qiskit.exceptions import QiskitError
from qiskit.primitives import BackendSamplerV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import IBMBackend, SamplerV2
from scipy.optimize import minimize

from formulation import best_feasible, build_formulation, qubo_to_ising
from results import SolverUnavailable, make_result


def counts_to_bits(key):
    """A Qiskit counts key as bits in variable order; Qiskit puts qubit 0 rightmost."""
    return [int(bit) for bit in reversed(key)]


def cost_operator(formulation):
    """The QUBO's Ising form as Z and ZZ terms, scaled to a largest coefficient of 1 so angles transfer.

    The constant is a global phase, so it is dropped.
    """
    h, J = qubo_to_ising(formulation.Q)
    scale = max(map(abs, [*h.values(), *J.values()]), default=0) or 1
    terms = [("Z", [a], c / scale) for a, c in h.items()]
    terms += [("ZZ", [a, b], c / scale) for (a, b), c in J.items()]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=formulation.num_variables).simplify()


def w_states(formulation):
    """Put each block into a W state, an equal superposition of its start times, in O(len(block)) gates."""
    circuit = QuantumCircuit(formulation.num_variables)
    for block in formulation.blocks:
        circuit.x(block[0])
        for i, (a, b) in enumerate(pairwise(block)):
            circuit.cry(2 * np.arccos(np.sqrt(1 / (len(block) - i))), a, b)
            circuit.cx(b, a)
    return circuit


def xy_mixer(formulation):
    """The XY mixer, -(XX + YY)/2 around a ring in each block, which moves a task between start times.

    It never sets two start times at once, and its ground state is the W state.
    """
    terms = []
    for block in formulation.blocks:
        pairs = list(pairwise(block))
        # a block of two gets a single pair, as a ring would couple it twice
        if len(block) > 2:
            pairs.append((block[-1], block[0]))
        for a, b in pairs:
            terms += [("XX", [a, b], -0.5), ("YY", [a, b], -0.5)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=formulation.num_variables)


def linear_ramp(p, delta_gamma, delta_beta):
    """arXiv:2405.09169 eq. 4: gamma ramps up and beta down over p layers, as betas then gammas."""
    layers = np.arange(p)
    return np.concatenate([(1 - layers / p) * delta_beta, (layers + 1) / p * delta_gamma])


def energies(operator, bits):
    """The energy of each row of bits under an operator of Z terms."""
    # each term counts its coefficient, negated when an odd number of the bits it acts on are set
    parity = np.asarray(bits, dtype=np.int64) @ operator.paulis.z.T.astype(np.int64) % 2
    return (1 - 2 * parity) @ operator.coeffs.real


def cvar(energies, weights, alpha):
    """The CVaR: the mean energy of the lowest alpha fraction of the weighted shots."""
    order = np.argsort(energies)
    energies, weights = energies[order], np.asarray(weights, dtype=float)[order]
    weights /= weights.sum()
    # each energy takes its weight, or whatever is left of alpha once the lower energies have had theirs
    taken = np.clip(alpha - (np.cumsum(weights) - weights), 0, weights)
    return taken @ energies / alpha


def qaoa_formulation(problem, backend):
    """The formulation without its one-hot penalty, if its qubits fit on the backend."""
    formulation = build_formulation(problem, for_qaoa=True)
    if formulation.num_variables > backend.num_qubits:
        raise SolverUnavailable(
            f"{formulation.num_variables} qubits needed, {backend.name} has {backend.num_qubits}"
        )
    return formulation


def ansatz(formulation, p):
    """The QAOA circuit with p layers, its angles left free."""
    circuit = qaoa_ansatz(
        cost_operator(formulation),
        reps=p,
        initial_state=w_states(formulation),
        mixer_operator=xy_mixer(formulation),
    )
    circuit.measure_all()
    return circuit


def sampler_and_pass_manager(backend, shots, seed):
    """A sampler and transpiler for the backend; seed fixes the transpiler and any simulated shots."""
    if isinstance(backend, IBMBackend):
        sampler = SamplerV2(mode=backend, options={"default_shots": shots})
    else:
        # what Runtime's local testing mode uses underneath, called directly as that mode ignores a seed of 0
        sampler = BackendSamplerV2(backend=backend, options={"default_shots": shots, "seed_simulator": seed})
    return sampler, generate_preset_pass_manager(backend=backend, seed_transpiler=seed)


def sample(sampler, circuit, angles):
    """Run the transpiled circuit once: Qiskit counts, and the QPU seconds (None on a simulator)."""
    result = sampler.run([(circuit, angles)]).result()
    spans = result.metadata.get("execution", {}).get("execution_spans")
    return result[0].data.meas.get_counts(), spans.duration if spans else None


def solve_qaoa(
    problem, backend, p=1, alpha=1.0, maxiter=100, shots=4096, seed=None, delta_gamma=0.6, delta_beta=0.3
):
    """QAOA with the angles tuned by COBYLA from LR-QAOA's ramp, as in IBM's QAOA tutorial.

    Each evaluation samples the circuit; alpha < 1 minimises the CVaR instead of the mean energy.
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    formulation = qaoa_formulation(problem, backend)
    cost = cost_operator(formulation)
    sampler, pass_manager = sampler_and_pass_manager(backend, shots, seed)
    circuit = pass_manager.run(ansatz(formulation, p))
    qpu_times = []

    def objective(angles):
        counts, qpu_time = sample(sampler, circuit, angles)
        qpu_times.append(qpu_time)
        bits = [counts_to_bits(key) for key in counts]
        return cvar(energies(cost, bits), list(counts.values()), alpha)

    t0 = time.perf_counter()
    optimum = minimize(
        objective,
        linear_ramp(p, delta_gamma, delta_beta),
        method="COBYLA",
        options={"maxiter": maxiter},
    )
    t1 = time.perf_counter()
    # the samples come from one more run at the optimised angles
    counts, qpu_time = sample(sampler, circuit, optimum.x)
    qpu_times.append(qpu_time)
    t2 = time.perf_counter()

    start_times, samples = best_feasible(
        [counts_to_bits(key) for key in counts], list(counts.values()), formulation, problem
    )
    metadata = {
        "calls_unit": "circuit executions",
        "backend": backend.name,
        "qubits": formulation.num_variables,
        "p": p,
        "shots": shots,
        "alpha": alpha,
        "optimizer_evaluations": int(optimum.nfev),
        "optimisation_time": t1 - t0,
        "optimal_point": optimum.x.tolist(),
        "qpu_access_time": None if None in qpu_times else sum(qpu_times),  # seconds
    }
    # one circuit execution per objective evaluation, and one more for the final distribution
    return make_result(
        problem, f"qaoa/{backend.name}", start_times, t2 - t0, int(optimum.nfev) + 1, metadata, samples
    )


def solve_lr_qaoa(problem, backend, p=10, shots=4096, seed=None, delta_gamma=0.6, delta_beta=0.3):
    """LR-QAOA: fixed linear ramps of the angles (arXiv:2405.09169), so no optimiser and one execution."""
    formulation = qaoa_formulation(problem, backend)
    sampler, pass_manager = sampler_and_pass_manager(backend, shots, seed)
    circuit = pass_manager.run(ansatz(formulation, p))

    t0 = time.perf_counter()
    counts, qpu_time = sample(sampler, circuit, linear_ramp(p, delta_gamma, delta_beta))
    t1 = time.perf_counter()

    start_times, samples = best_feasible(
        [counts_to_bits(key) for key in counts], list(counts.values()), formulation, problem
    )
    metadata = {
        "calls_unit": "circuit executions",
        "backend": backend.name,
        "qubits": formulation.num_variables,
        "p": p,
        "shots": shots,
        "delta_gamma": delta_gamma,
        "delta_beta": delta_beta,
        "depth": circuit.depth(),
        "two_qubit_gates": circuit.num_nonlocal_gates(),
        "qpu_access_time": qpu_time,  # seconds
    }
    return make_result(problem, f"lr-qaoa/{backend.name}", start_times, t1 - t0, 1, metadata, samples)


def estimate_circuit(problem, backend, p):
    """Depth and two-qubit gates of a p-layer circuit, all-to-all and on the backend, and time per shot."""
    formulation = build_formulation(problem, for_qaoa=True)
    circuit = ansatz(formulation, p)
    logical = generate_preset_pass_manager(optimization_level=1, basis_gates=["cx", "rz", "sx", "x"]).run(
        circuit
    )

    estimate = {
        "qubits": formulation.num_variables,
        "fits_backend": formulation.num_variables <= backend.num_qubits,
        "logical_depth": logical.depth(),
        "logical_two_qubit_gates": logical.num_nonlocal_gates(),
    }
    if estimate["fits_backend"]:
        device = generate_preset_pass_manager(backend=backend).run(circuit)
        estimate["device_depth"] = device.depth()
        estimate["device_two_qubit_gates"] = device.num_nonlocal_gates()
        try:
            estimate["shot_duration"] = device.estimate_duration(backend.target)  # seconds
        except QiskitError:  # no gate durations to go on
            estimate["shot_duration"] = None
    return estimate
