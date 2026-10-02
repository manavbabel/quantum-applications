import time
from itertools import pairwise

import numpy as np
from formulation import best_feasible, build_formulation, qubo_to_ising
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit.library import qaoa_ansatz
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import IBMBackend, SamplerV2
from results import SolverUnavailable, make_result
from scipy.optimize import minimize

# the QAOA solvers use the formulation without its one-hot penalty: each task's block of qubits starts
# in an equal superposition of its start times, and an XY mixer within the block keeps exactly one set
# bit order: qubit n is variable n, and every sample is turned into a list of bits in variable order


# convert a Qiskit counts key into a list of bits
# Qiskit keys are little-endian: qubit 0 is the rightmost character, so reverse it
def counts_to_bits(key):
    return [int(bit) for bit in reversed(key)]


# the cost Hamiltonian: the QUBO's Ising form as Z and ZZ terms, scaled so the largest coefficient is 1,
# so the same angles suit different instances; the constant is a global phase, so it is dropped
def cost_operator(formulation):
    h, J = qubo_to_ising(formulation.Q)
    scale = max(map(abs, [*h.values(), *J.values()]), default=0) or 1
    terms = [("Z", [a], c / scale) for a, c in h.items()]
    terms += [("ZZ", [a, b], c / scale) for (a, b), c in J.items()]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=formulation.num_variables).simplify()


# put each block into a W state (an equal superposition of its start times), using O(len(block)) gates
def w_states(formulation):
    circuit = QuantumCircuit(formulation.num_variables)
    for block in formulation.blocks:
        circuit.x(block[0])
        for i, (a, b) in enumerate(pairwise(block)):
            circuit.cry(2 * np.arccos(np.sqrt(1 / (len(block) - i))), a, b)
            circuit.cx(b, a)
    return circuit


# the XY mixer, -(XX + YY)/2 around a ring within each block
# it moves a task between start times without setting two at once, and its ground state is the W state
# a block of two gets a single pair, as a ring would couple it twice
def xy_mixer(formulation):
    terms = []
    for block in formulation.blocks:
        pairs = list(pairwise(block))
        if len(block) > 2:
            pairs.append((block[-1], block[0]))
        for a, b in pairs:
            terms += [("XX", [a, b], -0.5), ("YY", [a, b], -0.5)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=formulation.num_variables)


# the linear ramps from arXiv:2405.09169 eq. 4
# over layers i = 0 .. p-1 gamma ramps up and beta ramps down, neither reaching zero
# ordered betas then gammas, as qaoa_ansatz orders its parameters
def linear_ramp(p, delta_gamma, delta_beta):
    layers = np.arange(p)
    return np.concatenate([(1 - layers / p) * delta_beta, (layers + 1) / p * delta_gamma])


# the energy of each bitstring (bits in variable order, one row each) under an operator of Z terms
# each term counts its coefficient, negated when an odd number of the bits it acts on are set
def energies(operator, bits):
    parity = np.asarray(bits, dtype=np.int64) @ operator.paulis.z.T.astype(np.int64) % 2
    return (1 - 2 * parity) @ operator.coeffs.real


# the mean energy of the lowest alpha fraction of the shots (the CVaR), with alpha = 1 the plain mean
# weights say how often each energy came up; the shots straddling the alpha cut count in part
def cvar(energies, weights, alpha):
    order = np.argsort(energies)
    energies, weights = energies[order], np.asarray(weights, dtype=float)[order]
    weights /= weights.sum()
    # each energy's share of the alpha taken: its weight, or what is left of alpha once the lower
    # energies have taken theirs
    taken = np.clip(alpha - (np.cumsum(weights) - weights), 0, weights)
    return taken @ energies / alpha


# the formulation without its one-hot penalty, with one qubit per variable, which has to fit on the backend
def qaoa_formulation(problem, backend):
    formulation = build_formulation(problem, for_qaoa=True)
    if formulation.num_variables > backend.num_qubits:
        raise SolverUnavailable(
            f"{formulation.num_variables} qubits needed, {backend.name} has {backend.num_qubits}"
        )
    return formulation


# the QAOA circuit with p layers, its angles left free
def ansatz(formulation, p):
    circuit = qaoa_ansatz(
        cost_operator(formulation),
        reps=p,
        initial_state=w_states(formulation),
        mixer_operator=xy_mixer(formulation),
    )
    circuit.measure_all()
    return circuit


# a sampler and a transpiler for the backend
# an AerSimulator (or fake backend) runs locally, in Qiskit Runtime's local testing mode; an IBM backend
# runs on the device; seed fixes the transpiler and, on a simulator, the shots
def sampler_and_pass_manager(backend, shots, seed):
    options = {"default_shots": shots}
    if seed is not None and not isinstance(backend, IBMBackend):
        options["simulator"] = {"seed_simulator": seed}
    sampler = SamplerV2(mode=backend, options=options)
    return sampler, generate_preset_pass_manager(backend=backend, seed_transpiler=seed)


# one run of the transpiled circuit at the given angles: Qiskit counts, and the seconds the QPU spent on
# it (None on a simulator)
def sample(sampler, circuit, angles):
    result = sampler.run([(circuit, angles)]).result()
    spans = result.metadata.get("execution", {}).get("execution_spans")
    return result[0].data.meas.get_counts(), spans.duration if spans else None


# QAOA with the angles tuned by scipy's COBYLA, as in IBM's QAOA tutorial, starting from LR-QAOA's ramp
# each evaluation samples the circuit; alpha < 1 minimises the CVaR (the mean energy of the best alpha
# fraction of shots) instead of the mean
def solve_qaoa(
    problem, backend, p=1, alpha=1.0, maxiter=100, shots=4096, seed=None, delta_gamma=0.6, delta_beta=0.3
):
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


# LR-QAOA: the angles follow a fixed annealing-like schedule, so there is no optimiser and one execution
# the ramps and the default slopes are from arXiv:2405.09169
def solve_lr_qaoa(problem, backend, p=10, shots=4096, seed=None, delta_gamma=0.6, delta_beta=0.3):
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


# what one QAOA circuit with p layers needs: its depth and two-qubit gates in a generic all-to-all gate
# set, and if it fits the backend, the same on the device and (where gate durations are published, as on
# IBM hardware) the time per shot
def estimate_circuit(problem, backend, p):
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
