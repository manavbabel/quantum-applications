import time
from itertools import pairwise

import numpy as np
from formulation import best_feasible, build_formulation, qubo_to_ising
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit.library import qaoa_ansatz
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import IBMBackend, SamplerV2
from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.optimizers import COBYLA
from solver import Solver, SolverUnavailable, fraction

# the QAOA solvers below work on the formulation without its one-hot penalty
# each task's block of variables starts in an equal superposition of its start times, and an
# XY mixer within each block keeps exactly one of them set, so the constraint holds by construction
# bit order: qubit n stands for variable n; Qiskit reads its registers little-endian, so qubit n is
# bit n of an integer outcome and the rightmost character of a counts key (counts_to_bits reverses those)
# every sample below is turned into a list of bits in variable order

# convert a Qiskit counts key into a list of bits
# Qiskit keys are little-endian: qubit 0 is the rightmost character, so reverse it
def counts_to_bits(key):
    return [int(bit) for bit in reversed(key)]

# the cost Hamiltonian: the QUBO's Ising form, as Z and ZZ terms
# scaled so its largest coefficient is 1, which keeps the QAOA angles meaning the same thing from one
# instance to the next; the constant only adds a global phase, so it is dropped
# built directly, as qiskit-optimization's QuadraticProgram.to_ising takes minutes on ft06
def cost_operator(formulation):
    h, J = qubo_to_ising(formulation.Q)
    scale = max(map(abs, [*h.values(), *J.values()]), default=0) or 1
    terms = [("Z", [a], c / scale) for a, c in h.items()]
    terms += [("ZZ", [a, b], c / scale) for (a, b), c in J.items()]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=formulation.num_variables).simplify()

# put each block into an equal superposition of its start times, the W state
# built as a cascade, so it costs O(len(block)) gates rather than a state preparation's O(2^len(block))
def w_states(formulation):
    circuit = QuantumCircuit(formulation.num_variables)
    for block in formulation.blocks:
        circuit.x(block[0])
        for i, (a, b) in enumerate(pairwise(block)):
            circuit.cry(2 * np.arccos(np.sqrt(1 / (len(block) - i))), a, b)
            circuit.cx(b, a)
    return circuit

# the XY mixer, -(XX + YY)/2 around a ring within each block
# it moves a task between start times without ever setting two at once, and its ground state is the
# W state above, which is where an annealing-like ramp has to start
# one start time needs no mixer, and two share a single pair, which a ring would couple twice
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


# shared set-up for the gate-based solvers
# backend is an AerSimulator (or fake backend) for local runs, or an IBM backend for a real device
# both go through Qiskit Runtime's V2 sampler, the local ones in its local testing mode
# seed fixes the transpiler and, on a simulator, the shots; the hardware itself is not reproducible
class _QAOASolver(Solver):

    def __init__(self, backend, shots=4096, seed=None):
        self.backend = backend
        self.shots = shots
        self.name = f"{self.name}/{backend.name}"

        options = {"default_shots": shots}
        if seed is not None and not isinstance(backend, IBMBackend):
            options["simulator"] = {"seed_simulator": seed}
        self.sampler = SamplerV2(mode=backend, options=options)
        self.pass_manager = generate_preset_pass_manager(backend=backend, seed_transpiler=seed)

    # one qubit per variable, which has to fit on the backend
    def _formulation(self, problem):
        formulation = build_formulation(problem, for_qaoa=True)
        if formulation.num_variables > self.backend.num_qubits:
            raise SolverUnavailable(
                f"{formulation.num_variables} qubits needed, {self.backend.name} has {self.backend.num_qubits}"
            )
        return formulation

    # the logical circuit, with its angles left free
    # built as qiskit-optimization's QAOA builds its own, so both solvers run the same ansatz
    def _ansatz(self, formulation, p):
        circuit = qaoa_ansatz(
            cost_operator(formulation),
            reps=p,
            initial_state=w_states(formulation),
            mixer_operator=xy_mixer(formulation),
        )
        circuit.measure_all()
        return circuit

    def _metadata(self, formulation, p, samples):
        return {
            "calls_unit": "circuit executions",
            "backend": self.backend.name,
            "qubits": formulation.num_variables,
            "p": p,
            "shots": self.shots,
            "feasible_fraction": 1 - fraction(samples, None),
        }

    # what a solve would need; nothing here has to fit on the backend, except the device figures
    # the logical depth is counted in a generic gate set, with every qubit free to talk to every other
    # on the backend itself, routing adds depth, and a device that publishes its gate durations
    # (IBM hardware, a fake backend, but not a simulator) also prices one shot
    def _estimate(self, problem, p, iterations, executions):
        formulation = build_formulation(problem, for_qaoa=True)
        circuit = self._ansatz(formulation, p)
        logical = generate_preset_pass_manager(optimization_level=1, basis_gates=["cx", "rz", "sx", "x"]).run(circuit)

        estimate = {
            "qubits": formulation.num_variables,
            "fits_backend": formulation.num_variables <= self.backend.num_qubits,
            "optimiser_iterations": iterations,
            "circuit_executions": executions,
            "shots_per_execution": self.shots,
            "total_shots": executions * self.shots,
            "logical_depth": logical.depth(),
            "logical_two_qubit_gates": logical.num_nonlocal_gates(),
        }
        if estimate["fits_backend"]:
            device = self.pass_manager.run(circuit)
            estimate["device_depth"] = device.depth()
            estimate["device_two_qubit_gates"] = device.num_nonlocal_gates()
            try:
                estimate["shot_duration"] = device.estimate_duration(self.backend.target)  # seconds
            except QiskitError:  # no gate durations to go on
                estimate["shot_duration"] = None
        return estimate

# QAOA, with the angles found by a classical optimiser through qiskit-optimization
# they start on the linear ramp LR-QAOA uses below, which the optimiser then refines
# alpha < 1 minimises the CVaR (the mean energy of the best alpha fraction of shots) instead of the mean
class QAOASolver(_QAOASolver):
    name = "qaoa"

    @staticmethod
    def _optimizer(optimizer):
        return optimizer or COBYLA(maxiter=100)

    def _solve(self, problem, p=1, alpha=1.0, optimizer=None, delta_gamma=0.6, delta_beta=0.3):
        formulation = self._formulation(problem)
        evaluated = []  # when each objective evaluation finished
        qaoa = QAOA(
            self.sampler,
            self._optimizer(optimizer),
            reps=p,
            initial_state=w_states(formulation),
            mixer=xy_mixer(formulation),
            initial_point=linear_ramp(p, delta_gamma, delta_beta),
            aggregation=alpha,
            callback=lambda *_: evaluated.append(time.perf_counter()),
            pass_manager=self.pass_manager,
        )

        cost = cost_operator(formulation)
        t0 = time.perf_counter()
        eigen = qaoa.compute_minimum_eigenvalue(cost)
        t1 = time.perf_counter()

        # the samples are the distribution at the optimised angles, each shot a run of its own
        # the distribution is keyed by integer outcome, with qubit n as bit n
        distribution = eigen.eigenstate
        start_times, samples = best_feasible(
            [[(state >> n) & 1 for n in range(formulation.num_variables)] for state in distribution],
            list(distribution.values()),
            formulation,
            problem,
        )

        evaluations = int(eigen.cost_function_evals)
        metadata = self._metadata(formulation, p, samples)
        metadata.update({
            "alpha": alpha,
            "optimizer_evaluations": evaluations,
            "optimisation_time": evaluated[-1] - t0,
            "optimal_point": eigen.optimal_point.tolist(),
        })
        # one circuit execution per objective evaluation, and one more for the final distribution
        return start_times, t1 - t0, evaluations + 1, metadata, samples

    # the optimiser stops by maxiter evaluations at the latest, one circuit execution each (as for
    # COBYLA; SPSA takes two), then samples the final distribution once more
    # scipy's COBYLA raises maxiter to at least one more than its initial simplex, i.e. the 2p angles + 2
    def estimate(self, problem, p=1, alpha=1.0, optimizer=None, delta_gamma=0.6, delta_beta=0.3):
        iterations = max(self._optimizer(optimizer).settings["maxiter"], 2 * p + 2)
        return self._estimate(problem, p, iterations, iterations + 1)

# LR-QAOA: the angles follow a fixed annealing-like schedule, so there is no optimiser and one execution
# the ramps and the default slopes are from arXiv:2405.09169
class LRQAOASolver(_QAOASolver):
    name = "lr-qaoa"

    def _solve(self, problem, p=10, delta_gamma=0.6, delta_beta=0.3):
        formulation = self._formulation(problem)
        circuit = self.pass_manager.run(self._ansatz(formulation, p))

        t0 = time.perf_counter()
        job = self.sampler.run([(circuit, linear_ramp(p, delta_gamma, delta_beta))])
        counts = job.result()[0].data.meas.get_counts()
        t1 = time.perf_counter()

        # each shot is a run of its own
        start_times, samples = best_feasible(
            [counts_to_bits(key) for key in counts],
            list(counts.values()),
            formulation,
            problem,
        )

        metadata = self._metadata(formulation, p, samples)
        metadata.update({
            "delta_gamma": delta_gamma,
            "delta_beta": delta_beta,
            "depth": circuit.depth(),
            "two_qubit_gates": circuit.num_nonlocal_gates(),
        })
        return start_times, t1 - t0, 1, metadata, samples

    # one pass through the ramp, so one execution
    def estimate(self, problem, p=10, delta_gamma=0.6, delta_beta=0.3):
        return self._estimate(problem, p, 1, 1)
