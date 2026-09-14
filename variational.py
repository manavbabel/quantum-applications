# here we implement variational algorithms to solve the QUBO
# the backends are in another file

# our one-hot constraint is handled by construction
# each task's block of variables starts in an equal superposition of its start times
# and an XY mixer ring within that block keeps exactly one of them set

# for VQE we use a HEA with no subspace structure


from itertools import combinations

import numpy as np

from base import Solver, best_feasible
from formulation import counts_to_bits, energy, qubo_to_ising


# get the ring of neighbouring qubits within one block
# one qubit has a single start time and needs no mixer
# two qubits share a single pair, which the ring below would otherwise apply twice
def _mixer_pairs(block):
    if len(block) < 2:
        return []
    if len(block) == 2:
        return [(block[0], block[1])]
    return list(zip(block, block[1:] + block[:1]))

# undo the one-hot penalty, which build_formulation added as penalty * (sum_k x_ik - 1)^2
# the XY mixer keeps every block one-hot, so over the states QAOA can reach those terms are
# a constant: they only ever added a global phase, and they dominated the scale below
def _drop_one_hot(formulation):
    Q = dict(formulation.Q)
    for block in formulation.blocks:
        for n in block:
            Q[(n, n)] += formulation.penalty
        for pair in combinations(block, 2):
            Q[pair] -= 2 * formulation.penalty
    return {pair: coefficient for pair, coefficient in Q.items() if coefficient}

# put one block into an equal superposition of its start times
# this is the W state, the ground state of the mixer below, which a ramp has to start from
# built as a cascade, so it costs O(len(block)) gates rather than a state prep's O(2^len(block))
def _prepare_w(circuit, block):
    circuit.x(block[0])
    for i, (a, b) in enumerate(zip(block, block[1:])):
        circuit.cry(2 * np.arccos(np.sqrt(1 / (len(block) - i))), a, b)
        circuit.cx(b, a)

# the Ising fields and couplings, plus the largest coefficient
# dividing the angles by it keeps them meaning the same thing from one instance to the next
def _hamiltonian(Q):
    h, J, _ = qubo_to_ising(Q)
    magnitudes = [abs(v) for v in h.values()] + [abs(v) for v in J.values()]
    return h, J, max(magnitudes, default=1.0)

# one circuit for the whole run, with free angles to bind per iteration
# the 2 * reps parameters are the gammas followed by the betas
# they share one ParameterVector, whose elements sort by index, so a flat array of angles
# binds in that same order
def build_qaoa_circuit(formulation, reps):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    h, J, scale = _hamiltonian(_drop_one_hot(formulation))
    circuit = QuantumCircuit(formulation.num_variables)
    angles = ParameterVector("angle", 2 * reps)

    for block in formulation.blocks:  # every start time for a task, equally weighted,
        _prepare_w(circuit, block)    # which puts us inside the one-hot subspace

    for gamma, beta in zip(angles[:reps], angles[reps:]):
        for qubit, field in h.items():  # exp(-i gamma H_cost)
            circuit.rz(2 * gamma * field / scale, qubit)
        for (a, b), coupling in J.items():
            circuit.rzz(2 * gamma * coupling / scale, a, b)

        # one Trotter step of exp(-i beta H_mixer) per block, for H_mixer = -sum(XX + YY)
        # the minus sign makes the W state above the mixer's ground state
        # the ring's pairs overlap so the step is approximate, but each conserves block weight
        for block in formulation.blocks:
            for a, b in _mixer_pairs(block):
                circuit.rxx(-2 * beta, a, b)
                circuit.ryy(-2 * beta, a, b)

    circuit.measure_all()
    return circuit

# as above, one circuit with free angles
# its num_qubits * (reps + 1) parameters run layer by layer, qubit by qubit
def build_vqe_circuit(num_qubits, reps):
    """RY layers separated by a CZ ring: no subspace structure, so penalties do all the work."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    circuit = QuantumCircuit(num_qubits)
    parameters = ParameterVector("theta", num_qubits * (reps + 1))
    index = 0
    for layer in range(reps + 1):
        for qubit in range(num_qubits):
            circuit.ry(parameters[index], qubit)
            index += 1
        if layer < reps:
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
            if num_qubits > 2:
                circuit.cz(num_qubits - 1, 0)
    circuit.measure_all()
    return circuit

# extract some energy metric from the samples
# this can be the mean energy
# or the CVaR of the best alpha fraction
def aggregate_energy(counts, formulation, alpha=1.0):

    samples = sorted(
        (energy(counts_to_bits(key), formulation), count)
        for key, count in counts.items()
    )
    total = sum(count for _, count in samples)
    target = max(1, round(alpha * total))
    accumulated = taken = 0
    for value, count in samples:
        take = min(count, target - taken)
        accumulated += value * take
        taken += take
        if taken >= target:
            break
    return accumulated / taken

# base class for variational solvers
# because they all share: running circuit, getting count energy, best feasible sample
class _SampledVariationalSolver(Solver):
    base_name = "variational"

    def __init__(
        self, backend, shots=4096, alpha=1.0, seed=None
    ):
        self.backend = backend
        self.shots = shots
        self.alpha = alpha
        self.seed = seed

    @property
    def name(self):
        return f"{self.base_name}/{self.backend.name}"

    # tracker used for the solver process
    def _track(self, counts, formulation, problem, state):
        state["calls"] += 1
        state["best"] = best_feasible(
            (counts_to_bits(key) for key in counts),
            formulation,
            problem,
            state["best"],
        )

    @staticmethod
    def _finish(state, metadata):
        best = state["best"]
        return (best[1] if best else None), state["calls"], metadata

# XY-mixer QAOA with a classical optimiser
class QAOASolver(_SampledVariationalSolver):
    base_name = "qaoa"

    def __init__(self, reps, maxiter=150, optimizer="COBYLA", **options):
        super().__init__(**options)
        self.reps = reps
        self.maxiter = maxiter
        self.optimizer = optimizer

    def _solve(self, problem):
        from scipy.optimize import minimize

        formulation = self.formulation(problem)
        circuit = self.backend.prepare(build_qaoa_circuit(formulation, self.reps))
        state = {"calls": 0, "best": None}

        def objective(angles):
            counts = self.backend.sample(circuit, self.shots, angles)
            self._track(counts, formulation, problem, state)
            return aggregate_energy(counts, formulation, self.alpha)

        x0 = np.concatenate(
            [
                np.linspace(0.1, 0.9, self.reps),  # gammas ramp up
                np.linspace(0.9, 0.1, self.reps),  # betas ramp down
            ]
        )

        result = minimize(
            objective, x0, method=self.optimizer, options={"maxiter": self.maxiter}
        )

        metadata = {
            "calls_unit": "circuit executions",
            "reps": self.reps,
            "shots": self.shots,
            "alpha": self.alpha,
            "backend": self.backend.name,
            "final_objective": float(result.fun),
        }
        return self._finish(state, metadata)

# LR-QAOA: annealing-like schedule, no classical optimiser
# the ramps and the default slopes are from arXiv:2405.09169 eq. 4
class LRQAOASolver(_SampledVariationalSolver):
    base_name = "lr-qaoa"

    def __init__(self, p=20, delta_gamma=0.6, delta_beta=0.3, **options):
        super().__init__(**options)
        self.p = p
        self.delta_gamma = delta_gamma
        self.delta_beta = delta_beta

    def _solve(self, problem):
        formulation = self.formulation(problem)
        state = {"calls": 0, "best": None}

        # over layers i = 0 .. p-1 gamma ramps up and beta ramps down, neither reaching zero
        layers = np.arange(self.p)
        angles = np.concatenate([
            ((layers + 1) / self.p) * self.delta_gamma,
            (1 - layers / self.p) * self.delta_beta,
        ])
        circuit = self.backend.prepare(build_qaoa_circuit(formulation, self.p))
        self._track(self.backend.sample(circuit, self.shots, angles),
                    formulation, problem, state)

        metadata = {
            "calls_unit": "circuit executions",
            "p": self.p,
            "shots": self.shots,
            "delta_gamma": self.delta_gamma,
            "delta_beta": self.delta_beta,
            "backend": self.backend.name,
            "depth": circuit.depth(),
            "parameter_free": True,
        }
        return self._finish(state, metadata)

# HEA ansatz with a classical optimiser
# no one-hot structure, we rely on penalties
class VQESolver(_SampledVariationalSolver):
    base_name = "vqe"

    def __init__(self, p=3, maxiter=300, optimizer="COBYLA", alpha=0.25, **options):
        options.setdefault("alpha", alpha)
        super().__init__(**options)
        self.p = p
        self.maxiter = maxiter
        self.optimizer = optimizer

    def _solve(self, problem):
        from scipy.optimize import minimize

        formulation = self.formulation(problem)
        circuit = build_vqe_circuit(formulation.num_variables, self.p)
        num_parameters = circuit.num_parameters
        circuit = self.backend.prepare(circuit)
        state = {"calls": 0, "best": None}

        def objective(parameters):
            counts = self.backend.sample(circuit, self.shots, parameters)
            self._track(counts, formulation, problem, state)
            return aggregate_energy(counts, formulation, self.alpha)

        rng = np.random.default_rng(self.seed)
        x0 = rng.uniform(0, 2 * np.pi, num_parameters)
        # COBYLA's initial simplex is num_parameters + 1 points, so it needs that many
        # evaluations before it can make a move
        result = minimize(
            objective,
            x0,
            method=self.optimizer,
            options={"maxiter": max(self.maxiter, num_parameters + 1)},
        )

        metadata = {
            "calls_unit": "circuit executions",
            "p": self.p,
            "shots": self.shots,
            "alpha": self.alpha,
            "backend": self.backend.name,
            "parameters": num_parameters,
            "final_objective": float(result.fun),
        }
        return self._finish(state, metadata)
