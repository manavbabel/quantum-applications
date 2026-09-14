# where circuits are run
# cutensornet only on a system with an Nvidia GPU

from base import SolverUnavailable


# base class
class Backend:
    name = "backend"

    # get a circuit ready to run, once, so an optimiser can reuse it across every iteration
    def prepare(self, circuit):
        return circuit

    def sample(self, circuit, shots, parameter_values=None):
        raise NotImplementedError


# wraps any V2 sampler primitive, transpiling first if the target needs it
class SamplerBackend(Backend):
    def __init__(self, name, sampler, pass_manager=None):
        self.name = name
        self.sampler = sampler
        self.pass_manager = pass_manager

    def prepare(self, circuit):
        return circuit if self.pass_manager is None else self.pass_manager.run(circuit)

    def sample(self, circuit, shots, parameter_values=None):
        # a free circuit goes in a pub with its angles; values bind in circuit.parameters order
        pub = circuit if parameter_values is None else (circuit, parameter_values)
        result = self.sampler.run([pub], shots=shots).result()
        # measure_all gives the circuit a single classical register, so there is one entry
        return next(iter(result[0].data.values())).get_counts()


# local simulation; any extra kwargs (a noise_model, say) go to the AerSimulator underneath
def aer_backend(seed=None, **kwargs):
    from qiskit_aer.primitives import SamplerV2
    return SamplerBackend("aer", SamplerV2(seed=seed, options={"backend_options": kwargs}))


# a real device: the one named, or whichever is least busy
# seed only fixes the transpiler's stochastic passes; the hardware itself is not reproducible
def ibm_backend(backend_name=None, seed=None):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    try:
        service = QiskitRuntimeService()
        device = (service.backend(backend_name) if backend_name
                  else service.least_busy(operational=True, simulator=False))
    except Exception as exc:
        raise SolverUnavailable(f"cannot reach IBM Quantum: {exc}") from exc

    pass_manager = generate_preset_pass_manager(
        optimization_level=3, backend=device, seed_transpiler=seed
    )
    return SamplerBackend(f"ibm/{device.name}", SamplerV2(mode=device), pass_manager)


# class CuTensorNetBackend(Backend):
#     """cuQuantum's cuTensorNet: contract the circuit's tensor network instead of evolving a
#     statevector, then sample the resulting amplitudes.

#     UNVERIFIED. cuTensorNet requires an NVIDIA GPU and CUDA, neither of which is present on
#     the machine this was written on, so this path has never been executed. Install with
#     `uv sync --extra gpu` on a GPU box; treat the first run as debugging, not as a benchmark.
#     """

#     name = "cutensornet"

#     def __init__(self, seed=None):
#         self.seed = seed

#     @staticmethod
#     def _import_cuquantum():
#         # the namespace moved to cuquantum.tensornet in cuQuantum Python 25.03
#         try:
#             from cuquantum.tensornet import CircuitToEinsum, contract
#             return CircuitToEinsum, contract
#         except ImportError:
#             pass
#         try:
#             from cuquantum import CircuitToEinsum, contract
#             return CircuitToEinsum, contract
#         except ImportError as exc:
#             raise SolverUnavailable(
#                 f"cuquantum-python is not installed or has no GPU to use: {exc}") from exc

#     def sample(self, circuit, shots, parameter_values=None):
#         import numpy as np
#
#         if parameter_values is not None:
#             circuit = circuit.assign_parameters(parameter_values)

#         CircuitToEinsum, contract = self._import_cuquantum()

#         # CircuitToEinsum cannot handle mid-circuit measurement, so contract the unitary part
#         unmeasured = circuit.remove_final_measurements(inplace=False)
#         try:
#             converter = CircuitToEinsum(unmeasured, backend="cupy")
#             expression, operands = converter.state_vector()
#             statevector = contract(expression, *operands)
#         except Exception as exc:
#             raise SolverUnavailable(f"cuTensorNet contraction failed: {exc}") from exc

#         amplitudes = np.asarray(getattr(statevector, "get", lambda: statevector)())
#         probabilities = np.abs(amplitudes.reshape(-1)) ** 2
#         probabilities = probabilities / probabilities.sum()

#         rng = np.random.default_rng(self.seed)
#         drawn = rng.choice(probabilities.size, size=shots, p=probabilities)

#         # state_vector() indexes qubit 0 as the most significant axis; Qiskit counts keys are
#         # little-endian, so reverse each bitstring on the way out.
#         counts = {}
#         for value, occurrences in zip(*np.unique(drawn, return_counts=True)):
#             key = format(int(value), f"0{unmeasured.num_qubits}b")[::-1]
#             counts[key] = counts.get(key, 0) + int(occurrences)
#         return counts
