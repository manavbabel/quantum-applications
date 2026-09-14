# annealing solvers: draw many low-energy samples of the QUBO and keep the best feasible one
# simulated annealing runs locally, quantum annealing on a D-Wave QPU
# both take the same Formulation.Q, so comparing them isolates the sampler

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite

from base import Solver, SolverUnavailable, best_feasible


# shared by both: sample the QUBO, then decode and validate the samples
# subclasses implement _sample(bqm), returning (sampleset, metadata)
class _AnnealingSolver(Solver):

    def __init__(self, num_reads=1000, seed=None):
        self.num_reads = num_reads
        self.seed = seed

    def _solve(self, problem):
        formulation = self.formulation(problem)
        bqm = dimod.BinaryQuadraticModel.from_qubo(formulation.Q, offset=formulation.offset)
        sampleset, metadata = self._sample(bqm)

        # samples are keyed by variable label, which is our variable index, so read bit n by
        # label rather than trusting the sampleset's column order
        # identical reads may be merged into one row, so there can be fewer rows than reads
        samples = ([sample[n] for n in range(formulation.num_variables)]
                   for sample in sampleset.samples())
        # every sample is decoded and checked against the problem itself, so whatever noise
        # or a broken chain did to it, only a valid schedule comes back
        best = best_feasible(samples, formulation, problem)

        metadata.update({"calls_unit": "reads", "num_reads": self.num_reads})
        return (best[1] if best else None), self.num_reads, metadata


# classical simulated annealing
# the baseline for the QPU on the identical QUBO, and a free way to exercise this code
class SimulatedAnnealingSolver(_AnnealingSolver):
    name = "simulated-annealing"

    def _sample(self, bqm):
        sampler = SimulatedAnnealingSampler()
        return sampler.sample(bqm, num_reads=self.num_reads, seed=self.seed), {}


# quantum annealing on Advantage2, D-Wave's newest QPU, which Leap hosts in na-east-1
# the token comes from `dwave config create` or DWAVE_API_TOKEN
#
# embedding: Advantage2's qubits form a Zephyr graph, each coupled to at most 20 others, so a
# variable may become a chain of qubits held together by a strong ferromagnetic coupling
# minorminer finds the chains for each solve, the same ones every time when seeded
# the chain strength is Ocean's default, uniform torque compensation on the spin couplings
# the QPU scales the whole problem into its h and J ranges, so an over-strong chain would
# squeeze the makespan objective, one unit against a penalty of horizon + 1, into the noise
#
# schedule: a standard forward anneal lasting annealing_time microseconds; 20 is D-Wave's default
# each read also spends about 60 us on readout and delay, so a longer anneal buys fewer reads
# pauses and reverse anneals need tuning per device, so they are left out
class QuantumAnnealingSolver(_AnnealingSolver):

    def __init__(self, annealing_time=20, region="na-east-1", **options):
        super().__init__(**options)
        self.annealing_time = annealing_time
        try:
            qpu = DWaveSampler(region=region, solver={"topology__type": "zephyr"})
        except Exception as exc:  # no token, no Zephyr QPU online, network down
            raise SolverUnavailable(f"cannot reach a Zephyr QPU in {region}: {exc}") from exc
        self.sampler = EmbeddingComposite(qpu, embedding_parameters={"random_seed": self.seed})
        self.name = f"quantum-annealing/{qpu.properties['chip_id']}"

    def _sample(self, bqm):
        try:
            sampleset = self.sampler.sample(bqm, num_reads=self.num_reads,
                                            annealing_time=self.annealing_time,
                                            return_embedding=True)
        except ValueError as exc:  # minorminer found no embedding
            raise SolverUnavailable(f"cannot embed on {self.name}: {exc}") from exc

        info = sampleset.info
        chains = [len(chain) for chain in info["embedding_context"]["embedding"].values()]
        return sampleset, {
            "annealing_time": self.annealing_time,
            "problem_id": info["problem_id"],
            "qpu_access_time": info["timing"]["qpu_access_time"],  # microseconds
            "physical_qubits": sum(chains),
            "max_chain_length": max(chains),
            "chain_strength": info["embedding_context"]["chain_strength"],
            # the fraction of chains broken per read; a broken chain is unembedded by majority
            # vote, and a tie, as in any broken chain of two, rounds to 1
            "chain_break_fraction": float(np.average(sampleset.record.chain_break_fraction,
                                                     weights=sampleset.record.num_occurrences)),
        }
