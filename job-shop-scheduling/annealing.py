# annealing solvers
# locally (simulated) and on DWave QPUs

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite

from base import Solver, SolverUnavailable, best_feasible


# base solver: sample QUBO, decode and validate samples
# subclasses implement _sample(bqm), and return (sampleset, metadata)
class _AnnealingSolver(Solver):
    def __init__(self, num_reads=1000, seed=None):
        self.num_reads = num_reads
        self.seed = seed

    def _solve(self, problem):
        formulation = self.formulation(problem)
        bqm = dimod.BinaryQuadraticModel.from_qubo(
            formulation.Q, offset=formulation.offset
        )
        sampleset, metadata = self._sample(bqm)

        # sampls are keyed by variable label, i.e. our index, so we read
        # them in by label directly instead of column order
        samples = (
            [sample[n] for n in range(formulation.num_variables)]
            for sample in sampleset.samples()
        )
        # decode and check so only valid schedules are returned
        best = best_feasible(samples, formulation, problem)

        metadata.update({"calls_unit": "reads", "num_reads": self.num_reads})
        return (best[1] if best else None), self.num_reads, metadata


# classical simulated annealing
class SimulatedAnnealingSolver(_AnnealingSolver):
    name = "simulated-annealing"

    def _sample(self, bqm):
        sampler = SimulatedAnnealingSampler()
        return sampler.sample(bqm, num_reads=self.num_reads, seed=self.seed), {}


# real QPU - need to set up DWave access separately
class QuantumAnnealingSolver(_AnnealingSolver):
    def __init__(self, annealing_time=20, region="na-east-1", **options):
        super().__init__(**options)
        self.annealing_time = annealing_time
        try:
            qpu = DWaveSampler(region=region, solver={"topology__type": "zephyr"})
        except Exception as exc:  # no token, no Zephyr QPU online, network down
            raise SolverUnavailable(
                f"cannot reach a Zephyr QPU in {region}: {exc}"
            ) from exc
        self.sampler = EmbeddingComposite(
            qpu, embedding_parameters={"random_seed": self.seed}
        )
        self.name = f"quantum-annealing/{qpu.properties['chip_id']}"

    def _sample(self, bqm):
        try:
            sampleset = self.sampler.sample(
                bqm,
                num_reads=self.num_reads,
                annealing_time=self.annealing_time,
                return_embedding=True,
            )
        except ValueError as exc:  # minorminer found no embedding
            raise SolverUnavailable(f"cannot embed on {self.name}: {exc}") from exc

        info = sampleset.info
        chains = [
            len(chain) for chain in info["embedding_context"]["embedding"].values()
        ]
        return sampleset, {
            "annealing_time": self.annealing_time,
            "problem_id": info["problem_id"],
            "qpu_access_time": info["timing"]["qpu_access_time"],  # microseconds
            "physical_qubits": sum(chains),
            "max_chain_length": max(chains),
            "chain_strength": info["embedding_context"]["chain_strength"],
            "chain_break_fraction": float(
                np.average(
                    sampleset.record.chain_break_fraction,
                    weights=sampleset.record.num_occurrences,
                )
            ),
        }
