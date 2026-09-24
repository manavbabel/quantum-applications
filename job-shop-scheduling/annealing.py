import time

import dimod
import minorminer
import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from formulation import best_feasible, build_formulation
from solver import Solver, SolverUnavailable, fraction


# samples the QUBO with any dimod sampler
# by default a D-Wave QPU, found through the local dwave config
# locally, a simulated quantum annealer such as dwave.samplers.PathIntegralAnnealingSampler
# a QPU (or a dwave.system.testing.MockDWaveSampler) only has certain couplers, so there the QUBO
# is minor-embedded first
class AnnealingSolver(Solver):

    def __init__(self, sampler=None):
        if sampler is None:
            try:
                sampler = DWaveSampler()
            except Exception as exc:  # no token, no QPU online, network down
                raise SolverUnavailable(f"cannot reach a D-Wave QPU: {exc}") from exc
        self.sampler = sampler
        self.name = f"annealing/{sampler.properties.get('chip_id', type(sampler).__name__)}"

    # any other keyword goes straight to the sampler, e.g. seed for a simulator or annealing_time for a QPU
    def _solve(self, problem, num_reads=1000, **parameters):
        formulation = build_formulation(problem)
        bqm = dimod.BinaryQuadraticModel.from_qubo(formulation.Q, offset=formulation.offset)

        sampler = self.sampler
        structured = isinstance(sampler, dimod.Structured)
        # a QPU builds its coupler list on first use, which is a fact about the device, not part of solving
        target = sampler.edgelist if structured else None

        t0 = time.perf_counter()
        if structured:
            # finding the embedding is a one-off cost ahead of the reads, so it is timed on its own
            # each variable gets a self-loop, so one with no couplings is still embedded
            source = list(bqm.quadratic) + [(v, v) for v in bqm.variables]
            embedding = minorminer.find_embedding(source, target)
            embedding_time = time.perf_counter() - t0
            # minorminer gives up by returning an empty embedding, rather than raising
            if not embedding:
                raise SolverUnavailable(f"cannot embed {bqm.num_variables} variables on {self.name}")
            sampler = FixedEmbeddingComposite(sampler, embedding)
            parameters["return_embedding"] = True  # to report the chain strength

        sampleset = sampler.sample(bqm, num_reads=num_reads, **parameters)
        sampleset.resolve()  # a QPU answers asynchronously, so wait for it inside the timing
        t1 = time.perf_counter()

        # sampleset columns follow the BQM's own variable order, so pick them out by label
        columns = [sampleset.variables.index(n) for n in range(formulation.num_variables)]
        # each read is a run of its own
        start_times, samples = best_feasible(
            sampleset.record.sample[:, columns],
            sampleset.record.num_occurrences,
            formulation,
            problem,
        )

        metadata = {
            "calls_unit": "reads",
            "variables": formulation.num_variables,
            "lowest_energy": sampleset.first.energy,
            "feasible_fraction": 1 - fraction(samples, None),
        }
        if structured:
            chains = [len(chain) for chain in embedding.values()]
            qpu_access_time = sampleset.info.get("timing", {}).get("qpu_access_time")  # microseconds
            metadata.update({
                "embedding_time": embedding_time,  # seconds, within the TTS
                "qpu_access_time": qpu_access_time and qpu_access_time / 1e6,  # seconds
                "physical_qubits": sum(chains),
                "max_chain_length": max(chains),
                "chain_strength": sampleset.info["embedding_context"]["chain_strength"],
                "chain_break_fraction": float(
                    np.average(
                        sampleset.record.chain_break_fraction,
                        weights=sampleset.record.num_occurrences,
                    )
                ),
            })
        return start_times, t1 - t0, num_reads, metadata, samples

    # the logical problem that has to be embedded, and how much QPU time the reads should take
    # only a QPU can price its time, and it is quoted for one qubit per variable, which is a lower
    # bound: chains in the embedding take more
    def estimate(self, problem, num_reads=1000, **parameters):
        formulation = build_formulation(problem)
        bqm = dimod.BinaryQuadraticModel.from_qubo(formulation.Q, offset=formulation.offset)

        estimate = {
            "variables": bqm.num_variables,
            "couplings": bqm.num_interactions,
            "num_reads": num_reads,
            "qpu_access_time": None,
        }
        if isinstance(self.sampler, dimod.Structured):
            estimate["hardware_qubits"] = len(self.sampler.nodelist)
        solver = getattr(self.sampler, "solver", None)
        if hasattr(solver, "estimate_qpu_access_time"):
            microseconds = solver.estimate_qpu_access_time(
                bqm.num_variables, num_reads=num_reads, **parameters
            )
            estimate["qpu_access_time"] = float(microseconds) / 1e6  # seconds
        return estimate
