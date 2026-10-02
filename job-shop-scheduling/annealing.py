import time

import dimod
import minorminer
import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from formulation import best_feasible, build_formulation
from results import SolverUnavailable, make_result


# a D-Wave QPU, found through the local dwave config
def dwave_sampler():
    try:
        return DWaveSampler()
    except Exception as exc:  # no token, no QPU online, network down
        raise SolverUnavailable(f"cannot reach a D-Wave QPU: {exc}") from exc


# samples the QUBO with any dimod sampler: a D-Wave QPU from dwave_sampler(), or locally a simulator
# such as dwave.samplers.PathIntegralAnnealingSampler
# a QPU only has certain couplers, so the QUBO is minor-embedded onto it first
# any other keyword goes straight to the sampler, e.g. seed for a simulator or annealing_time for a QPU
def solve_annealing(problem, sampler, num_reads=1000, **parameters):
    formulation = build_formulation(problem)
    bqm = dimod.BinaryQuadraticModel.from_qubo(formulation.Q, offset=formulation.offset)
    name = f"annealing/{sampler.properties.get('chip_id', type(sampler).__name__)}"

    structured = isinstance(sampler, dimod.Structured)
    # read before the timer starts, as a QPU fetches its coupler list on first use
    target = sampler.edgelist if structured else None

    t0 = time.perf_counter()
    if structured:
        # timed separately, as the embedding is a one-off cost
        # self-loops make sure variables with no couplings are embedded too
        source = list(bqm.quadratic) + [(v, v) for v in bqm.variables]
        embedding = minorminer.find_embedding(source, target)
        embedding_time = time.perf_counter() - t0
        # minorminer returns an empty embedding when it fails
        if not embedding:
            raise SolverUnavailable(f"cannot embed {bqm.num_variables} variables")
        sampler = FixedEmbeddingComposite(sampler, embedding)
        parameters["return_embedding"] = True  # to report the chain strength

    sampleset = sampler.sample(bqm, num_reads=num_reads, **parameters)
    sampleset.resolve()  # wait for the QPU's asynchronous answer inside the timing
    t1 = time.perf_counter()

    # sampleset columns follow the BQM's own variable order, so pick them out by label
    columns = [sampleset.variables.index(n) for n in range(formulation.num_variables)]
    start_times, samples = best_feasible(
        sampleset.record.sample[:, columns], sampleset.record.num_occurrences, formulation, problem
    )

    metadata = {
        "calls_unit": "reads",
        "variables": formulation.num_variables,
        "lowest_energy": sampleset.first.energy,
    }
    if structured:
        chains = [len(chain) for chain in embedding.values()]
        qpu_access_time = sampleset.info.get("timing", {}).get("qpu_access_time")  # microseconds
        metadata.update(
            {
                "embedding_time": embedding_time,  # seconds, within the TTS
                "qpu_access_time": qpu_access_time and qpu_access_time / 1e6,  # seconds
                "physical_qubits": sum(chains),
                "max_chain_length": max(chains),
                "chain_strength": sampleset.info["embedding_context"]["chain_strength"],
                "chain_break_fraction": float(
                    np.average(
                        sampleset.record.chain_break_fraction, weights=sampleset.record.num_occurrences
                    )
                ),
            }
        )
    return make_result(problem, name, start_times, t1 - t0, num_reads, metadata, samples)


# the logical problem size, and on a QPU the expected access time for the reads
# (quoted for one qubit per variable, so a lower bound once chains are added)
def estimate_annealing(problem, sampler, num_reads=1000, **parameters):
    formulation = build_formulation(problem)
    bqm = dimod.BinaryQuadraticModel.from_qubo(formulation.Q, offset=formulation.offset)

    estimate = {
        "variables": bqm.num_variables,
        "couplings": bqm.num_interactions,
        "num_reads": num_reads,
        "qpu_access_time": None,
    }
    if isinstance(sampler, dimod.Structured):
        estimate["hardware_qubits"] = len(sampler.nodelist)
    solver = getattr(sampler, "solver", None)
    if hasattr(solver, "estimate_qpu_access_time"):
        microseconds = solver.estimate_qpu_access_time(bqm.num_variables, num_reads=num_reads, **parameters)
        estimate["qpu_access_time"] = float(microseconds) / 1e6  # seconds
    return estimate
