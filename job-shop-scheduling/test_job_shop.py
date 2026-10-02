import dimod
import pytest
from classical import solve_exact
from formulation import build_formulation, decode
from problem import Problem
from qaoa import counts_to_bits, sample, sampler_and_pass_manager
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# the first few random problems that the bounds don't already solve, small enough to brute force
def small_problems(count=3, max_variables=16):
    problems = []
    for seed in range(1000):
        problem = Problem.random(
            num_tasks=4,
            num_machines=3,
            min_parents=2,
            max_parents=2,
            min_duration=1,
            max_duration=5,
            seed=seed,
        )
        if (
            problem.lower_bound < problem.horizon
            and build_formulation(problem).num_variables <= max_variables
        ):
            problems.append(problem)
            if len(problems) == count:
                return problems


@pytest.mark.parametrize("problem", small_problems(), ids=lambda problem: problem.name)
def test_qubo_ground_states_are_optimal_schedules(problem):
    optimum = solve_exact(problem).makespan
    formulation = build_formulation(problem)
    bqm = dimod.BinaryQuadraticModel.from_qubo(formulation.Q, offset=formulation.offset)
    sampleset = dimod.ExactSolver().sample(bqm)

    # the lowest energy is the optimal makespan, and every state with it is an optimal schedule
    energies = sampleset.record.energy
    assert energies.min() == optimum
    columns = [sampleset.variables.index(n) for n in range(formulation.num_variables)]
    for bits in sampleset.record.sample[energies == energies.min()][:, columns]:
        start_times = decode(bits, formulation)[: len(problem)]
        assert problem.is_feasible(start_times)
        assert problem.makespan(start_times) == optimum


# Qiskit prints qubit 0 as the rightmost character of a counts key; counts_to_bits and decode undo that
def test_qiskit_counts_decode_in_variable_order():
    problem = Problem([(0, 1, ()), (0, 2, ())])  # two tasks sharing a machine
    formulation = build_formulation(problem, for_qaoa=True)
    wanted = [(0, 2), (1, 0), (2, 3)]  # (task, start time), with the makespan task last

    circuit = QuantumCircuit(formulation.num_variables)
    for n, variable in enumerate(formulation.variables):
        if variable in wanted:
            circuit.x(n)
    circuit.measure_all()
    sampler, _ = sampler_and_pass_manager(AerSimulator(), shots=1, seed=0)
    counts, _ = sample(sampler, circuit, [])

    (key,) = counts
    assert decode(counts_to_bits(key), formulation) == [2, 0, 3]


def test_is_feasible():
    # T0 and T1 share machine 0; T2 runs on machine 1 once T0 is done
    problem = Problem([(0, 2, ()), (0, 1, ()), (1, 1, (0,))])
    assert problem.is_feasible([0, 2, 2])
    assert not problem.is_feasible([0, 1, 2])  # T1 overlaps T0
    assert not problem.is_feasible([0, 2, 1])  # T2 starts before T0 finishes
    assert not problem.is_feasible([0, None, 2])  # T1 has no start time
