import collections
from typing import Literal
import numpy as np
from qiskit import (QuantumCircuit, transpile)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import Estimator
from qiskit.providers import Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from qiskit_algorithms import (TimeEvolutionProblem, TimeEvolutionResult,
                               TrotterQRTE)
from qiskit_algorithms.observables_evaluator import estimate_observables

from qiskit_nature.second_q.mappers import (BosonicLogarithmicMapper,
                                            BravyiKitaevMapper, MixedMapper)
from qiskit_nature.second_q.operators import BosonicOp, FermionicOp, MixedOp

np.set_printoptions(precision=6, suppress=True)

# Messages
def message_output(message: str, filename: str | None = None) -> None:
    "Allows to print messages to file at run-time"
    # First, print to console
    print(message, end="")
    # Then print to file
    if filename is not None:
        with open(filename, 'a+', encoding="UTF-8") as output:
            output.write(message)
            output.close()

def get_h_qed(el_eigenvals: list[float],
              ph_energies: list[float],
              lm_couplings: list[float]) -> tuple[FermionicOp, BosonicOp, MixedOp, MixedOp]:
    """"
    This method generates the QED Hamiltonian for a given set of electron eigenvalues,
    photon energies and light-matter couplings.
    
    Args:
        el_eigenvals: The electron eigenvalues (in Hartree).
        ph_energies: The photon energies (in Hartree).
        lm_couplings: The light-matter couplings (in Hartree).
    """
    # The QED Hamiltonian is componsed of three terms, the uncoupled electron Hamiltonian,
    # the uncoupled photon Hamiltonian and the interaction Hamiltonian.
    # First, generate the uncoupled electron Hamiltonian
    h_el = FermionicOp({})
    for i, eigenval in enumerate(el_eigenvals):
        h_el += FermionicOp({f'+_{i} -_{i}': eigenval}, num_spin_orbitals=len(el_eigenvals))
    # Next, generate the uncoupled photon Hamiltonian
    assert len(ph_energies) == len(lm_couplings)
    h_ph = BosonicOp({})
    for i, ph_energy in enumerate(ph_energies):
        h_ph += ph_energy * BosonicOp({'': 0.5, f'+_{i} -_{i}': 1}, num_modes=len(ph_energies))
    # Finally, generate the interaction Hamiltonian
    h_int = MixedOp({})
    for i, eigenval in enumerate(el_eigenvals):
        for j, eigenval in enumerate(el_eigenvals):
            if j >= i:
                continue
            absorption_op = FermionicOp({f'+_{i} -_{j}': 1}, num_spin_orbitals=len(el_eigenvals))
            emission_op = FermionicOp({f'+_{j} -_{i}': 1}, num_spin_orbitals=len(el_eigenvals))
            ph_creation = BosonicOp({})
            ph_annihilation = BosonicOp({})
            for k, ph_energy in enumerate(ph_energies):
                ph_creation += BosonicOp({f'+_{k}': lm_couplings[k]}, num_modes=len(ph_energies))
                ph_annihilation += BosonicOp({f'-_{k}': lm_couplings[k]},num_modes=len(ph_energies))
            # Build the interaction
            h_int += MixedOp({
                ("F", "B"): [(-1., absorption_op, ph_annihilation),(-1., emission_op, ph_creation)],
            })
    # Finally, put it all together
    h_qed = MixedOp(
        {
            ("F",): [(1.0, h_el)],
            ("B",): [(1.0, h_ph)],
        }
    ) + h_int
    return h_el, h_ph, h_int, h_qed

def get_mapper(number_of_modes: int, number_of_fock_states: int) -> MixedMapper:
    """This method returns the mapper to be used for the QED Hamiltonian."""
    # First, define the mappers for the fermionic and bosonic parts of the Hamiltonian
    bk_mapper = BravyiKitaevMapper()
    b_log_mapper = BosonicLogarithmicMapper(max_occupation=number_of_fock_states)
    # Then, define the mixed mapper
    hilbert_space_registers = {"F": 2, "B": number_of_modes} # Position, Length
    hilbert_space_registers_types = {"F": FermionicOp, "B": BosonicOp} # Position, type
    return MixedMapper(
        {"F": bk_mapper, "B": b_log_mapper},
        hilbert_space_registers,
        hilbert_space_registers_types)

def get_time_evolution_circuit(
        h_mapped: SparsePauliOp,
        initial_state: Statevector,
        delta_t: float) -> QuantumCircuit:
    """"This method generates the circuit corresponding to the time evolution of a single timestep
    of the given Hamiltonian."""
    # First, perform a single Trotter step
    problem = TimeEvolutionProblem(h_mapped, initial_state=initial_state, time=delta_t)
    trotter = TrotterQRTE(estimator=Estimator(), num_timesteps=1)
    # Return the circuit corresponding to the evolved state
    return trotter.evolve(problem).evolved_state

def time_evolve(h_mapped: SparsePauliOp,
                initial_state: Statevector,
                aux_operators: list[SparsePauliOp],
                final_time: float,
                delta_t: float) -> TimeEvolutionResult:
    """"
    This method evolves the given initial state under the given Hamiltonian for the given time.
    """
    problem = TimeEvolutionProblem(
        h_mapped, initial_state=initial_state, aux_operators=aux_operators, time=final_time)
    trotter = TrotterQRTE(estimator=Estimator(), num_timesteps=int(final_time / delta_t))
    return trotter.evolve(problem)

def get_keys_as_list(dictionary) -> list:
    return list(map(lambda x: x[0], dictionary.items()))

def qubit_idx_to_str_idx(qubit_idx: int, str_len: int) -> int:
    return str_len - qubit_idx - 1

def custom_time_evolve(h_mapped: SparsePauliOp,
                       observables_mapped: list[SparsePauliOp],
                       initial_state: Statevector,
                       evolution_stategy: Literal["tc", "ct", "tct"],
                       optimization_level: int,
                       backend: Backend,
                       estimator: Estimator,
                       final_time: float,
                       delta_t: float) -> TimeEvolutionResult:
    """"
    This method allows to time evolve an optimized circuit for a given Hamiltonian and initial
    state.

    Args:
        h_mapped: The Hamiltonian (already mapped to the qubit space) to be time evolved.
        observables_mapped: The observables (already mapped to the qubit space) to be estimated.
        initial_state: The initial state of the system.
        evolution_stategy: The optimization strategy to be used by the transpiler.\n
            - "tc": First transpile the single timestep circuit and then combine the circuits.\n
            - "ct": First combine the unoptimized circuit, then transpile the combined circuit.\n
            - "tct": First transpile the single timestep circuit and then combine the circuits.
                Finally, transpile again the combined circuit.
        optimization_level: The optimization level to be used by the transpiler.
        backend: The backend to be used for the simulation.
        estimator: The estimator to be used for the simulation.
        final_time: The final time of the simulation.
        delta_t: The time step of the simulation.

    Returns:
        A tuple containing the final result of the simulation and the optimized time evolution
        circuit for a single time step.
    """

    # 1. Initialize the single timestep evolution circuit
    time_evolution_circuit = QuantumCircuit(h_mapped.num_qubits)
    single_step_evolution_gate = PauliEvolutionGate(h_mapped, delta_t, synthesis=SuzukiTrotter())
    time_evolution_circuit.append(single_step_evolution_gate, time_evolution_circuit.qubits)
    # 2. Define thew array with the simesteps
    time: list[float] = [(delta_t * x) for x in range(1, int(final_time / delta_t) + 1)]
    if evolution_stategy == "tc":
        return transpile_combine_strategy(
            time_evolution_circuit,
            initial_state,
            observables_mapped,
            time,
            optimization_level,
            estimator,
            backend)
    if evolution_stategy == "ct":
        return combine_transpile_strategy(
            time_evolution_circuit,
            initial_state,
            observables_mapped,
            time,
            optimization_level,
            estimator,
            backend)
    if evolution_stategy == "tct":
        return transpile_combine_transpile_strategy(
            time_evolution_circuit,
            initial_state,
            observables_mapped,
            time,
            optimization_level,
            estimator,
            backend)
    raise ValueError("Invalid evolution strategy")


def get_optimization_map(circuit: QuantumCircuit) -> dict:
    """
    Now define a map from the old circuit layout to the new one. This must be done because the
    transpiler could change the qubit layout.
    The optimized layout map is a dictionary that maps the qubit index in the optimized circuit
    to the qubit index in the initial circuit.
    """
    optimized_layout_map = {}
    for item in circuit.layout.initial_virtual_layout().get_virtual_bits().items():
        optimized_layout_map[item[1]] = item[0]._index
    return collections.OrderedDict(sorted(optimized_layout_map.items()))

def optimize_obervables(observables: list[SparsePauliOp], optimized_layout_map: dict, num_qubits: int) -> list[SparsePauliOp]:
    optimized_observables: list[SparsePauliOp] = []
    # First loop over the observables that should be computed
    for idx, observable in enumerate(observables):
        optimized_observable: list[tuple[str, complex | np.complex128]] = []
        # Then loop over the terms of the observable. These are the Pauli strings (e.g. 'IIIZ')
        for op, coeff in observable.to_list():
            # For each term, we need to map the qubit indices to the new layout
            optimized_op: str = "".join(
                op[qubit_idx_to_str_idx(s, num_qubits)]
                for s in np.asarray([optimized_layout_map[k] for k in range(num_qubits)])
            )
            optimized_observable.append((optimized_op[::-1], coeff))
        optimized_observables.append(SparsePauliOp.from_list(optimized_observable))
    return optimized_observables

def optimize_init_state(initial_state: Statevector, optimized_layout_map: dict, num_qubits: int) -> Statevector:
    old_init_state: str = get_keys_as_list(initial_state.to_dict())[0]
    optimized_init_state: str = "".join(
        old_init_state[qubit_idx_to_str_idx(s, num_qubits)]
        for s in np.asarray([optimized_layout_map[k] for k in range(num_qubits)])
    )
    return Statevector.from_label(optimized_init_state[::-1])

def transpile_combine_strategy(single_step_evolution_circuit: QuantumCircuit,
                               initial_state: Statevector,
                               observables: list[SparsePauliOp],
                               time: list[float],
                               optimization_level: int,
                               estimator: Estimator,
                               backend: Backend) -> QuantumCircuit:
    """"
    This method combines the single step evolution circuit and transpiles the combined circuit.
    """
    # 1. Optimize the circuit
    single_step_evolution_circuit_optimized: QuantumCircuit = \
        transpile(single_step_evolution_circuit, backend, optimization_level=optimization_level)
    # 2. Get the optimization map
    optimized_layout_map = get_optimization_map(single_step_evolution_circuit_optimized)
    # 3. Initialized the evolved state
    evolved_state = QuantumCircuit(single_step_evolution_circuit_optimized.qubits)
    # 4. Optimize the initial state to match the new qubit layout.
    optimized_init_state: Statevector = \
        optimize_init_state(initial_state, optimized_layout_map, evolved_state.num_qubits)
    # 4.1 Prepare the state in the circuit
    evolved_state.prepare_state(optimized_init_state)
    # 5. Optimize the observables to match the new qubit layout
    optimized_observables: list[SparsePauliOp] = \
        optimize_obervables(observables, optimized_layout_map, evolved_state.num_qubits)
    # 6. Get the observables at time 0
    observables_result = [
        estimate_observables(
            estimator,
            evolved_state,
            optimized_observables,
            None,
            1e-12
        )]
    for idx, t in enumerate(time):
        message_output(f"Time step {idx + 1}/{len(time)}\n", "output")
        evolved_state.compose(single_step_evolution_circuit_optimized, inplace=True)
        # Print info
        message_output(
        f"Circuit optimized with level: {optimization_level}. Operation count:\n", "output")
        operations = evolved_state.count_ops()
        for op in operations:
            message_output(f"{op}: {operations[op]}\n", "output")
        # Save circuit (only for the first two steps because after that it gets too long)
        if idx < 2:
            evolved_state.draw(output="mpl", filename=f"results/circuits/circuit_t_{t:.4f}.png")
        # Compute observables
        observables_result.append(
            estimate_observables(
                estimator,
                evolved_state,
                optimized_observables,
                None,
                1e-12
            )
        )
    # Return the result
    return TimeEvolutionResult(evolved_state,
                                observables_result[-1],
                                observables_result,
                                times=np.array([0.0] + time))

def combine_transpile_strategy(single_step_evolution_circuit: QuantumCircuit,
                               initial_state: Statevector,
                               observables: list[SparsePauliOp],
                               time: list[float],
                               optimization_level: int,
                               estimator: Estimator,
                               backend: Backend) -> QuantumCircuit:
    """"
    This method transpiles the combined circuit.
    """
    # 1. Define and initialize the evolved state
    evolved_state = QuantumCircuit(single_step_evolution_circuit.qubits)
    evolved_state.prepare_state(initial_state)
    # 2. Get the t = 0 observables
    observables_result = [
        estimate_observables(estimator, evolved_state, observables, None, 1e-12)
        ]
    # 3. Time evolution
    for idx, t in enumerate(time):
        message_output(f"Time step {idx + 1}/{len(time)}\n", "output")
        # First, compose the unoptimized circuit
        evolved_state.compose(single_step_evolution_circuit, inplace=True)
        # Then, transpile the combined circuit
        optimized_circuit: QuantumCircuit = \
            transpile(evolved_state, backend, optimization_level=optimization_level)
        # Save circuit (only for the first two steps because after that it gets too long)
        if idx < 2:
            optimized_circuit.draw(output="mpl", filename=f"results/circuits/circuit_t_{t:.4f}.png")
        message_output(
        f"Circuit optimized with level: {optimization_level}. Operation count:\n", "output")
        operations = optimized_circuit.count_ops()
        for op in operations:
            message_output(f"{op}: {operations[op]}\n", "output")
        # Get the optimization map and optimize the observables
        optimized_layout_map = get_optimization_map(optimized_circuit)
        optimized_observables: list[SparsePauliOp] = \
            optimize_obervables(observables, optimized_layout_map, optimized_circuit.num_qubits)
        # Get the observables at time t
        observables_result.append(
            estimate_observables(estimator, optimized_circuit, optimized_observables, None, 1e-12)
        )
    # Return the result
    return TimeEvolutionResult(evolved_state,
                                observables_result[-1],
                                observables_result,
                                times=np.array([0.0] + time))

def transpile_combine_transpile_strategy(single_step_evolution_circuit: QuantumCircuit,
                                        initial_state: Statevector,
                                        observables: list[SparsePauliOp],
                                        time: list[float],
                                        optimization_level: int,
                                        estimator: Estimator,
                                        backend: Backend) -> QuantumCircuit:
    """"
    This method combines the single step evolution circuit, transpiles the combined
    circuit and then transpiles the combined circuit.
    """
    # 1. Optimize the circuit
    single_step_evolution_circuit_optimized: QuantumCircuit = \
        transpile(single_step_evolution_circuit, backend, optimization_level=optimization_level)
    # 2. Get the optimization map
    optimized_layout_map = get_optimization_map(single_step_evolution_circuit_optimized)
    # 3. Initialized the evolved state
    base_circuit = QuantumCircuit(single_step_evolution_circuit_optimized.qubits)
    # 4. Optimize the initial state to match the new qubit layout.
    t_0_optimized_init_state: Statevector = \
        optimize_init_state(initial_state, optimized_layout_map, base_circuit.num_qubits)
    # 4.1 Prepare the state in the circuit
    base_circuit.prepare_state(t_0_optimized_init_state)
    # 5. Optimize the observables to match the new qubit layout
    t_0_optimized_observables: list[SparsePauliOp] = \
        optimize_obervables(observables, optimized_layout_map, base_circuit.num_qubits)
    # 6. Get the observables at time 0
    observables_result = [
        estimate_observables(
            estimator,
            base_circuit,
            t_0_optimized_observables,
            None,
            1e-12
        )]
    # 7. Time evolution
    optimized_circuit_without_initial_state = QuantumCircuit(single_step_evolution_circuit_optimized.qubits)
    for idx, t in enumerate(time):
        message_output(f"Time step {idx + 1}/{len(time)}\n", "output")
        base_circuit = QuantumCircuit(single_step_evolution_circuit_optimized.qubits)
        base_circuit.prepare_state(t_0_optimized_init_state)
        # 1. Compose the unoptimized circuit
        optimized_circuit_without_initial_state.compose(single_step_evolution_circuit_optimized, inplace=True)
        circuit: QuantumCircuit = base_circuit.compose(optimized_circuit_without_initial_state)
        # 2. Transpile the combined circuit
        optimized_circuit: QuantumCircuit = \
            transpile(circuit, backend, optimization_level=optimization_level)
        optimized_layout_map = get_optimization_map(optimized_circuit)
        # Save circuit (only for the first two steps because after that it gets too long)
        if idx < 2:
            optimized_circuit.draw(output="mpl", filename=f"results/circuits/circuit_t_{t:.4f}.png")
        message_output(
        f"Circuit optimized with level: {optimization_level}. Operation count:\n", "output")
        operations = optimized_circuit.count_ops()
        for op in operations:
            message_output(f"{op}: {operations[op]}\n", "output")
        # Optimize the observables
        optimized_observables: list[SparsePauliOp] = \
            optimize_obervables(t_0_optimized_observables, optimized_layout_map, optimized_circuit.num_qubits)
        # Get the observables at time t
        observables_result.append(
            estimate_observables(estimator, optimized_circuit, optimized_observables, None, 1e-12)
        )
    # Return the result
    return TimeEvolutionResult(optimized_circuit_without_initial_state,
                                observables_result[-1],
                                observables_result,
                                times=np.array([0.0] + time))