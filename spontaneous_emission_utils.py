from typing import List, Literal
import numpy as np
from qiskit import (QuantumCircuit, transpile)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.quantumregister import Qubit
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
from qat.interop.qiskit import qiskit_to_qlm

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

def get_h_qed(el_eigenvals: List[float],
              ph_energies: List[float],
              lm_couplings: List[float]) -> tuple[FermionicOp, BosonicOp, MixedOp, MixedOp]:
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
                ("F", "B"): [(1., absorption_op, ph_annihilation),(1., emission_op, ph_creation)],
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
                aux_operators: List[SparsePauliOp],
                final_time: float,
                delta_t: float) -> TimeEvolutionResult:
    """"
    This method evolves the given initial state under the given Hamiltonian for the given time.
    """
    problem = TimeEvolutionProblem(
        h_mapped, initial_state=initial_state, aux_operators=aux_operators, time=final_time)
    trotter = TrotterQRTE(estimator=Estimator(), num_timesteps=int(final_time / delta_t))
    return trotter.evolve(problem)

def custom_time_evolve(h_mapped: SparsePauliOp,
                       observables_mapped: List[SparsePauliOp],
                       initial_state: dict[int,
                                           tuple[complex | np.complex128, complex | np.complex128]],
                       evolution_stategy: Literal["tc", "ct", "tct"],
                       evolution_synthesis: Literal["suzuki_trotter", "lie_trotter"],
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
        initial_state: It shall be a dictionary where the key is the qubit index and the value is
            a list of complex numbers representing the state of the qubit. e.g. To initialize the
            qubit in the state |0>, the value shall be (1, 0).
        evolution_stategy: The optimization strategy to be used by the transpiler.\n
            - "tc": First transpile the single timestep circuit and then combine the circuits.\n
            - "ct": First combine the unoptimized circuit, then transpile the combined circuit.\n
            - "tct": First transpile the single timestep circuit and then combine the circuits.
                Finally, transpile again the combined circuit.
        evolution_synthesis: The synthesis method to be used time evolution.
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
    if evolution_synthesis == "suzuki_trotter":
        single_step_evolution_gate: PauliEvolutionGate = \
            PauliEvolutionGate(h_mapped, delta_t, synthesis=SuzukiTrotter())
    else:
        single_step_evolution_gate: PauliEvolutionGate = \
            PauliEvolutionGate(h_mapped, delta_t, synthesis=LieTrotter())
    time_evolution_circuit.append(single_step_evolution_gate, time_evolution_circuit.qubits)
    # 2. Define thew array with the simesteps
    time: List[float] = [(delta_t * x) for x in range(1, int(final_time / delta_t) + 1)]
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

def count_gates(qc: QuantumCircuit) -> dict[Qubit, int]:
    """
    This method counts the number of gates acting on each qubit of the circuit.

    Args:
        qc: The quantum circuit to be analyzed.

    Returns:
        A dictionary containing the number of gates acting on each qubit of the circuit.
    """
    gate_count: dict[Qubit, int] = { qubit: 0 for qubit in qc.qubits }
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count

def remove_idle_wires(qc: QuantumCircuit) -> QuantumCircuit:
    """
    This method removes the idle wires from the circuit. An idle wire is a qubit that has no gates,
    i.e. the number of gates acting on the qubit is zero.

    Args:
        qc: The quantum circuit to be analyzed.

    Returns:
        A QuantumCircuit with the idle wires removed.
    """
    qc_out = qc.copy()
    gate_count = count_gates(qc_out)
    for qubit, count in gate_count.items():
        if count == 0:
            qc_out.qubits.remove(qubit)
    return qc_out

def optimize_obervables(observables: List[SparsePauliOp],
                        circuit: QuantumCircuit) -> List[SparsePauliOp]:
    """
    This method optimizes the observables to match the new qubit layout.
    Args:
        observables: The observables to be optimized.
        circuit: The circuit with the new qubit layout, as obtained from the transpiler.
    Returns:
        A list containing the optimized observables.
    """
    optimized_observables: List[SparsePauliOp] = []
    final_index_layout: List[int] = circuit.layout.final_index_layout()
    # First loop over the observables that should be computed
    for _, observable in enumerate(observables):
        optimized_observable: List[tuple[str, complex | np.complex128]] = []
        # Then loop over the terms of the observable. These are the Pauli strings (e.g. 'IIIZ')
        for op, coeff in observable.to_list():
            optimized_op: List[str] = ["I"] * circuit.num_qubits
            # For each term, we need to map the qubit indices to the new layout
            for idx, i in enumerate(final_index_layout):
                optimized_op[i] = op[len(final_index_layout) - 1 - idx]
            optimized_observable.append(("".join(optimized_op)[::-1], coeff))
        optimized_observables.append(SparsePauliOp.from_list(optimized_observable))
    return optimized_observables

def remove_idle_qubit_from_obervables(observables: List[SparsePauliOp],
                        circuit: QuantumCircuit) -> List[SparsePauliOp]:
    """
    This method optimizes the observables to match the new qubit layout.
    Args:
        observables: The observables to be optimized.
        circuit: The circuit with the new qubit layout, as obtained from the transpiler.
    Returns:
        A list containing the optimized observables.
    """
    optimized_observables: List[SparsePauliOp] = []
    # First loop over the observables that should be computed
    for _, observable in enumerate(observables):
        optimized_observable: List[tuple[str, complex | np.complex128]] = []
        # Then loop over the terms of the observable. These are the Pauli strings (e.g. 'IIIZ')
        for op, coeff in observable.to_list():
            optimized_op: List[str] = ["I"] * circuit.num_qubits
            # For each term, we need to map the qubit indexes to the new layout
            # `circuit.qubits` is an ordered list of the qubits in the circuit. The first element
            # represents the most significant qubit
            for idx, qubit in enumerate(circuit.qubits):
                # Get info about this qubit. The index represents the position of the qubit in the
                # string with the original number of qubits
                qubit_idx: int = circuit.find_bit(qubit).index
                optimized_op[idx] = op[len(op) - 1 - qubit_idx]
            optimized_observable.append(("".join(optimized_op)[::-1], coeff))
        optimized_observables.append(SparsePauliOp.from_list(optimized_observable))
    return optimized_observables

def optimize_init_state(
        initial_state: dict[int, tuple[complex | np.complex128, complex | np.complex128]],
        circuit: QuantumCircuit
    ) -> dict[int, tuple[complex | np.complex128, complex | np.complex128]]:
    """
    This method optimizes the initial state to match the new qubit layout.
    Args:
        initial_state: The initial state to be optimized.
        circuit: The circuit with the new qubit layout, as obtained from the transpiler.
    Returns:
        A Statevector containing the optimized initial state.
    """
    optimized_init_state: dict[int, tuple[complex | np.complex128, complex | np.complex128]] = {}
    for idx, i in enumerate(circuit.layout.final_index_layout()):
        optimized_init_state[i] = initial_state[idx]
    return optimized_init_state

def prepare_circuit(
        circuit: QuantumCircuit,
        initial_state: dict[int, tuple[complex | np.complex128, complex | np.complex128]]
    ) -> QuantumCircuit:
    """
    This method prepares the circuit with the initial state.
    Args:
        circuit: The circuit to be prepared.
        initial_state: The initial state of the system. It shall be a dictionary where the
            key is the qubit index and the value is a list of complex numbers representing
            the state of the qubit.
    Returns:
        A QuantumCircuit with the initial state prepared.
    """
    for qubit, state in initial_state.items():
        circuit.prepare_state([state[0], state[1]], qubit)
    return circuit

def transpile_combine_strategy(single_step_evolution_circuit: QuantumCircuit,
                               initial_state: dict[
                                   int, tuple[complex | np.complex128, complex | np.complex128]],
                               observables: List[SparsePauliOp],
                               time: List[float],
                               optimization_level: int,
                               estimator: Estimator,
                               backend: Backend) -> QuantumCircuit:
    """"
    This method combines the single step evolution circuit and transpiles the combined circuit.
    """
    # 1. Optimize the circuit
    single_step_evolution_circuit_optimized: QuantumCircuit = \
        transpile(single_step_evolution_circuit, backend, optimization_level=optimization_level)
    # 2. Initialized the evolved state
    optimized_circuit = QuantumCircuit(single_step_evolution_circuit_optimized.qubits)
    # 3. Optimize the initial state to match the new qubit layout.
    optimized_init_state = \
        optimize_init_state(initial_state, single_step_evolution_circuit_optimized)
    # 3.1 Prepare the state in the circuit
    optimized_circuit: QuantumCircuit = prepare_circuit(optimized_circuit, optimized_init_state)
    # 4. Optimize the observables to match the new qubit layout
    optimized_observables: List[SparsePauliOp] = \
            optimize_obervables(observables, single_step_evolution_circuit_optimized)
    # 5. Get the observables at time 0
    observables_result = [
        estimate_observables(
            estimator,
            optimized_circuit,
            optimized_observables,
            None,
            1e-12
        )]
    # 6. Time evolution
    for idx, t in enumerate(time):
        message_output(f"Time step {idx + 1}/{len(time)}\n", "output")
        optimized_circuit.compose(single_step_evolution_circuit_optimized, inplace=True)
        # Print info
        message_output(
        f"Circuit optimized with level: {optimization_level}. Operation count:\n", "output")
        operations = optimized_circuit.count_ops()
        for op in operations:
            message_output(f"{op}: {operations[op]}\n", "output")
        # Save circuit (only for the first two steps because after that it gets too long)
        if idx < 2 and optimized_circuit.num_qubits <= 8:
            optimized_circuit.draw(output="mpl", filename=f"results/circuits/circuit_t_{t:.4f}.png")
        # Compute observables
        observables_result.append(
            estimate_observables(
                estimator,
                optimized_circuit,
                optimized_observables,
                None,
                1e-12
            )
        )
    # Return the result
    return TimeEvolutionResult(optimized_circuit,
                                observables_result[-1],
                                observables_result,
                                times=np.array([0.0] + time))

def combine_transpile_strategy(single_step_evolution_circuit: QuantumCircuit,
                               initial_state: dict[
                                   int, tuple[complex | np.complex128, complex | np.complex128]],
                               observables: List[SparsePauliOp],
                               time: List[float],
                               optimization_level: int,
                               estimator: Estimator,
                               backend: Backend) -> QuantumCircuit:
    """"
    This method transpiles the combined circuit.
    """
    # 1. Define and initialize the evolved state
    evolved_state = QuantumCircuit(single_step_evolution_circuit.qubits)
    evolved_state: QuantumCircuit = prepare_circuit(evolved_state, initial_state)
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

        # Optimize the observables
        optimized_observables: List[SparsePauliOp] = \
            optimize_obervables(observables, optimized_circuit)

        # Remove unused qubits from circuit description
        optimized_circuit = remove_idle_wires(optimized_circuit)
        optimized_observables = \
            remove_idle_qubit_from_obervables(optimized_observables, optimized_circuit)

        # Save circuit (only for the first two steps because after that it gets too long)
        if idx < 2 and optimized_circuit.num_qubits <= 8:
            optimized_circuit.draw(output="mpl", filename=f"results/circuits/circuit_t_{t:.4f}.png")
        message_output(
        f"Circuit optimized with level: {optimization_level}. Operation count:\n", "output")
        operations = optimized_circuit.count_ops()
        for op in operations:
            message_output(f"{op}: {operations[op]}\n", "output")

        qlm_circuit = qiskit_to_qlm(optimized_circuit, sep_measures=True)[0]
        file_name = f"results/circuits/circuit_qlm_t_{t:.4f}.circ"
        qlm_circuit.dump(file_name)

        # Get the observables at time t
        observables_result.append(
            estimate_observables(estimator, optimized_circuit, optimized_observables, None, 1e-12)
        )
    # Return the result
    return TimeEvolutionResult(evolved_state,
                                observables_result[-1],
                                observables_result,
                                times=np.array([0.0] + time))

def transpile_combine_transpile_strategy(
        single_step_evolution_circuit: QuantumCircuit,
        initial_state: dict[int, tuple[complex | np.complex128, complex | np.complex128]],
        observables: List[SparsePauliOp],
        time: List[float],
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
    # 2. Initialized the evolved state
    base_circuit = QuantumCircuit(single_step_evolution_circuit_optimized.qubits)
    # 3. Optimize the initial state to match the new qubit layout.
    t_0_optimized_init_state = \
        optimize_init_state(initial_state, single_step_evolution_circuit_optimized)
    # 3.1 Prepare the state in the circuit
    base_circuit: QuantumCircuit = prepare_circuit(base_circuit, t_0_optimized_init_state)
    # 4. Optimize the observables to match the new qubit layout
    t_0_optimized_observables: List[SparsePauliOp] = \
        optimize_obervables(observables, single_step_evolution_circuit_optimized)
    # 5. Get the observables at time 0
    observables_result = [
        estimate_observables(
            estimator,
            base_circuit,
            t_0_optimized_observables,
            None,
            1e-12
        )]
    # 6. Time evolution
    for idx, t in enumerate(time):
        message_output(f"Time step {idx + 1}/{len(time)}\n", "output")
        # 1. Compose the unoptimized circuit
        base_circuit.compose(single_step_evolution_circuit_optimized, inplace=True)
        # 2. Transpile the combined circuit
        optimized_circuit: QuantumCircuit = \
            transpile(base_circuit.copy(), backend, optimization_level=optimization_level)
        # Save circuit (only for the first two steps because after that it gets too long)
        if idx < 2 and optimized_circuit.num_qubits <= 8:
            optimized_circuit.draw(output="mpl", filename=f"results/circuits/circuit_t_{t:.4f}.png")
        message_output(
        f"Circuit optimized with level: {optimization_level}. Operation count:\n", "output")
        operations = optimized_circuit.count_ops()
        for op in operations:
            message_output(f"{op}: {operations[op]}\n", "output")
        # Optimize the observables
        optimized_observables: List[SparsePauliOp] = \
            optimize_obervables(t_0_optimized_observables, optimized_circuit)
        # Get the observables at time t
        observables_result.append(
            estimate_observables(estimator, optimized_circuit, optimized_observables, None, 1e-12)
        )
    # Return the result
    return TimeEvolutionResult(optimized_circuit,
                                observables_result[-1],
                                observables_result,
                                times=np.array([0.0] + time))
