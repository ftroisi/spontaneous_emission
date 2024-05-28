import numpy as np
from qiskit import (ClassicalRegister, QuantumCircuit, QuantumRegister,
                    transpile)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import Estimator
from qiskit.providers import Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter

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

def custom_time_evolve(h_mapped: SparsePauliOp,
                       observables_mapped: list[SparsePauliOp],
                       initial_state: Statevector,
                       optimization_level: int,
                       backend: Backend,
                       estimator: Estimator,
                       final_time: float,
                       delta_t: float) -> tuple[TimeEvolutionResult, QuantumCircuit]:
    """"
    This method allows to time evolve an optimized circuit for a given Hamiltonian and initial
    state.

    Args:
        h_mapped: The Hamiltonian (already mapped to the qubit space) to be time evolved.
        observables_mapped: The observables (already mapped to the qubit space) to be estimated.
        initial_state: The initial state of the system.
        optimization_level: The optimization level to be used by the transpiler.
        backend: The backend to be used for the simulation.
        estimator: The estimator to be used for the simulation.
        final_time: The final time of the simulation.
        delta_t: The time step of the simulation.

    Returns:
        A tuple containing the final result of the simulation and the optimized time evolution
        circuit for a single time step.
    """
    observables = []
    # Initialized the evolved state
    evolved_state = QuantumCircuit(
        QuantumRegister(h_mapped.num_qubits, 'q'), ClassicalRegister(h_mapped.num_qubits + 1, 'c'))
    # Initialize the time evolution circuit
    time_evolution_circuit = QuantumCircuit(h_mapped.num_qubits)
    single_step_evolution_gate = PauliEvolutionGate(h_mapped, delta_t, synthesis=LieTrotter(reps=1))
    time_evolution_circuit.append(single_step_evolution_gate, time_evolution_circuit.qubits)
    # Optimize the circuit
    time_evolution_circuit: QuantumCircuit = \
        transpile(time_evolution_circuit, backend, optimization_level=optimization_level)
    message_output(f"Circuit optimized with optimization level: {optimization_level}. Operation count:\n", "output")
    operations = time_evolution_circuit.count_ops()
    for op in operations:
        message_output(f"{op}: {operations[op]}\n", "output")
    # Define the initial state
    init_state = QuantumCircuit(h_mapped.num_qubits)
    init_state.prepare_state(initial_state)
    evolved_state.append(init_state, init_state.qubits)
    # Get the observables at time 0
    observables.append(
        estimate_observables(
            estimator,
            evolved_state,
            observables_mapped,
            None,
            1e-12
        )
    )
    # Time evolution
    time = [(delta_t * x) for x in range(1, int(final_time / delta_t) + 1)]
    for idx, t in enumerate(time):
        message_output(f"Time step {idx + 1}/{len(time)}\n", "output")
        evolved_state = evolved_state.compose(time_evolution_circuit)
        observables.append(
            estimate_observables(
                estimator,
                evolved_state,
                observables_mapped,
                None,
                1e-12
            )
        )
    time.insert(0, 0.0)
    # Return the result
    return (TimeEvolutionResult(evolved_state,
                                observables[-1],
                                observables,
                                times=np.array(time)),
            time_evolution_circuit)
