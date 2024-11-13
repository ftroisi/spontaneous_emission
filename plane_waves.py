import os
import sys
import time
from typing import Dict, List

import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_nature.second_q.operators import BosonicOp, FermionicOp, MixedOp

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as Estimator

sys.path.append('./')
import spontaneous_emission_utils as utils
import io_tools as io

C = 137.03599 # Speed of light in atomic units

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/circuits"):
    os.makedirs("results/circuits")

# Read input file
parsed_input_file = dict(io.read_input_file("input"))
# Define some useful variables
number_of_modes: int | None = parsed_input_file.get("number_of_modes", None)
modes_energies: List[float] = parsed_input_file.get("modes_energies", [])
cavity_length: float = parsed_input_file.get("cavity_length", 1.0)

# Define modes frequency and couplings
# Equation 6 of: https://www.pnas.org/doi/full/10.1073/pnas.1615509114#sec-5
# omega_a = pi * c * a /L
if len(modes_energies) == 0 and number_of_modes is None:
    raise ValueError("If modes_energies is not provided, number_of_modes must be provided")
if len(modes_energies) == 0:
    modes_energies = [np.pi * C * (alpha + 1) / cavity_length for alpha in range(number_of_modes)]
# g_a = sqrt(omega_a / L) * sin(a * pi / 2)
lm_couplings: List[np.float64] = \
    [50*np.sqrt(omega / cavity_length) * np.sin((2*alpha + 1) * np.pi / 2) if alpha % 2 == 0 else 0
        for alpha, omega in enumerate(modes_energies)]

# Remove modes with zero coupling
valid_modes = np.where(~np.isclose(np.array(lm_couplings), 0))[0]
number_of_modes: int = len(valid_modes)
modes_energies = [modes_energies[mode] for mode in valid_modes]
lm_couplings = [lm_couplings[mode] for mode in valid_modes]

# Print the parameters
io.message_output("Parameters:\n", "output")
io.message_output(f"Electron eigenvalues: {parsed_input_file["elec_energies"]}\n", "output")
for i in range(number_of_modes):
    io.message_output(
        f"Photon mode {2*i+1}: Energy: {modes_energies[i]} H.a.; LM coupling: {lm_couplings[i]}\n",
        "output")
io.message_output("\n", "output")

# NOW, COMPUTE USING THE UTILS
# 1. GET QED HAMILTONIAN
h_el, h_ph, h_int, h_qed = utils.get_h_qed_plane_waves(
    parsed_input_file["elec_energies"], modes_energies, lm_couplings)
# 2. DEFINE THE OPERATORS to be measured
observables: dict[str, MixedOp] = {}
if "energy" in parsed_input_file["observables"]:
    observables["total_energy"] = h_qed
    observables["electron_energy"] = MixedOp({("F"): [(1.0, h_el)]})
    observables["photon_energy"] = MixedOp({("B"): [(1.0, h_ph)]})
    observables["interaction_energy"] = h_int
if "particle_number" in parsed_input_file["observables"]:
    # Electron number in excited state
    observables["electron_state_1_occupation"] = MixedOp(
        {("F"): [(1.0, FermionicOp({"+_1 -_1": 1},
                                   num_spin_orbitals=len(parsed_input_file["elec_energies"])))]}
    )
    for i in range(number_of_modes):
        # Photon number in mode i
        observables[f"photon_mode_{i}_occupation"] = MixedOp(
            {("B"): [(1.0, BosonicOp({f"+_{i} -_{i}": 1}, num_modes=number_of_modes))]}
        )
if "ph_correlation" in parsed_input_file["observables"]:
    # Photon correlation between modes i and j
    # https://www.pnas.org/doi/full/10.1073/pnas.1615509114#sec-5, E field operator
    for i in range(number_of_modes):
        for j in range(i, number_of_modes):
            # First, we define the operators
            op1 = BosonicOp({f"+_{i} +_{j}": 1}, num_modes=number_of_modes)
            op2 = BosonicOp({f"+_{i} -_{j}": 1}, num_modes=number_of_modes)
            op3 = BosonicOp({f"-_{i} +_{j}": 1}, num_modes=number_of_modes)
            op4 = BosonicOp({f"-_{i} -_{j}": 1}, num_modes=number_of_modes)
            prefactor = 0.5 * np.sqrt(1 / (modes_energies[i] * modes_energies[j]))
            # Then, we put them all together
            observables[f"photon_correlation_{i}_{j}"] = MixedOp(
                {("B"): [(prefactor, op1), (prefactor, op2), (prefactor, op3), (prefactor, op4)]})
# 3. DEFINE THE MAPPERS
mixed_papper = \
    utils.get_mapper(number_of_modes, parsed_input_file["number_of_fock_states"])
# 4. MAP THE HAMILTONIAN AND OBSERVABLES
hqed_mapped = mixed_papper.map(h_qed)
observables_mapped: dict[str, SparsePauliOp] = {}
for key, value in observables.items():
    observables_mapped[key] = mixed_papper.map(value)
# 5. DEFINE THE INITIAL STATE: The matter is in the excited state, the photons in the vacuum state
init_state: dict[int, tuple[complex | np.complex128, complex | np.complex128]] = {}
for n in range(number_of_modes):
    init_state[n] = (np.complex128(1), np.complex128(0))
# Add matter part
init_state[number_of_modes] = (np.complex128(1), np.complex128(0))
init_state[number_of_modes + 1] = (np.complex128(0), np.complex128(1))
# 6. DEFINE THE HARDWARE
if parsed_input_file["hardware"] == "generic_simulator":
    backend = AerSimulator.from_backend(GenericBackendV2(num_qubits=hqed_mapped.num_qubits))
    estimator = Estimator(options=dict(run_options={"shots": None}))
else:
    try:
        hardware: str = parsed_input_file["hardware"].split("noisy_simulator_")[1] \
            if "noisy_simulator" in parsed_input_file["hardware"] else parsed_input_file["hardware"]
        # First, get the token
        token: str | None = None
        with open("ibm_token", "r", encoding="UTF-8") as f:
            token = f.readline().split("\n")[0]
            f.close()
        # Then, get the available backends
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        service.backends(simulator=False)
        # Finally, pick the selected the backend
        backend = service.backend(hardware)
        estimator = Estimator(options={"shots": None})  # TODO: use BackendEstimator
        # Initialize the noise model from the selected hardware
        if "noisy_simulator" in hardware:
            noise_model: NoiseModel = NoiseModel.from_backend(backend)
            backend: AerSimulator = AerSimulator.from_backend(backend)
            estimator = Estimator(options={
                "shots": parsed_input_file["shots"], "noise_model": noise_model})
        io.message_output(f"Backend: {hardware}. Num qubits = {backend.num_qubits}\n", "output")
    except ValueError as e:
        backend = GenericBackendV2(num_qubits=hqed_mapped.num_qubits)
        io.message_output(f"Error: {e}. Using generic BE instead\n", "output")
# 7. Time evolve
io.message_output(f"Starting time evolution with delta_t = {parsed_input_file["delta_t"]} and " +
    f"final_time = {parsed_input_file["final_time"]}\n", "output")
start_time = time.time()
result: utils.TimeEvolutionResult = utils.custom_time_evolve(
    hqed_mapped,
    observables_mapped,
    init_state,
    parsed_input_file["time_evolution_strategy"],
    parsed_input_file["time_evolution_synthesis"],
    parsed_input_file["optimization_level"],
    backend,
    estimator,
    parsed_input_file["final_time"],
    parsed_input_file["delta_t"]
)
io.message_output(f"Time elapsed: {time.time() - start_time}s", "output")
