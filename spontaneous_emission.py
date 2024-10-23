import math
import os
import sys
import time
from typing import List

import numpy as np
from qiskit.primitives import BackendEstimator, Estimator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_nature.second_q.operators import BosonicOp, FermionicOp, MixedOp
IBM_RUNTIME_AVAILABLE = True
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError:
    IBM_RUNTIME_AVAILABLE = False
IQM_RUNTIME_AVAILABLE = True
try:
    from mqp.qiskit_provider import MQPProvider
except ImportError:
    IQM_RUNTIME_AVAILABLE = False
sys.path.append('./')
import spontaneous_emission_utils as utils

C = 137.03599 # Speed of light in atomic units

# Define the parameters
# 1D H atom with soft coulomb (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.063819)
electron_eigenvalues: List[float] = [-0.6738, -0.2798]
photon_energies = []
number_of_modes: int | None = None
cavity_length: float = 1.0
number_of_fock_states: int = 1
init_state: dict[int, tuple[complex | np.complex128, complex | np.complex128]] | None = None
# Time evolution parameters
delta_t: float = 0.1
final_time: float = 4.0
# Optimization
hardware: str = "generic_simulator"
optimization_level: int = 3
time_evolution_strategy: str = "tct"
time_evolution_synthesis: str = "lie_trotter"
# Observables
observables_requested: List[str] = ["energy", "particle_number", "ph_correlation"]

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/circuits"):
    os.makedirs("results/circuits")

# Read input file
with open('input', 'r', encoding="UTF-8") as f:
    lines: List[str] = f.readlines()
    for i, line in enumerate(lines):
        if line[0] == '#':
            continue
        value: List[str] = line.split("\n")[0].split('=')
        # Assign values
        # System parameters
        if value[0].replace(' ', '') == "elec_energies":
            electron_eigenvalues: List[float] = []
            for electron_eigenvalue in value[1].split(';'):
                electron_eigenvalues.append(float(electron_eigenvalue.replace(' ', '')))
        elif value[0].replace(' ', '') == "photon_energies":
            photon_energies: List[float] = []
            for photon_energy in value[1].split(';'):
                photon_energies.append(float(photon_energy.replace(' ', '')))
        elif value[0].replace(' ', '') == "cavity_length":
            cavity_length = float(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "number_of_fock_states":
            number_of_fock_states = int(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "number_of_modes":
            number_of_modes = int(value[1].replace(' ', ''))
        # Time evolution parameters
        elif value[0].replace(' ', '') == "delta_t":
            delta_t = float(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "final_time":
            final_time = float(value[1].replace(' ', ''))
        # Optimization
        elif value[0].replace(' ', '') == "optimization_level":
            optimization_level = int(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "hardware":
            hardware: str = value[1].replace(' ', '')
        elif value[0].replace(' ', '') == "time_evolution_strategy":
            time_evolution_strategy: str = value[1].replace(' ', '')
        elif value[0].replace(' ', '') == "time_evolution_synthesis":
            time_evolution_synthesis: str = value[1].replace(' ', '')
        # Observables
        elif value[0].replace(' ', '') == "observables":
            observables_requested: List[float] = []
            for obs in value[1].split(';'):
                observables_requested.append(obs.replace(' ', ''))
    f.close()

# Define modes frequency and couplings
# Equation 6 of: https://www.pnas.org/doi/full/10.1073/pnas.1615509114#sec-5
# omega_a = pi * c * a /L
if len(photon_energies) == 0 and number_of_modes is None:
    raise ValueError("If photon_energies is not provided, number_of_modes must be provided")
if len(photon_energies) == 0:
    photon_energies = [np.pi * C * (alpha + 1) / cavity_length for alpha in range(number_of_modes)]
# g_a = sqrt(omega_a / L) * sin(a * pi / 2)
lm_couplings: List[np.float64] = \
    [40*np.sqrt(omega / cavity_length) * np.sin((2*alpha + 1) * np.pi / 2) if alpha % 2 == 0 else 0
        for alpha, omega in enumerate(photon_energies)]

# Remove modes with zero coupling
valid_modes = np.where(np.array(lm_couplings) != 0)[0]
number_of_modes: int = len(valid_modes)
photon_energies = [photon_energies[mode] for mode in valid_modes]
lm_couplings = [lm_couplings[mode] for mode in valid_modes]

# Print the parameters
utils.message_output("Parameters:\n", "output")
utils.message_output(f"Electron eigenvalues: {electron_eigenvalues}\n", "output")
for i in range(number_of_modes):
    utils.message_output(
        f"Photon mode {2*i+1}: Energy: {photon_energies[i]} H.a.; LM coupling: {lm_couplings[i]}\n",
        "output")
utils.message_output("\n", "output")

# NOW, COMPUTE USING THE UTILS
# 1. GET QED HAMILTONIAN
h_el, h_ph, h_int, h_qed = utils.get_h_qed(electron_eigenvalues, photon_energies, lm_couplings)
# 2. DEFINE THE OPERATORS to be measured
observables: List[MixedOp] = []
if "energy" in observables_requested:
    observables.append(h_qed) # Total energy
    observables.append(MixedOp({("F"): [(1.0, h_el)]})) # Electron energy
    observables.append(MixedOp({("B"): [(1.0, h_ph)]})) # Photon energy
    observables.append(h_int) # Interaction energy
if "particle_number" in observables_requested:
    observables.append(MixedOp({("F"): [
        # Electron number in excited state
        (1.0, FermionicOp({"+_1 -_1": 1}, num_spin_orbitals=len(electron_eigenvalues)))]}))
    for i in range(number_of_modes):
        # Photon number in mode i
        observables.append(
            MixedOp({("B"): [(1.0, BosonicOp({f"+_{i} -_{i}": 1}, num_modes=number_of_modes))]}))
if "ph_correlation" in observables_requested:
    # Photon correlation between modes i and j
    # https://www.pnas.org/doi/full/10.1073/pnas.1615509114#sec-5, E field operator
    for i in range(number_of_modes):
        for j in range(i, number_of_modes):
            # First, we define the operators
            op1 = BosonicOp({f"+_{i} +_{j}": 1}, num_modes=number_of_modes)
            op2 = BosonicOp({f"+_{i} -_{j}": 1}, num_modes=number_of_modes)
            op3 = BosonicOp({f"-_{i} +_{j}": 1}, num_modes=number_of_modes)
            op4 = BosonicOp({f"-_{i} -_{j}": 1}, num_modes=number_of_modes)
            prefactor = 0.5 * np.sqrt(1 / (photon_energies[i] * photon_energies[j]))
            # Then, we put them all together
            observables.append(MixedOp(
                {("B"): [(prefactor, op1), (prefactor, op2), (prefactor, op3), (prefactor, op4)]}))
# 3. DEFINE THE MAPPERS
mixed_papper = utils.get_mapper(number_of_modes, number_of_fock_states)
# 4. MAP THE HAMILTONIAN AND OBSERVABLES
hqed_mapped = mixed_papper.map(h_qed)
observables_mapped = [mixed_papper.map(op) for op in observables]
# 5. DEFINE THE INITIAL STATE: The matter is in the excited state, the photons in the vacuum state
init_state: dict[int, tuple[complex | np.complex128, complex | np.complex128]] = {}
for n in range(number_of_modes):
    init_state[n] = (np.complex128(1), np.complex128(0))
# Add matter part
init_state[number_of_modes] = (np.complex128(1), np.complex128(0))
init_state[number_of_modes + 1] = (np.complex128(0), np.complex128(1))
# 6. DEFINE THE HARDWARE
if hardware == "generic_simulator":
    backend = GenericBackendV2(num_qubits=hqed_mapped.num_qubits)
else:
    try:
        token: str | None = None
        if "ibm" in hardware:
            # First, get the token
            with open("ibm_token", "r", encoding="UTF-8") as f:
                token = f.readline().split("\n")[0]
                f.close()
            # Then, get the available backends
            service = QiskitRuntimeService(channel="ibm_quantum", token=token)
            service.backends(simulator=False)
            # Finally, pick the select the backend
            backend = service.backend(hardware)
        elif hardware == "QExa20":
            # First, get the token
            with open("lrz_token", "r", encoding="UTF-8") as f:
                token = f.readline().split("\n")[0]
                f.close()
            # Then, get the available backends
            provider = MQPProvider(token=token)
            # Finally, pick the select the backend
            backend = provider.backend(hardware)
    except ValueError as e:
        backend = GenericBackendV2(num_qubits=hqed_mapped.num_qubits)
        utils.message_output(f"Error: {e}. Using generic BE instead\n", "output")
utils.message_output(f"Backend: {hardware}. Num qubits = {backend.num_qubits}\n", "output")
# 7. Time evolve
utils.message_output(
    f"Starting time evolution with delta_t = {delta_t} and final_time = {final_time}\n", "output")
start_time = time.time()
estimator = Estimator() if hardware == "generic_simulator" else BackendEstimator(backend=backend)
estimator.set_options(shots=None if hardware == "generic_simulator" else 1e4)
result: utils.TimeEvolutionResult = utils.custom_time_evolve(
    hqed_mapped, observables_mapped, init_state, time_evolution_strategy,
    time_evolution_synthesis, optimization_level, backend, estimator, final_time, delta_t)
utils.message_output(f"Time elapsed: {time.time() - start_time}s", "output")
