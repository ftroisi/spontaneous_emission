import sys
import os
import math
import time
import numpy as np
from qiskit_nature.second_q.operators import FermionicOp, BosonicOp, MixedOp
from qiskit.primitives import Estimator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector
sys.path.append('./')
import spontaneous_emission_utils as utils

C = 137.03599 # Speed of light in atomic units

# Define the parameters
electron_eigenvalues = [-0.6738, -0.2798] # 1D H atom with soft coulomb (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.063819)
photon_energies = []
number_of_modes = None
cavity_length = 1.0
number_of_fock_states = 1
initial_state: str | None = None
# Time evolution parameters
delta_t = 0.1
final_time = 4.0
# Optimization
optimization_level = 3
time_evolution_strategy: str = "tct"
time_evolution_synthesis: str = "lie_trotter"
# Observables
observables_requested: list[str] = ["energy", "particle_number", "ph_correlation"]

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/circuits"):
    os.makedirs("results/circuits")

# Read input file
with open('input', 'r', encoding="UTF-8") as f:
    lines: list[str] = f.readlines()
    for i, line in enumerate(lines):
        if line[0] == '#':
            continue
        value: list[str] = line.split("\n")[0].split('=')
        # Assign values
        # System parameters
        if value[0].replace(' ', '') == "elec_energies":
            electron_eigenvalues: list[float] = []
            for electron_eigenvalue in value[1].split(';'):
                electron_eigenvalues.append(float(electron_eigenvalue.replace(' ', '')))
        elif value[0].replace(' ', '') == "photon_energies":
            photon_energies: list[float] = []
            for photon_energy in value[1].split(';'):
                photon_energies.append(float(photon_energy.replace(' ', '')))
        elif value[0].replace(' ', '') == "cavity_length":
            cavity_length = float(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "delta_t":
            delta_t = float(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "final_time":
            final_time = float(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "number_of_fock_states":
            number_of_fock_states = int(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "initial_state":
            initial_state: str = value[1].replace(' ', '')
        elif value[0].replace(' ', '') == "optimization_level":
            optimization_level = int(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "time_evolution_strategy":
            time_evolution_strategy: str = value[1].replace(' ', '')
        elif value[0].replace(' ', '') == "time_evolution_synthesis":
            time_evolution_synthesis: str = value[1].replace(' ', '')
        elif value[0].replace(' ', '') == "observables":
            observables_requested: list[float] = []
            for obs in value[1].split(';'):
                observables_requested.append(obs.replace(' ', ''))
    f.close()

# Define modes frequency and couplings
# Equation 6 of: https://www.pnas.org/doi/full/10.1073/pnas.1615509114#sec-5
# omega_a = pi * c * a /L
if len(photon_energies) == 0 and number_of_modes is None:
    raise ValueError("If photon_energies is not provided, number_of_modes must be provided")
if len(photon_energies) == 0:
    photon_energies = [np.pi * C * alpha / cavity_length for alpha in range(1, number_of_modes, 2)]
number_of_modes: int = len(photon_energies)
# g_a = sqrt(omega_a / L) * sin(a * pi / 2)
lm_couplings: list[np.float64] = \
    [np.sqrt(omega / cavity_length) * np.sin((2*alpha + 1) * np.pi / 2) for alpha, omega in enumerate(photon_energies)]

# NOW, COMPUTE USING THE UTILS
# 1. GET QED HAMILTONIAN
h_el, h_ph, h_int, h_qed = utils.get_h_qed(electron_eigenvalues, photon_energies, lm_couplings)
# 2. DEFINE THE OPERATORS to be measured
observables: list[MixedOp] = []
if "energy" in observables_requested:
    observables.append(h_qed) # Total energy
    observables.append(MixedOp({("F"): [(1.0, h_el)]})) # Electron energy
    observables.append(MixedOp({("B"): [(1.0, h_ph)]})) # Photon energy
    observables.append(h_int) # Interaction energy
if "particle_number" in observables_requested:
    observables.append(MixedOp({("F"): [
        (1.0, FermionicOp({"+_1 -_1": 1}, num_spin_orbitals=len(electron_eigenvalues)))]})) # Electron number in mode 1
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
if initial_state is not None:
    init_state = Statevector.from_label(initial_state)
else:
    init_state = Statevector.from_label(
        "10" + "0" * number_of_modes * math.ceil(np.log2(number_of_fock_states + 1)))
# 6. DEFINE THE HARDWARE
backend = GenericBackendV2(num_qubits=hqed_mapped.num_qubits)
# 7. Time evolve
utils.message_output(
    f"Starting time evolution with delta_t = {delta_t} and final_time = {final_time}\n", "output")
start_time = time.time()
result: utils.TimeEvolutionResult = utils.custom_time_evolve(
    hqed_mapped, observables_mapped, init_state, time_evolution_strategy,
    time_evolution_synthesis, optimization_level, backend, Estimator(), final_time, delta_t)
utils.message_output(f"Time elapsed: {time.time() - start_time}s", "output")
# 8. Save results
observables_result = np.array(np.array(result.observables)[:, :, 0])
np.savez("results/time_evolution", times=result.times, observables=observables_result)
