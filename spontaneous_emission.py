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

# Define the parameters
electron_eigenvalues = [0.0, 1.0]
photon_energies = [1.]
number_of_fock_states = 1
initial_state: str | None = None
# Time evolution parameters
delta_t = 0.1
final_time = 4.0

if not os.path.exists("results"):
    os.makedirs("results")

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
        elif value[0].replace(' ', '') == "delta_t":
            delta_t = float(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "final_time":
            final_time = float(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "number_of_fock_states":
            number_of_fock_states = int(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "initial_state":
            initial_state: str = value[1].replace(' ', '')
    f.close()

number_of_modes = len(photon_energies)
lm_couplings = [1/np.sqrt(x) for x in photon_energies]

# NOW, COMPUTE USING THE UTILS
# 1. GET QED HAMILTONIAN
h_el, h_ph, h_int, h_qed = utils.get_h_qed(electron_eigenvalues, photon_energies, lm_couplings)
# 2. DEFINE THE OPERATORS to be measured
measurements: list[MixedOp] = [
    h_qed, # Total energy
    MixedOp({("F"): [(1.0, h_el)]}), # Electron energy
    MixedOp({("B"): [(1.0, h_ph)]}), # Photon energy
    h_int, # Interaction energy
    MixedOp({("F"): [
        (1.0, FermionicOp({"+_1 -_1": 1}, num_spin_orbitals=len(electron_eigenvalues)))]}) # Electron number in mode 1
]
for i in range(number_of_modes):
    # Photon number in mode i
    measurements.append(
        MixedOp({("B"): [(1.0, BosonicOp({f"+_{i} -_{i}": 1}, num_modes=number_of_modes))]}))
# 3. DEFINE THE MAPPERS
mixed_papper = utils.get_mapper(number_of_modes, number_of_fock_states)
# 4. MAP THE HAMILTONIAN AND OBSERVABLES
hqed_mapped = mixed_papper.map(h_qed)
observables_mapped = [mixed_papper.map(op) for op in measurements]
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
result = utils.custom_time_evolve(
    hqed_mapped, observables_mapped, init_state, 1, backend, Estimator(), final_time, delta_t)
utils.message_output(f"Time elapsed: {time.time() - start_time}s", "output")
# 8. Save results
observables_result = np.array(np.array(result[0].observables)[:, :, 0])
np.savez("results/time_evolution", times=result[0].times, observables=observables_result)
result[1].draw(output="mpl", filename="results/single_timestep_circuit")
