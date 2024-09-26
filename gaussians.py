import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Literal

from qiskit_nature.second_q.operators import BosonicOp, FermionicOp, MixedOp
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.primitives import Estimator

sys.path.append('./')
import spontaneous_emission_utils as utils

C = 137.03599 # Speed of light in atomic units

# Define the parameters
# 1D H atom with soft coulomb (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.063819)
electron_eigenvalues: List[float] = [-0.6738, -0.2798]
photon_energies = []
number_of_modes: int | None = None
number_of_gaussians: int | None = None
gaussian_interaction_type: Literal['nn', '2nn', '3nn'] = 'nn' # Nearest neighbor, 2nd nearest neighbor, 3rd nearest neighbor
bilinear_threshold: float = 0.1 # 10% of the bilinear element
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

def visualize_matrix(M, type_to_plot='abs', diag=True, log=False):
    if type_to_plot == 'abs':
        M_p = np.abs(M)
    elif type_to_plot == 'real':
        M_p = np.real(M)
    elif type_to_plot == 'imag':
        M_p = np.real(M)

    if diag is False:
        M_p -= np.diag(np.diag(M_p))

    fig = plt.figure(1)
    ax1 = fig.gca()
    if log:
        cp = ax1.matshow(np.log(M_p), interpolation='none')  # cmap='Reds')
    else:
        cp = ax1.matshow(M_p, interpolation='none', cmap='Reds')
    cbar = fig.colorbar(cp, ax=ax1)
    cbar.set_label('M.'+type_to_plot, rotation=270, labelpad=17)
    plt.show()

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/circuits"):
    os.makedirs("results/circuits")

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
        elif value[0].replace(' ', '') == "number_of_gaussians":
            number_of_gaussians = int(value[1].replace(' ', ''))
        elif value[0].replace(' ', '') == "gaussian_interaction_type":
            gaussian_interaction_type = value[1].replace(' ', '')
        elif value[0].replace(' ', '') == "bilinear_threshold":
            bilinear_threshold = float(value[1].replace(' ', ''))
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
if number_of_gaussians is None:
    raise ValueError("number_of_gaussians must be provided")
if len(photon_energies) == 0 and number_of_modes is None:
    raise ValueError("If photon_energies is not provided, number_of_modes must be provided")
if len(photon_energies) == 0:
    photon_energies = [np.pi * C * (alpha + 1) / cavity_length for alpha in range(number_of_modes)]
number_of_modes: int = len(photon_energies)
# Define data for the gaussians
x_data = np.arange(-cavity_length/2, cavity_length/2, 0.1)
sigma = cavity_length / (3.5*number_of_gaussians)
mu = list(np.linspace(-cavity_length/2 + 3*sigma, cavity_length/2 - 3*sigma, number_of_gaussians))

# Define the Gaussian function
def gaussian(x, x0, sigma0):
    """Single Gaussian function."""
    return np.exp(-((x - x0) ** 2) / (2 * sigma0 ** 2))

def get_lm_coupling(k):
    """Get the coupling for a given mode."""
    if k % 2 == 0:
        return 0
    return np.sqrt(photon_energies[k - 1] / cavity_length)

# Define the cavity mode to approximate (e.g., standing wave: cos(kx) or sin(kx))
def plane_wave(x, k):
    """Plane wave function (standing wave in a cavity)."""
    if k % 2 == 0:
        return photon_energies[k - 1] * np.sin((k*np.pi)/cavity_length * x)  # photon_energies[k - 1] * np.sin((k*np.pi)/cavity_length * x)
    return photon_energies[k - 1] * np.cos((k*np.pi)/cavity_length * x)

# Plot the Gaussians
if False:
    plt.figure(figsize=(10, 6))
    for i in range(number_of_gaussians):
        plt.plot(x_data, gaussian(x_data, mu[i], sigma), label=f"Gaussian {i+1}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Define the overlap integrals bvetween a plane wave and a Gaussian
coeffs = np.zeros((number_of_modes, number_of_gaussians))
for i in range(number_of_modes):
    plane_w = plane_wave(x_data, i + 1)
    normalized_pn = plane_w / np.linalg.norm(plane_w)
    for j in range(number_of_gaussians):
        gauss = gaussian(x_data, mu[j], sigma)
        normalized_gauss = gauss / np.linalg.norm(gauss)
        # Fit the plane wave to the Gaussian
        coeffs[i, j] = np.dot(normalized_pn, normalized_gauss)

# Define the combined coefficients that will appear in the Hamiltonian. For instance, the element (i, j) will be
# the sum of all the coeffience of the operators b^dagger_i b_j that appear in the Hamiltonian.
# This corresponds to: gaussian_coeffs[i, j] = sum_{k=0}^{n_plane_waves} coeffs[k, i] * coeffs[k, j]
gaussian_diag_coeffs = np.zeros((number_of_gaussians, number_of_gaussians))
gaussian_bilinear_coeffs = np.zeros((number_of_gaussians))
for i in range(number_of_gaussians):
    couplings = [get_lm_coupling(k + 1) for k in range(number_of_modes)]
    gaussian_bilinear_coeffs[i] = np.sum(couplings * coeffs[:, i])
    for j in range(number_of_gaussians):
        gaussian_diag_coeffs[i, j] = np.sum(photon_energies * coeffs[:, i] * coeffs[:, j])

h_el, h_ph, h_int, h_qed = utils.get_h_qed_gauss(
    electron_eigenvalues, number_of_gaussians, gaussian_diag_coeffs, gaussian_bilinear_coeffs, gaussian_interaction_type, bilinear_threshold)
utils.message_output(str(h_qed), "output")
# 2. DEFINE THE OPERATORS to be measured
observables: List[MixedOp] = []
if "energy" in observables_requested:
    observables.append(h_qed) # Total energy
    observables.append(MixedOp({("F"): [(1.0, h_el)]})) # Electron energy
    observables.append(MixedOp({("B"): [(1.0, h_ph)]})) # Photon energy
    observables.append(h_int) # Interaction energy
if "particle_number" in observables_requested:
    observables.append(MixedOp({("F"): [
        # Electron number in mode 1
        (1.0, FermionicOp({"+_1 -_1": 1}, num_spin_orbitals=len(electron_eigenvalues)))]}))
    # Recompute the gaussian diagonal coeffs (without the frequency)
    gaussian_diag_coeffs = np.zeros((number_of_gaussians, number_of_gaussians))
    neighbors = range(-1, 2) if gaussian_interaction_type == 'nn' else \
        range(-2, 3) if gaussian_interaction_type == '2nn' else range(-3, 4)
    for i in range(number_of_gaussians):
        for j in neighbors:
            if i + j >= 0 and i + j < number_of_gaussians:
                gaussian_diag_coeffs[i, i+j] = np.sum(coeffs[:, i] * coeffs[:, i+j])
    # Expand the particle operator for each plane wave in the new basis
    for i in range(number_of_modes):
        # Photon number in mode i
        ph_num = BosonicOp({})
        for i in range(number_of_gaussians):
            for j in neighbors:
                if i + j >= 0 and i + j < number_of_gaussians:
                    ph_num += BosonicOp({f'+_{i} -_{i+j}': gaussian_diag_coeffs[i, i+j]}, num_modes=number_of_gaussians)
        observables.append(MixedOp({("B"): [(1.0, ph_num)]}))
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
mixed_papper = utils.get_mapper(number_of_gaussians, number_of_fock_states)
# 4. MAP THE HAMILTONIAN AND OBSERVABLES
hqed_mapped = mixed_papper.map(h_qed)
observables_mapped = [mixed_papper.map(op) for op in observables]
# 5. DEFINE THE INITIAL STATE: The matter is in the excited state, the photons in the vacuum state
init_state: dict[int, tuple[complex | np.complex128, complex | np.complex128]] = {}
for n in range(number_of_gaussians):
    init_state[n] = (np.complex128(1), np.complex128(0))
# Add matter part
init_state[number_of_gaussians] = (np.complex128(1), np.complex128(0))
init_state[number_of_gaussians + 1] = (np.complex128(0), np.complex128(1))
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
