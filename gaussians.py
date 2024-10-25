import sys
import time
import os
from typing import List, Literal
import numpy as np
import matplotlib.pyplot as plt

from qiskit_nature.second_q.operators import BosonicOp, FermionicOp, MixedOp
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.primitives import Estimator

sys.path.append('./')
import spontaneous_emission_utils as utils

C = 137.03599 # Speed of light in atomic units

# Define the parameters
# 1D H atom with soft coulomb (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.063819)
electron_eigenvalues: List[float] = [-0.6738, -0.2798]
modes_energies = []
number_of_modes: int | None = None
number_of_gaussians: int | None = None
# Nearest neighbor, 2nd nearest neighbor, 3rd nearest neighbor
interaction_type: Literal['nn', '2nn', '3nn'] = 'nn'
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

def visualize_matrix(M, type_to_plot: Literal["abs", "real"]='abs', log=False):
    if type_to_plot == 'abs':
        M_p = np.abs(M)
    else:
        M_p = np.real(M)

    fig = plt.figure(1)
    ax1 = fig.gca()
    if log:
        cp = ax1.matshow(np.log(M_p), interpolation='none')  # cmap='Reds')
    else:
        cp = ax1.matshow(M_p, interpolation='none', cmap='Reds')
    cbar = fig.colorbar(cp, ax=ax1)
    cbar.set_label('M.'+type_to_plot, rotation=270, labelpad=17)
    plt.show()

# Define the Gaussian function
def gaussian(x, x0, sigma0, normalized=True):
    """Single Gaussian function."""
    g = np.exp(-((x - x0) ** 2) / (2 * sigma0 ** 2))
    return g / np.linalg.norm(g) if normalized else g

# Define the cavity mode to approximate (e.g., standing wave: cos(kx) or sin(kx))
def plane_wave(x, k, normalized=True):
    """Plane wave function (standing wave in a cavity)."""
    if k % 2 == 0:
        pn = modes_energies[k] * np.cos(((k + 1) * np.pi) / cavity_length * x)
    else:
        pn = modes_energies[k] * np.sin(((k + 1) * np.pi) / cavity_length * x)
    return pn / np.linalg.norm(pn) if normalized else pn

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
        elif value[0].replace(' ', '') == "modes_energies":
            modes_energies: List[float] = []
            for mode_energy in value[1].split(';'):
                modes_energies.append(float(mode_energy.replace(' ', '')))
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
if len(modes_energies) == 0 and number_of_modes is None:
    raise ValueError("If modes_energies is not provided, number_of_modes must be provided")
if len(modes_energies) == 0:
    modes_energies = [np.pi * C * (alpha + 1) / cavity_length for alpha in range(number_of_modes)]
number_of_modes: int = len(modes_energies)
lm_couplings: List[np.float64] = \
    [40*np.sqrt(omega / cavity_length) * np.sin((2*alpha + 1) * np.pi / 2) if alpha % 2 == 0 else 0
        for alpha, omega in enumerate(modes_energies)]
# Define data for the gaussians
x_data = np.arange(-cavity_length/2, cavity_length/2, 0.1)
sigma = cavity_length / (4*number_of_gaussians)
mu = list(
    np.linspace(-cavity_length/2 + 2.5*sigma, cavity_length/2 - 2.5*sigma, number_of_gaussians))

# Print the parameters
utils.message_output("Parameters:\n", "output")
utils.message_output(f"Electron eigenvalues: {electron_eigenvalues}\n", "output")
for i in range(number_of_modes):
    utils.message_output(
        f"Photon mode {i + 1}: Energy: {modes_energies[i]} H.a.; LM coupling: {lm_couplings[i]}\n",
        "output")
utils.message_output("\n", "output")

# Plot the Gaussians
if False:
    plt.figure(figsize=(10, 6))
    for i in range(number_of_gaussians):
        plt.plot(x_data, gaussian(x_data, mu[i], sigma), label=f"Gaussian {i+1}")
    plt.legend()
    plt.grid(True)
    plt.show()

if False:
    plt.figure(figsize=(10, 6))
    for i in range(1,number_of_modes,2):
        plt.plot(x_data, plane_wave(x_data, i), label=f"Plane wave {i+1}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Define the overlap integrals bvetween a plane wave and a Gaussian
projections = np.zeros((number_of_modes, number_of_gaussians))
for i in range(number_of_modes):
    plane_w = plane_wave(x_data, i, normalized=True)
    for j in range(number_of_gaussians):
        gauss = gaussian(x_data, mu[j], sigma, normalized=True)
        # Fit the plane wave to the Gaussian
        projections[i, j] = np.dot(plane_w, gauss)

# Define the combined coefficients that will appear in the Hamiltonian. E.g. the element (i, j) is
# the sum of all the coeffience of the operators b^dagger_i b_j that appear in the Hamiltonian.
# This corresponds to: gaussian_coeffs[i, j] = sum_{k=0}^{n_plane_waves} coeffs[k, i] * coeffs[k, j]
overlap_tensor = np.zeros((number_of_gaussians, number_of_gaussians))
uncoupled_photon_h_tensor = np.zeros((number_of_gaussians, number_of_gaussians))
bilinear_coupling_tensor = np.zeros((number_of_gaussians))
for i in range(number_of_gaussians):
    bilinear_coupling_tensor[i] = np.sum(lm_couplings * projections[:, i])
    gauss_i = gaussian(x_data, mu[i], sigma, normalized=True)
    for j in range(number_of_gaussians):
        uncoupled_photon_h_tensor[i, j] =\
                np.sum(modes_energies * projections[:, i] * projections[:, j])
        gauss_j = gaussian(x_data, mu[j], sigma, normalized=True)
        overlap_tensor[i, j] = np.dot(gauss_i, gauss_j)

h_el, h_ph, h_int, h_qed = utils.get_h_qed_gauss_localized_basis(
    electron_eigenvalues,
    number_of_gaussians,
    overlap_tensor,
    uncoupled_photon_h_tensor,
    bilinear_coupling_tensor,
    interaction_type,
    bilinear_threshold)
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
    neighbors = range(-1, 2) if interaction_type == 'nn' else \
        range(-2, 3) if interaction_type == '2nn' else range(-3, 4)
    # Expand the particle operator for each plane wave in the new basis
    for n in range(number_of_modes):
        # Photon number in mode i
        ph_num = BosonicOp({})
        for i in range(number_of_gaussians):
            for j in neighbors:
                if i + j >= 0 and i + j < number_of_gaussians:
                    ph_num += BosonicOp({
                        f'+_{i} -_{i+j}': np.conj(projections[n, i]) * projections[n, i+j]
                        }, num_modes=number_of_gaussians)
        observables.append(MixedOp({("B"): [(1.0, ph_num)]}))
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
if hardware == "generic_simulator":
    backend = GenericBackendV2(num_qubits=hqed_mapped.num_qubits)
else:
    try:
        # First, get the token
        token: str | None = None
        with open("ibm_token", "r", encoding="UTF-8") as f:
            token = f.readline().split("\n")[0]
            f.close()
        # Then, get the available backends
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        service.backends(simulator=False)
        # Finally, pick the select the backend
        backend = service.backend(hardware)
        utils.message_output(f"Backend: {hardware}. Num qubits = {backend.num_qubits}\n", "output")
    except ValueError as e:
        backend = GenericBackendV2(num_qubits=hqed_mapped.num_qubits)
        utils.message_output(f"Error: {e}. Using generic BE instead\n", "output")
# 7. Time evolve
utils.message_output(
    f"Starting time evolution with delta_t = {delta_t} and final_time = {final_time}\n", "output")
start_time = time.time()
estimator = Estimator()
estimator.set_options(shots=None)
result: utils.TimeEvolutionResult = utils.custom_time_evolve(
    hqed_mapped, observables_mapped, init_state, time_evolution_strategy,
    time_evolution_synthesis, optimization_level, backend, estimator, final_time, delta_t)
utils.message_output(f"Time elapsed: {time.time() - start_time}s", "output")
