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
photon_energies = []
number_of_modes: int | None = None
number_of_basis_functions: int | None = None
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
        elif value[0].replace(' ', '') == "number_of_basis_functions":
            number_of_basis_functions = int(value[1].replace(' ', ''))
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
if number_of_basis_functions is None:
    raise ValueError("number_of_basis_functions must be provided")
if len(photon_energies) == 0 and number_of_modes is None:
    raise ValueError("If photon_energies is not provided, number_of_modes must be provided")
if len(photon_energies) == 0:
    photon_energies = [np.pi * C * (alpha + 1) / cavity_length for alpha in range(number_of_modes)]
number_of_modes: int = len(photon_energies)
lm_couplings: List[np.float64] = \
    [40*np.sqrt(omega / cavity_length) * np.sin((2*alpha + 1) * np.pi / 2) if alpha % 2 == 0 else 0
        for alpha, omega in enumerate(photon_energies)]
# Define data for the gaussians
x_data = np.arange(-cavity_length/2, cavity_length/2, 0.1)
coeff = 2*number_of_basis_functions / cavity_length
mu = list(np.linspace(-cavity_length/2 + 1/coeff, cavity_length/2 - 1/coeff, number_of_basis_functions))
print(coeff)
print(mu)

# Define the Gaussian function
def triangular_basis(x, x0, m):
    """Single Gaussian function."""
    y = 1 - m*np.abs(x - x0)
    y[y < 0] = 0
    return y

# Define the cavity mode to approximate (e.g., standing wave: cos(kx) or sin(kx))
def plane_wave(x, k):
    """Plane wave function (standing wave in a cavity)."""
    if k % 2 == 0:
        return photon_energies[k] * np.cos(((k + 1) * np.pi) / cavity_length * x)
    return photon_energies[k] * np.sin(((k + 1) * np.pi) / cavity_length * x)

# Print the parameters
utils.message_output("Parameters:\n", "output")
utils.message_output(f"Electron eigenvalues: {electron_eigenvalues}\n", "output")
for i in range(number_of_modes):
    utils.message_output(
        f"Photon mode {i + 1}: Energy: {photon_energies[i]} H.a.; LM coupling: {lm_couplings[i]}\n",
        "output")
utils.message_output("\n", "output")

# Plot the Basis functions
if False:
    plt.figure(figsize=(10, 6))
    for i in range(number_of_basis_functions):
        plt.plot(x_data, triangular_basis(x_data, mu[i], coeff), label=f"Gaussian {i+1}")
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

# Define the overlap integrals between a plane wave and a Gaussian
coeffs = np.zeros((number_of_modes, number_of_basis_functions))
for i in range(number_of_modes):
    plane_w = plane_wave(x_data, i)
    normalized_pn = plane_w / np.linalg.norm(plane_w)
    for j in range(number_of_basis_functions):
        gauss = triangular_basis(x_data, mu[j], coeff)
        normalized_gauss = gauss / np.linalg.norm(gauss)
        # Fit the plane wave to the Gaussian
        coeffs[i, j] = np.dot(normalized_pn, normalized_gauss)

# Define the combined coefficients that will appear in the Hamiltonian. For instance, the element (i, j) will be
# the sum of all the coeffience of the operators b^dagger_i b_j that appear in the Hamiltonian.
# This corresponds to: gaussian_coeffs[i, j] = sum_{k=0}^{n_plane_waves} coeffs[k, i] * coeffs[k, j]
gaussian_diag_coeffs = np.zeros((number_of_basis_functions, number_of_basis_functions))
gaussian_bilinear_coeffs = np.zeros((number_of_basis_functions))
for i in range(number_of_basis_functions):
    gaussian_bilinear_coeffs[i] = np.sum(lm_couplings * coeffs[:, i])
    for j in range(number_of_basis_functions):
        gaussian_diag_coeffs[i, j] = np.sum(photon_energies * coeffs[:, i] * coeffs[:, j])

visualize_matrix(gaussian_diag_coeffs, type_to_plot='abs')
plt.plot(gaussian_bilinear_coeffs)
plt.show()

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
    neighbors = range(-1, 2) if gaussian_interaction_type == 'nn' else \
        range(-2, 3) if gaussian_interaction_type == '2nn' else range(-3, 4)
    # Expand the particle operator for each plane wave in the new basis
    for n in range(number_of_modes):
        # Photon number in mode i
        ph_num = BosonicOp({})
        for i in range(number_of_gaussians):
            for j in neighbors:
                if i + j >= 0 and i + j < number_of_gaussians:
                    ph_num += BosonicOp({f'+_{i} -_{i+j}': np.conj(coeffs[n, i]) * coeffs[n, i+j]},
                                        num_modes=number_of_gaussians)
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
