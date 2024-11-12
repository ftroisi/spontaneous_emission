import sys
import time
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from qiskit_nature.second_q.operators import BosonicOp, FermionicOp, MixedOp
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as Estimator

sys.path.append('./')
import spontaneous_emission_utils as utils
import io_tools as io

C = 137.03599 # Speed of light in atomic units

# Define the cardinal sine function
def cardinal_sine_basis(x, x0, normalized=True):
    """
    Cardinal sine function:
    https://math.stackexchange.com/questions/2175638/orthogonality-of-periodic-sinc-function
    """
    sinc = np.sinc(np.pi * (x - x0)) * np.exp(1j * 2*np.pi*x)
    return sinc / np.linalg.norm(sinc) if normalized else sinc

# Define the Gaussian function
def gaussian(x, x0, sigma0, normalized=True):
    """Single Gaussian function."""
    g = np.exp(-((x - x0) ** 2) / (2 * sigma0 ** 2))
    return g / np.linalg.norm(g) if normalized else g

# Define the triangular function
def triangular_basis(x, x0, m, normalized=True):
    """Triangular function."""
    y = 1 - m*np.abs(x - x0)
    y[y < 0] = 0
    return y / np.linalg.norm(y) if normalized else y

# Define the cavity mode to approximate (e.g., standing wave: cos(kx) or sin(kx))
def plane_wave(x, omega, k, normalized=True):
    """Plane wave function (standing wave in a cavity)."""
    if k % 2 == 0:
        pn = omega[k] * np.cos(((k + 1) * np.pi) / cavity_length * x)
    else:
        pn = omega[k] * np.sin(((k + 1) * np.pi) / cavity_length * x)
    return pn / np.linalg.norm(pn) if normalized else pn

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/circuits"):
    os.makedirs("results/circuits")

# Read input file
parsed_input_file = dict(io.read_input_file("input"))
# Define some useful variables
number_of_modes: int | None = parsed_input_file.get("number_of_modes", None)
modes_energies: List[float] = parsed_input_file.get("modes_energies", [])
number_of_basis_functions: int | None = parsed_input_file.get("number_of_basis_functions", None)
cavity_length: float = parsed_input_file.get("cavity_length", 1.0)

# Define modes frequency and couplings
# Equation 6 of: https://www.pnas.org/doi/full/10.1073/pnas.1615509114#sec-5
# omega_a = pi * c * a /L
if parsed_input_file.get("number_of_basis_functions", None) is None:
    raise ValueError("number_of_basis_functions must be provided")
if len(modes_energies) == 0 and number_of_modes is None:
    raise ValueError("If modes_energies is not provided, number_of_modes must be provided")
if len(modes_energies) == 0:
    modes_energies = [np.pi * C * (alpha + 1) / cavity_length for alpha in range(number_of_modes)]
number_of_modes: int = len(modes_energies)
lm_couplings: List[np.float64] = \
    [40*np.sqrt(omega / cavity_length) * np.sin((2*alpha + 1) * np.pi / 2) if alpha % 2 == 0 else 0
        for alpha, omega in enumerate(modes_energies)]
# Define data for the the basis functions
x_data = np.arange(-cavity_length/2, cavity_length/2, 0.1)
angular_coeff = 2*number_of_basis_functions / cavity_length
centers = list(np.linspace(
    -cavity_length/2 + 1/angular_coeff,
    cavity_length/2 - 1/angular_coeff,
    number_of_basis_functions))
#centers = list(np.linspace(int(-cavity_length/2), int(cavity_length/2), number_of_basis_functions))

# Print the parameters
io.message_output("Parameters:\n", "output")
io.message_output(f"Electron eigenvalues: {parsed_input_file["elec_energies"]}\n", "output")
for i in range(number_of_modes):
    io.message_output(
        f"Photon mode {i + 1}: Energy: {modes_energies[i]} H.a.; LM coupling: {lm_couplings[i]}\n",
        "output")
io.message_output("\n", "output")

# Plot the Basis functions
if False:
    plt.figure(figsize=(10, 6))
    for i in range(number_of_basis_functions):
        func = triangular_basis(x_data, centers[i], angular_coeff)
        plt.plot(x_data, func, label=f"Basis Func {i+1}")
    plt.legend()
    plt.grid(True)
    plt.show()

if False:
    plt.figure(figsize=(10, 6))
    for i in range(1,number_of_modes,2):
        plt.plot(x_data, plane_wave(x_data, modes_energies, i), label=f"Plane wave {i+1}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Define the overlap integrals between a plane wave and a Gaussian
projections = np.zeros((number_of_modes, number_of_basis_functions))
for i in range(number_of_modes):
    plane_w = plane_wave(x_data, modes_energies, i, normalized=True)
    for j in range(number_of_basis_functions):
        new_basis_func = triangular_basis(x_data, centers[j], angular_coeff, normalized=True)
        # Fit the plane wave to the Gaussian
        projections[i, j] = np.dot(plane_w, new_basis_func)

# Define the combined coefficients that will appear in the Hamiltonian. E.g., the element (i, j) is
# the sum of all the coeffients of the operators b^dagger_i b_j that appear in the Hamiltonian.
# This is: uncoupled_photon_h_tensor[i, j] = sum_{k=0}^{n_plane_waves} coeffs[k, i] * coeffs[k, j]
overlap_tensor = np.zeros((number_of_basis_functions, number_of_basis_functions))
uncoupled_photon_h_tensor = np.zeros((number_of_basis_functions, number_of_basis_functions))
bilinear_coupling_tensor = np.zeros((number_of_basis_functions))
for i in range(number_of_basis_functions):
    bilinear_coupling_tensor[i] = np.sum(lm_couplings * projections[:, i])
    basis_i = triangular_basis(x_data, centers[i], angular_coeff, normalized=True)
    for j in range(number_of_basis_functions):
        uncoupled_photon_h_tensor[i, j] = \
            np.sum(modes_energies * projections[:, i] * projections[:, j])
        basis_j = triangular_basis(x_data, centers[j], angular_coeff, normalized=True)
        overlap_tensor[i, j] = np.dot(basis_i, basis_j)

h_el, h_ph, h_int, h_qed = utils.get_h_qed_localized_basis(
    parsed_input_file["elec_energies"],
    number_of_basis_functions,
    overlap_tensor=overlap_tensor,
    uncoupled_photon_h_tensor=uncoupled_photon_h_tensor,
    bilinear_coupling_tensor=bilinear_coupling_tensor,
    interaction_type=parsed_input_file["local_basis_interaction_type"],
    bilinear_threshold=parsed_input_file["bilinear_threshold"])
io.message_output(str(h_qed), "output")
# 2. DEFINE THE OPERATORS to be measured
observables: List[MixedOp] = []
if "energy" in parsed_input_file["observables"]:
    observables.append(h_qed) # Total energy
    observables.append(MixedOp({("F"): [(1.0, h_el)]})) # Electron energy
    observables.append(MixedOp({("B"): [(1.0, h_ph)]})) # Photon energy
    observables.append(h_int) # Interaction energy
if "particle_number" in parsed_input_file["observables"]:
    observables.append(MixedOp({("F"): [
        # Electron number in mode 1
        (1., FermionicOp({"+_1 -_1": 1}, num_spin_orbitals=len(parsed_input_file["elec_energies"])))
    ]}))
    neighbors = range(-1, 2) if parsed_input_file["local_basis_interaction_type"] == 'nn' else \
        range(-2, 3) if parsed_input_file["local_basis_interaction_type"] == '2nn' else range(-3, 4)
    # Expand the particle operator for each plane wave in the new basis
    for n in range(number_of_modes):
        # Photon number in mode i
        ph_num = BosonicOp({})
        for i in range(number_of_basis_functions):
            for j in neighbors:
                if i + j >= 0 and i + j < number_of_basis_functions:
                    ph_num += BosonicOp({
                        f'+_{i} -_{i+j}': np.conj(projections[n, i]) * projections[n, i+j]
                        }, num_modes=number_of_basis_functions)
        observables.append(MixedOp({("B"): [(1.0, ph_num)]}))
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
            observables.append(MixedOp(
                {("B"): [(prefactor, op1), (prefactor, op2), (prefactor, op3), (prefactor, op4)]}))
# 3. DEFINE THE MAPPERS
mixed_papper = \
    utils.get_mapper(number_of_basis_functions, parsed_input_file["number_of_fock_states"])
# 4. MAP THE HAMILTONIAN AND OBSERVABLES
hqed_mapped = mixed_papper.map(h_qed)
observables_mapped = [mixed_papper.map(op) for op in observables]
# 5. DEFINE THE INITIAL STATE: The matter is in the excited state, the photons in the vacuum state
init_state: dict[int, tuple[complex | np.complex128, complex | np.complex128]] = {}
for n in range(number_of_basis_functions):
    init_state[n] = (np.complex128(1), np.complex128(0))
# Add matter part
init_state[number_of_basis_functions] = (np.complex128(1), np.complex128(0))
init_state[number_of_basis_functions + 1] = (np.complex128(0), np.complex128(1))
# 6. DEFINE THE HARDWARE
if parsed_input_file["hardware"] == "generic_simulator":
    backend = GenericBackendV2(num_qubits=hqed_mapped.num_qubits)
    estimator = Estimator(options={"shots": None})
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
