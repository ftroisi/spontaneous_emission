""""This module contains ulitiy functions for io and visualization of data"""
from typing import List, Literal
import numpy as np
import matplotlib.pyplot as plt

def visualize_matrix(m, type_to_plot: Literal["abs", "real"]='abs', log=False) -> None:
    """Visualize a matrix M"""
    if type_to_plot == 'abs':
        matrix = np.abs(m)
    else:
        matrix = np.real(m)

    fig = plt.figure(1)
    ax1 = fig.gca()
    if log:
        cp = ax1.matshow(np.log(matrix), interpolation='none')
    else:
        cp = ax1.matshow(matrix, interpolation='none', cmap='Reds')
    cbar = fig.colorbar(cp, ax=ax1)
    cbar.set_label('M.'+type_to_plot, rotation=270, labelpad=17)
    plt.show()

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

def read_input_file(
        filename: str | None = None) -> dict[str, str | int | float | List[float] | List[str]]:
    "Reads input file and returns a dictionary with the values"
    values: dict[str, str | int | float | List[float] | List[str]] = {}
    filename = filename if filename is not None else 'input'
    with open(filename, 'r', encoding="UTF-8") as f:
        lines: List[str] = f.readlines()
        for _, line in enumerate(lines):
            if line[0] == '#' or line.strip() == '' or line.isspace():
                continue
            l: List[str] = line.split("\n")[0].split("#")[0].split('=')
            key, value = l[0], l[1]
            # Assign values
            # System parameters
            if key.strip() == "elec_energies":
                electron_eigenvalues: List[float] = []
                for electron_eigenvalue in value.split(';'):
                    electron_eigenvalues.append(float(electron_eigenvalue.strip()))
                values["elec_energies"] = electron_eigenvalues
            elif key.strip() == "modes_energies":
                modes_energies: List[float] = []
                for mode_energy in value.split(';'):
                    modes_energies.append(float(mode_energy.strip()))
                values["modes_energies"] = modes_energies
            elif key.strip() == "cavity_length":
                values["cavity_length"] = float(value.strip())
            elif key.strip() == "number_of_fock_states":
                values["number_of_fock_states"] = int(value.strip())
            elif key.strip() == "number_of_modes":
                values["number_of_modes"] = int(value.strip())
            # Basis functions approximation parameters
            elif key.strip() == "number_of_basis_functions":
                values["number_of_basis_functions"] = int(value.strip())
            elif key.strip() == "local_basis_interaction_type":
                values["local_basis_interaction_type"] = value.strip()
            elif key.strip() == "bilinear_threshold":
                values["bilinear_threshold"] = float(value.strip())
            # Time evolution parameters
            elif key.strip() == "delta_t":
                values["delta_t"] = float(value.strip())
            elif key.strip() == "final_time":
                values["final_time"] = float(value.strip())
            # Optimization
            elif key.strip() == "optimization_level":
                values["optimization_level"] = int(value.strip())
            elif key.strip() == "hardware":
                values["hardware"] = value.strip()
            elif key.strip() == "shots":
                values["shots"] = int(value.strip())
            elif key.strip() == "time_evolution_strategy":
                values["time_evolution_strategy"] = value.strip()
            elif key.strip() == "time_evolution_synthesis":
                values["time_evolution_synthesis"] = value.strip()
            # Observables
            elif key.strip() == "observables":
                observables_requested: List[float] = []
                for obs in value.split(';'):
                    observables_requested.append(obs.strip())
                values["observables"] = observables_requested
        f.close()
    return values
