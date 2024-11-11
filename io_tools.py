""""This module contains ulitiy functions for io and visualization of data"""
from typing import Literal
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
