"""Evaluator of observables for algorithms."""

from __future__ import annotations
from typing import Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import BaseEstimatorV2 as BaseEstimator, PrimitiveResult
from qiskit.primitives.containers.data_bin import DataBin
from qiskit.primitives.containers.pub_result import PubResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.primitives.primitive_job import PrimitiveJob

def estimate_observables(
    estimator: BaseEstimator,
    quantum_state: QuantumCircuit,
    observables: dict[str, BaseOperator],
    threshold: float = 1e-12,
) -> dict[str, tuple[float, dict[str, Any]]]:
    """
    Accepts a sequence of operators and calculates their expectation values - means
    and metadata. They are calculated with respect to a quantum state provided. A user
    can optionally provide a threshold value which filters mean values falling below the threshold.

    Args:
        estimator: An estimator primitive used for calculations.
        quantum_state: A (parameterized) quantum circuit preparing a quantum state that expectation
            values are computed against.
        observables: A list of operators whose expectation values are to be calculated.
        parameter_values: Optional list of parameters values to evaluate the quantum circuit on.
        threshold: A threshold value that defines which mean values should be neglected (helpful for
            ignoring numerical instabilities close to 0).

    Returns:
        A list of tuples (mean, metadata).

    Raises:
        AlgorithmError: If a primitive job is not successful.
    """
    if len(observables.values()) > 0:
        observables_list: ObservablesArray = _prepare_observables(observables)
        # Create the estimator pub
        pub = EstimatorPub(quantum_state, observables_list, precision=threshold)
        try:
            estimator_job: PrimitiveJob[PrimitiveResult[PubResult]] = estimator.run(pubs=[pub])
            data: DataBin = estimator_job.result()[0].data
        except Exception as exc:
            raise ValueError("The primitive job failed!") from exc

        # Extract the expectation values, discarding values below threshold
        observables_means = np.array(
            data["evs"] * (np.abs(data["evs"]) > threshold) if "evs" in data.keys() else [])
        # Get the standard deviation
        observables_stds = np.array(data["stds"] if "stds" in data.keys() else [])
        # zip means and stds into tuples
        observables_results = list(zip(observables_means, observables_stds))
    else:
        observables_results = []

    return _prepare_result(observables_results, observables)

def _prepare_observables(observables: dict[str, BaseOperator]) -> ObservablesArray:
    """
    Replaces all occurrence of operators equal to 0 in the list with an equivalent
    ``SparsePauliOp`` operator.
    """
    # First, transform the dict of observables into a list
    observables_list = list(observables.values())
    # Then, replace 0 operators with SparsePauliOp
    zero_op: SparsePauliOp = SparsePauliOp.from_list([("I" * observables_list[0].num_qubits, 0)])
    for ind, observable in enumerate(observables_list):
        if observable == 0:
            observables_list[ind] = zero_op
    # Finally, create the ObservalesArray object
    return ObservablesArray(observables_list)

def _prepare_result(
    observables_results: list[tuple[float, dict]],
    observables: dict[str, BaseOperator],
) -> dict[str, tuple[float, dict[str, Any]]]:
    """
    Prepares a list of tuples of eigenvalues and metadata tuples from
    ``observables_results`` and ``observables``.

    Args:
        observables_results: A list of tuples (mean, metadata).
        observables: A list or a dictionary of operators whose expectation values are to be
            calculated.

    Returns:
        A list or a dictionary of tuples (mean, metadata).
    """
    if len(observables.keys()) != len(observables_results):
        raise ValueError("The number of requetsed observables and their results do not match!")

    observables_eigenvalues: dict[str, tuple[float, dict]] = {}
    for key, value in zip(observables.keys(), observables_results):
        observables_eigenvalues[key] = value

    return observables_eigenvalues
