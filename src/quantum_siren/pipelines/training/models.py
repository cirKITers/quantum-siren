import pennylane as qml
from pennylane import numpy as np
import hashlib
import os
import torch

from typing import Dict, Optional, Tuple, Callable, Union, List

import logging

log = logging.getLogger(__name__)

from qml_essentials.model import Model


class TorchReluModel(torch.nn.Module):
    def __init__(
        self,
        n_inputs,
        n_hidden,
        n_layers,
    ):
        super().__init__()

        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(n_inputs, n_hidden, bias=False),
        #     *[torch.nn.Linear(n_hidden, n_hidden, bias=False), torch.nn.ReLU()]
        #     * (n_layers - 1),
        #     torch.nn.Linear(n_hidden, 1),
        #     torch.nn.ReLU(),
        # )
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x).flatten()


class TorchModel(Model, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)
        torch.nn.Module.__init__(self)

        # this overwrites the circuit in qml essentials by addding the torch interface
        self.qnode = qml.QNode(
            self._circuit,
            qml.device("default.qubit", shots=self.shots, wires=self.n_qubits),
            interface="torch",
        )

        self.circuit = qml.qnn.TorchLayer(self.qnode, {"params": self.params.shape})

    def _iec(
        self,
        inputs: np.ndarray,
        data_reupload: bool = True,
    ) -> None:
        """Encoding of two dimensional data using RX and RY gates.
        The input is repeated across all qubits (vertically),
        as specified by the shape of the input.

        This overwrites the qml-essentials implementation by adding 2D input

        Args:
            inputs (torch.Tensor | np.ndarray): Input data with the first value
            parameterizing the RX gate and the second value parameterizing the RY gate.
            Expects form to be [n_qubits, batch, 2]
        """
        assert data_reupload, "This model only supports data reuploading"

        if inputs is None:
            inputs = torch.zeros((self.n_qubits, 1, 3))
        else:
            inputs = torch.stack([inputs] * self.n_qubits)

        for qubit, qubit_params in enumerate(inputs):
            if qubit_params.shape[1] == 1:
                qml.RX(qubit_params[:, 0], wires=qubit)
            elif qubit_params.shape[1] == 2:
                qml.RX(qubit_params[:, 0], wires=qubit)
                qml.RY(qubit_params[:, 1], wires=qubit)
            elif qubit_params.shape[1] == 3:
                qml.Rot(
                    qubit_params[:, 0],
                    qubit_params[:, 1],
                    qubit_params[:, 2],
                    wires=qubit,
                )
            else:
                raise ValueError(
                    "The number of parameters for this IEC cannot be greater than 3"
                )

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Perform a forward pass of the quantum circuit.
        This overwrites the implementation in qml-essentials by adding the torch interface

        """
        # Call forward method which handles the actual caching etc.
        return torch.nn.Module.__call__(self, *args, **kwargs)

    # def predict(self, context, model_input):
    #     if type(model_input) != torch.Tensor:
    #         model_input = torch.tensor(model_input)
    #     return self.forward(model_input)

    def forward(self, inputs, **kwargs):
        kwargs.setdefault("cache", False)
        kwargs.setdefault("force_mean", True)

        # as we're not getting a torch tensor, but a pennylane Tensor, we have to convert again
        return self._forward(params=self.params, inputs=inputs, **kwargs)

    def _forward(
        self,
        params: np.ndarray,
        inputs: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
        cache: Optional[bool] = False,
        execution_type: Optional[str] = None,
        force_mean: Optional[bool] = False,
    ) -> np.ndarray:
        """
        REMOVE WHEN MERGED

        Perform a forward pass of the quantum circuit.

        Args:
            params (np.ndarray): Weight vector of shape
                [n_layers, n_qubits*n_params_per_layer].
            inputs (np.ndarray): Input vector of shape [1].
            noise_params (Optional[Dict[str, float]], optional): The noise parameters.
                Defaults to None which results in the last
                set noise parameters being used.
            cache (Optional[bool], optional): Whether to cache the results.
                Defaults to False.
            execution_type (str, optional): The type of execution.
                Must be one of 'expval', 'density', or 'probs'.
                Defaults to None which results in the last set execution type
                being used.

        Returns:
            np.ndarray: The output of the quantum circuit.
                The shape depends on the execution_type.
                - If execution_type is 'expval', returns an ndarray of shape
                    (1,) if output_qubit is -1, else (len(output_qubit),).
                - If execution_type is 'density', returns an ndarray
                    of shape (2**n_qubits, 2**n_qubits).
                - If execution_type is 'probs', returns an ndarray
                    of shape (2**n_qubits,) if output_qubit is -1, else
                    (2**len(output_qubit),).

        Raises:
            NotImplementedError: If the number of shots is not None or if the
                expectation value is True.
        """
        # set the parameters as object attributes
        if noise_params is not None:
            self.noise_params = noise_params
        if execution_type is not None:
            self.execution_type = execution_type

        # the qasm representation contains the bound parameters, thus it is ok to hash that
        hs = hashlib.md5(
            repr(
                {
                    "n_qubits": self.n_qubits,
                    "n_layers": self.n_layers,
                    "pqc": self.pqc.__class__.__name__,
                    "dru": self.data_reupload,
                    "params": params,
                    "noise_params": self.noise_params,
                    "execution_type": self.execution_type,
                    "inputs": inputs,
                    "output_qubit": self.output_qubit,
                }
            ).encode("utf-8")
        ).hexdigest()

        result: Optional[np.ndarray] = None
        if cache:
            name: str = f"pqc_{hs}.npy"

            cache_folder: str = ".cache"
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)

            file_path: str = os.path.join(cache_folder, name)

            if os.path.isfile(file_path):
                result = np.load(file_path)

        if result is None:
            # if density matrix requested or noise params used
            if self.execution_type == "density" or self.noise_params is not None:
                result = self.circuit_mixed(
                    params=params,
                    inputs=inputs,
                )
            else:
                if isinstance(self.circuit, qml.qnn.torch.TorchLayer):
                    result = self.circuit(
                        inputs=inputs,
                    )
                else:
                    result = self.circuit(
                        params=params,
                        inputs=inputs,
                    )

        if self.execution_type == "expval" and isinstance(self.output_qubit, list):
            if isinstance(result, list):
                result = np.stack(result)

            # Calculating mean value after stacking, to not
            # discard gradient information
            if force_mean:
                # exception for torch layer because it swaps batch and output dimension
                if isinstance(self.circuit, qml.qnn.torch.TorchLayer):
                    result = result.mean(axis=-1)
                else:
                    result = result.mean(axis=0)

        if cache:
            np.save(file_path, result)

        return result
