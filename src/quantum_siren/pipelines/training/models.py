import pennylane as qml
from pennylane import numpy as np

import torch

from .ansaetze import ansaetze

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
        self.circuit = qml.QNode(
            self._circuit,
            qml.device("default.qubit", shots=self.shots, wires=self.n_qubits),
            interface="torch",
        )

        self.qlayer = qml.qnn.TorchLayer(self.circuit, {"params": self.params.shape})

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
            inputs = np.zeros((self.n_qubits, 1, 3))
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
        return torch.tensor(
            self._forward(params=self.params, inputs=inputs, **kwargs),
            requires_grad=True,
        )
