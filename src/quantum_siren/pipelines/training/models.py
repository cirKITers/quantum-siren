import pennylane as qml
from pennylane import numpy as np

import torch

from .ansaetze import ansaetze

import logging

log = logging.getLogger(__name__)

from qml_essentials.model import Model


class TorchModel(Model, torch.nn.Module):
    def __init__(self, *args, outputs=1, **kwargs):
        Model.__init__(self, *args, **kwargs)
        torch.nn.Module.__init__(self)

        self.qnode = qml.QNode(
            self._torch_circuit,
            qml.device("default.qubit", shots=self.shots, wires=self.n_qubits),
            interface="torch",
        )
        # self.qnodes = [
        #     qml.QNode(
        #         self._torch_circuit,
        #         qml.device("default.qubit", shots=self.shots, wires=self.n_qubits),
        #         interface="torch",
        #     )
        #     for _ in range(outputs)
        # ]

        self.qlayer = qml.qnn.TorchLayer(self.qnode, {"params": self.params.shape})
        # self.qlayers = [
        #     qml.qnn.TorchLayer(qnode, {"params": self.params.shape})
        #     for qnode in self.qnodes
        # ]

        pass

    def _torch_circuit(self, params, inputs=None):
        if inputs is None:
            inputs = self._inputs
        else:
            self._inputs = inputs
        return self._circuit(params, inputs)

    def _iec(
        self,
        inputs: np.ndarray,
        data_reupload: bool = True,
    ) -> None:
        """Encoding of two dimensional data using RX and RY gates.
        The input is repeated across all qubits (vertically),
        as specified by the shape of the input.

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

        Args:
            inputs (np.ndarray): input vector of size 1
            params (Optional[np.ndarray], optional): weight vector of size n_layers*(n_qubits*3-1). Defaults to None.
            noise_params (Optional[Dict[str, float]], optional): dictionary with noise parameters. Defaults to None.
            cache (Optional[bool], optional): cache the circuit. Defaults to False.
            state_vector (bool, optional): measure the state vector instead of the wave function. Defaults to False.

        Returns:
            np.ndarray: Expectation value of PauliZ(0) of the circuit.
        """
        # Call forward method which handles the actual caching etc.
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def predict(self, context, model_input):
        if type(model_input) != torch.Tensor:
            model_input = torch.tensor(model_input)
        return self.forward(model_input)

    def forward(self, model_input):
        def call_qlayer(inputs):
            return torch.stack(
                [qlayer(inputs[:, l]) for qlayer, l in zip(self.qlayers, model_input)],
                axis=1,
            )

        if self.output_qubit < 0:
            out = torch.mean(call_qlayer(model_input), axis=1)
        else:
            out = call_qlayer(model_input)

        return out


# class TorchModelv(torch.nn.Module):
#     # class Module(torch.nn.Module):
#     def __init__(
#         self,
#         n_qubits: int,
#         shots: int,
#         circuit_type: str,
#         iec_ansatz: str,
#         n_layers: int,
#         data_reupload: bool,
#         output_interpretation: int,
#         max_workers,
#     ) -> None:
#         super().__init__()

#         log.info(f"Creating Model with {n_qubits} Qubits, {n_layers} Layers.")

#         self.shots = None if shots == "None" else shots
#         self.max_workers = None if max_workers == "None" else max_workers
#         self.n_qubits = n_qubits
#         # Following the ideas from https://doi.org/10.48550/arXiv.2008.08605
#         # we add an additional layer to "sourround" our encoding
#         self.n_layers = n_layers  # number of "visible" layers
#         self._n_layers_p1 = n_layers + 1  # number of actual layers for weight init etc.

#         self.iec = getattr(ansaetze, iec_ansatz, ansaetze.nothing)
#         self.vqc = getattr(ansaetze, circuit_type, ansaetze.nothing)

#         if output_interpretation >= 0:
#             output_interpretation = int(output_interpretation)
#             assert output_interpretation < self.n_qubits, (
#                 f"Output interpretation parameter {output_interpretation} "
#                 "can either be a qubit (integer smaller n_qubits) or <0 (all qubits)"
#             )

#         self.output_interpretation = output_interpretation

#         self.data_reupload = data_reupload

#         dev = qml.device(
#             "default.qubit",
#             wires=self.n_qubits,
#             shots=self.shots,
#             max_workers=self.max_workers,
#         )

#         self.qnode = qml.QNode(self.circuit, dev, interface="torch")
#         # print(qml.draw(self.circuit)(torch.rand(self._n_layers_p1, n_qubits, self.vqc(None)), torch.tensor([[0,1]])))
#         self.qlayer = qml.qnn.TorchLayer(
#             self.qnode,
#             {"weights": [self._n_layers_p1, n_qubits, self.vqc(None)]},
#         )

#     def circuit(self, weights, inputs=None):
#         if inputs is None:
#             inputs = self._inputs
#         else:
#             self._inputs = inputs

#         dru = torch.zeros(len(weights))
#         if self.data_reupload != 0:
#             dru[:: int(1 / self.data_reupload)] = 1

#         # when iterating weights, the first dim. is the layer, the second is qubits
#         for layer, layer_params in enumerate(weights[:-1]):  # N of (N+1) layers
#             self.vqc(layer_params)
#             qml.Barrier()
#             if layer == 0 or dru[layer] == 1:
#                 self.iec(
#                     torch.stack([inputs] * self.n_qubits),
#                 )  # half because the coordinates already have 2 dims

#         self.vqc(weights[-1])  # the N+1 layer

#         if self.output_interpretation < 0:
#             return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
#         else:
#             return qml.expval(qml.PauliZ(self.output_interpretation))

#     def predict(self, context, model_input):
#         if type(model_input) != torch.Tensor:
#             model_input = torch.tensor(model_input)
#         return self.forward(model_input)

#     def forward(self, model_input):
#         if self.output_interpretation < 0:
#             out = torch.mean(self.qlayer(model_input), axis=1)
#         else:
#             out = self.qlayer(model_input)

#         return out
