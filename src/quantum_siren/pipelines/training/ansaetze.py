import pennylane as qml
import pennylane.numpy as np
import torch


class ansaetze:
    @staticmethod
    def nothing(params):
        pass

    @staticmethod
    def circuit_19(params: torch.Tensor | np.ndarray):
        """
        Generates a single layer of circuit_19
        See https://arxiv.org/pdf/1905.10876.pdf

        Args:
            params (torch.Tensor|np.ndarray): Parameters that are
            being utilized in the layer.
            Expects form to be [n_qubits, n_gates_per_layer],
            where n_gates_per_layer=3 in this case.
            If None, then the number of required params per layer is returned.
        """
        if params is None:
            return 3

        for qubit, qubit_params in enumerate(params):
            qml.RX(qubit_params[0], wires=qubit)
            qml.RZ(qubit_params[1], wires=qubit)

        for qubit, qubit_params in enumerate(params):
            qml.CRX(
                qubit_params[2],
                wires=[
                    params.shape[0] - qubit - 1,
                    (params.shape[0] - qubit) % params.shape[0],
                ],
            )

    @staticmethod
    def circuit_18(params: torch.Tensor | np.ndarray):
        """
        Generates a single layer of circuit_18
        See https://arxiv.org/pdf/1905.10876.pdf

        Args:
            params (torch.Tensor|np.ndarray): Parameters that are
            being utilized in the layer.
            Expects form to be [n_qubits, n_gates_per_layer],
            where n_gates_per_layer=3 in this case.
            If None, then the number of required params per layer is returned.
        """
        if params is None:
            return 3

        for qubit, qubit_params in enumerate(params):
            qml.RX(qubit_params[0], wires=qubit)
            qml.RZ(qubit_params[1], wires=qubit)

        for qubit, params in enumerate(params):
            qml.CRZ(
                qubit_params[2],
                wires=[
                    params.shape[0] - qubit - 1,
                    (params.shape[0] - qubit) % params.shape[0],
                ],
            )

    @staticmethod
    def default(params: torch.Tensor | np.ndarray):
        """Encoding of two dimensional data using RX and RY gates.
        The input is repeated across all qubits (vertically),
        as specified by the shape of the input.

        Args:
            params (torch.Tensor | np.ndarray): Input data with the first value
            parameterizing the RX gate and the second value parameterizing the RY gate.
            Expects form to be [n_qubits, 2]
        """
        for qubit, qubit_params in enumerate(params):
            qml.RX(qubit_params[:, 0], wires=qubit)
            if qubit_params.shape[1] > 1:
                qml.RY(qubit_params[:, 1], wires=qubit)

    @staticmethod
    def spread_layers(params: torch.Tensor | np.ndarray):
        """Encoding of two dimensional data using RY gates interleaving
        across the qubits.
        Here, the first qubits takes the first parameter, the second the second,
        the third one the first again and so on.

        Args:
            params (torch.Tensor | np.ndarray): Input data with the first value
            parameterizing the RX gate and the second value parameterizing the RY gate.
            Expects form to be [n_qubits, 2]
        """
        for qubit, qubit_params in enumerate(params):
            if 2 * qubit + 1 > params.shape[0] - 1:
                break
            qml.RY(qubit_params[0], wires=2 * qubit)
            if qubit_params.shape[1] > 1:
                qml.RY(qubit_params[1], wires=2 * qubit + 1)
