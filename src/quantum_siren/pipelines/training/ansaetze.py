import pennylane as qml
import pennylane.numpy as np
import torch


class ansaetze:
    @staticmethod
    def nothing(params):
        pass

    @staticmethod
    def circuit_19(params: torch.tensor | np.ndarray):
        """
        Generates a single layer of circuit_19
        See https://arxiv.org/pdf/1905.10876.pdf

        Args:
            params (torch.tensor|np.ndarray): Parameters that are
            being utilized in the layer.
            Expects form to be [n_qubits, n_gates_per_layer],
            where n_gates_per_layer=3 in this case.
            If None, then the number of required params per layer is returned.
        """
        if params is None:
            return 3

        for qubit, params in enumerate(params):
            qml.RX(params[0], wires=qubit)
            qml.RZ(params[1], wires=qubit)

        for qubit, params in enumerate(params):
            qml.CRX(
                params[2],
                wires=[
                    params.shape[0] - qubit - 1,
                    (params.shape[0] - qubit) % params.shape[0],
                ],
            )

    @staticmethod
    def circuit_18(params: torch.tensor | np.ndarray):
        """
        Generates a single layer of circuit_18
        See https://arxiv.org/pdf/1905.10876.pdf

        Args:
            params (torch.tensor|np.ndarray): Parameters that are
            being utilized in the layer.
            Expects form to be [n_qubits, n_gates_per_layer],
            where n_gates_per_layer=3 in this case.
            If None, then the number of required params per layer is returned.
        """
        if params is None:
            return 3

        for qubit, params in enumerate(params):
            qml.RX(params[0], wires=qubit)
            qml.RZ(params[1], wires=qubit)

        for qubit, params in enumerate(params):
            qml.CRZ(
                params[2],
                wires=[
                    params.shape[0] - qubit - 1,
                    (params.shape[0] - qubit) % params.shape[0],
                ],
            )

    @staticmethod
    def default(params: torch.tensor | np.ndarray):
        """Encoding of two dimensional data using RX and RY gates.
        The input is repeated across all qubits (vertically), as specified by the shape of the input.

        Args:
            params (torch.tensor | np.ndarray): Input data with the first value parameterizing the RX gate
            and the second value parameterizing the RY gate. Expects form to be [n_qubits, 2]
        """
        for qubit, params in enumerate(params):
            qml.RX(params[0], wires=qubit)
            qml.RY(params[1], wires=qubit)

    @staticmethod
    def spread_layers(params: torch.tensor | np.ndarray):
        """Encoding of two dimensional data using RY gates interleaving across the qubits.
        Here, the first qubits takes the first parameter, the second the second, the third one the first again and so on.

        Args:
            params (torch.tensor | np.ndarray): Input data with the first value parameterizing the RX gate
            and the second value parameterizing the RY gate. Expects form to be [n_qubits, 2]
        """
        for qubit, params in enumerate(params):
            if 2 * qubit + 1 > params.shape[0] - 1:
                break
            qml.RY(params[0], wires=2 * qubit)
            qml.RY(params[1], wires=2 * qubit + 1)
