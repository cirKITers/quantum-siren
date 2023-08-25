import pennylane as qml

import torch

from .ansaetze import ansaetze

import mlflow

class Model(mlflow.pyfunc.PythonModel, torch.nn.Module):
    # class Module(torch.nn.Module):
    def __init__(self, n_qubits, shots, vqc_ansatz, iec_ansatz, n_layers, data_reupload) -> None:
        super().__init__()

        self.shots = None if shots=="None" else shots
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.iec = getattr(ansaetze, iec_ansatz, ansaetze.nothing)
        self.vqc = getattr(ansaetze, vqc_ansatz, ansaetze.nothing)
        
        self.data_reupload = data_reupload

        dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
        self.forward = qml.QNode(self.circuit, dev, interface="torch")
        self.predict = self.forward

        self.initialize_params(n_qubits=self.n_qubits, n_layers=self.n_layers, n_gates_per_layer=self.vqc(None))


    def initialize_params(self, n_qubits, n_layers, n_gates_per_layer):
        self.params = torch.nn.Parameter(torch.rand(size=(n_layers,n_qubits,n_gates_per_layer), requires_grad=True))

    def circuit(self, model_input):
        for l, l_params in enumerate(self.params):
            if l == 0 or (l > 0 and self.data_reupload):
                self.iec(torch.stack([model_input]*(self.n_qubits//2)), limit=self.n_qubits-(l//2)) # half because the coordinates already have 2 dims

            self.vqc(l_params)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def predict(self, context, model_input):
        if type(model_input) != torch.Tensor:
            model_input = torch.tensor(model_input)
        return self.forward(model_input)
        
