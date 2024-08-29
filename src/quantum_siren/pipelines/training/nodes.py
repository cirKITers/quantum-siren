"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.12
"""

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torch.utils.data import DataLoader
from rich.progress import track

import mlflow

from typing import Optional, List, Dict

from .models import TorchModel, TorchReluModel
from .optimizer import QNG, Adam

from ...helpers.visualization import generate_figure


import logging

log = logging.getLogger(__name__)

optimizers = {
    "QNG": QNG,
    "Adam": Adam,
}


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.002) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.best_loss: Optional[float] = None
        self.early_stop: bool = False
        self.loss_log: List[float] = []

    def ask(self, loss: float) -> bool:
        self.loss_log.append(loss)
        if len(self.loss_log) > self.patience:
            var = torch.var(torch.tensor(self.loss_log[-self.patience :]))
        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

        return self.early_stop


class Instructor:
    def __init__(
        self,
        n_layers: int,
        n_qubits: int,
        circuit_type: str,
        data_reupload: bool,
        learning_rate: float,
        shots: int,
        report_figure_every_n_steps: int,
        optimizer: str,
        output_qubit: str,
        initialization: str,
        loss: str,
        seed: int,
        early_stopping: bool,
    ) -> None:
        """
        Initializes the object with the given parameters.

        Args:
            n_layers (int): Number of layers.
            n_qubits (int): Number of qubits.
            circuit_type (str): VQC ansatz.
            iec_ansatz (str): IEC ansatz.
            data_reupload (bool): Flag for data reupload.
            learning_rate (float): Learning rate.
            shots (int): Number of shots.
            report_figure_every_n_steps (int): Number of steps till summary.
            optimizer (str): Optimizer type.
            output_interpretation (str): Output interpretation.
            loss (str): Loss type.
            seed (int): Random seed.
            max_workers (int): Maximum number of workers.

        Returns:
            None
        """
        # this sets a global seed, that, according to documentation, affects the
        # weight initialization and dataloader
        torch.manual_seed(seed)

        self.steps_till_summary = report_figure_every_n_steps

        self.model = TorchModel(
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type,
            # iec_ansatz,
            shots=shots,
            data_reupload=data_reupload,
            output_qubit=output_qubit,
            initialization=initialization,
        )
        dummy = False
        self.dummyModel = None
        if dummy:
            self.dummyModel = TorchReluModel(
                n_inputs=2, n_hidden=100, n_layers=self.model.degree
            )
            # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            # params = sum([torch.prod(p.size()) for p in model_parameters])
        if early_stopping:
            self.earlyStopping = EarlyStopping()
        else:
            self.earlyStopping = None

        if optimizer == "QNG":
            self.optim = QNG(
                params=self.model.parameters(),
                lr=learning_rate,
                qnode=self.model.qnode,
                argnum=None,
            )
        elif optimizer == "Adam":
            self.optim = Adam(params=self.model.parameters(), lr=learning_rate)
            if self.dummyModel:
                self.optim = Adam(params=self.dummyModel.parameters(), lr=learning_rate)
        else:
            raise KeyError(f"No optimizer {optimizer} in {optimizers}")

        self.metrics = {
            "mse": self.mse,
            "ssim": self.ssim,
            "fft_ssim": self.fft_ssim,
            "psnr": self.psnr,
        }

        # set the sign for the loss depending on the metric type
        # a "-1" means "maximize this metric" and a "1" means "minimize"
        if loss == "fft_ssim":
            self.loss = self.fft_ssim
            self.loss_sign = -1
        elif loss == "mse":
            self.loss = self.mse
            self.loss_sign = 1
        elif loss == "psnr":
            self.loss = self.psnr
            self.loss_sign = -1
        elif loss == "ssim":
            self.loss = self.ssim
            self.loss_sign = -1
        else:
            raise KeyError(f"No loss {loss} in {self.metrics}")

    def cost(self, *args: any) -> float:
        return self.loss(*args) * self.loss_sign

    def fft_ssim(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Calculate the Structural Similarity Index (SSIM) between the predicted
        and target tensors using the Fast Fourier Transform (FFT) method.

        Args:
            pred (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The SSIM value between the predicted and target tensors.
        """
        if self.sidelength == -1:
            return -1
        pred_spectrum = torch.fft.fft2(pred.view(self.sidelength, self.sidelength))
        pred_spectrum = torch.fft.fftshift(pred_spectrum)

        target_spectrum = torch.fft.fft2(target.view(self.sidelength, self.sidelength))
        target_spectrum = torch.fft.fftshift(target_spectrum)

        val_abs = self.ssim(
            torch.log(pred_spectrum.abs()), torch.log(target_spectrum.abs())
        )
        val_phase = self.ssim(pred_spectrum.angle(), target_spectrum.angle())

        return (
            val_abs + val_phase
        ) / 2  # because we want to match phase and amplitude but keep the result <=1

    def ssim(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Calculate the Structural Similarity Index (SSIM) between the
        predicted and target tensors.

        Args:
            pred (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            float: The calculated Structural Similarity Index (SSIM) value.
        """
        if self.sidelength == -1:
            return -1
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        val = ssim(
            pred.reshape(1, 1, self.sidelength, self.sidelength),
            target.reshape(1, 1, self.sidelength, self.sidelength),
        )

        return val

    def psnr(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Calculate the Peak Signal-to-Noise Ratio (PSNR) between the
        predicted and target tensors.

        Args:
            pred (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            float: The PSNR value.
        """
        if self.sidelength == -1:
            return -1
        psnr = PeakSignalNoiseRatio(data_range=1.0)
        val = psnr(
            pred.reshape(1, 1, self.sidelength, self.sidelength),
            target.reshape(1, 1, self.sidelength, self.sidelength),
        )

        return val

    def mse(self, pred: torch.Tensor, target: torch.Tensor):
        """
            Calculate the mean squared error (MSE) between the
        predicted tensor and the target tensor.

            Args:
                pred (torch.Tensor): The predicted tensor.
                target (torch.Tensor): The target tensor.

            Returns:
                torch.Tensor: The mean squared error value.
        """
        if len(target.shape) > 1:  # TODO: remove once channels are supported
            target = target[:, 0]
        val = ((pred - target) ** 2).mean()

        return val

    def set_sidelength(self, dataloader: DataLoader):
        if len(dataloader.dataset.shape) == 3:
            self.sidelength = dataloader.dataset.sidelength
        else:
            self.sidelength = -1  # Indicate that we cannot use fft_ssim etc.

    def train(self, dataloader: DataLoader, steps: int):
        """
        Trains the model using the given input and ground truth data
        for the specified number of steps.

        Args:
            dataloader (torch.utils.data.DataLoader): The input data.
            steps (int): The number of steps to train the model.

        Returns:
            torch.nn.Module: The trained model.
        """
        self.set_sidelength(dataloader)

        log.info(f"Training for {steps} steps")
        for step in track(range(steps), description="Training"):
            loss_val = 0

            # Iterate the dataloader
            for coord, target in iter(dataloader):
                if self.dummyModel is not None:
                    pred = self.dummyModel(coord)
                else:
                    pred = self.model(coord)

                loss = self.cost(pred, target)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                loss_val += loss.item()

            # Debug print to show progress in dev mode
            log.debug(f"Step {step}:\t Loss: {loss_val / len(dataloader)}")

            # Retrieve coordinates, predictions and target for reporting
            coords = dataloader.dataset.coords
            if self.dummyModel is not None:
                pred = self.dummyModel(coord).cpu().detach()
            else:
                pred = self.model(coord).cpu().detach()
            targets = dataloader.dataset.values

            for name, metric in self.metrics.items():
                if len(targets.shape) > 1:
                    mlflow.log_metric(
                        name, metric(pred, targets[:, 0]) / len(dataloader), step
                    )
                else:
                    mlflow.log_metric(
                        name, metric(pred, targets) / len(dataloader), step
                    )

            # Report figures
            if not step % self.steps_till_summary:
                fig = generate_figure(
                    shape=[
                        dataloader.dataset.shape[0],
                        dataloader.dataset.shape[1],
                        1,  # TODO: remove once channels are supported
                    ],
                    coords=dataloader.dataset.coords,
                    values=pred,
                    sidelength=dataloader.dataset.sidelength,
                )

                if fig is not None:
                    # Report this figure directly to mlflow (not via kedro)
                    # to show progress in mlflow dashboard
                    mlflow.log_figure(fig, f"prediction_step_{step}.html")

            # Early Stopping
            if self.earlyStopping is not None and self.earlyStopping.ask(loss_val):
                log.info(f"Early stopping triggered in step {step}")
                break

        return self.model


def training(
    n_layers: int,
    n_qubits: int,
    circuit_type: str,
    data_reupload: bool,
    learning_rate: float,
    shots: int,
    report_figure_every_n_steps: int,
    optimizer: str,
    output_qubit: int,
    initialization: str,
    loss: str,
    dataloader,
    steps: int,
    seed: int,
    early_stopping: bool,
):
    """
    A function to train a model using an instructor,
    log the model to MlFlow, and return the trained model.

    Parameters:
        n_layers (int): Number of layers
        n_qubits (int): Number of qubits
        circuit_type (str): VQC ansatz
        iec_ansatz (str): IEC ansatz
        data_reupload (bool): Data reupload flag
        learning_rate (float): Learning rate
        shots (int): Number of shots
        report_figure_every_n_steps (int): Report figure every n steps
        optimizer (str): Optimizer type
        output_interpretation (int): Output interpretation
        loss (str): Loss function
        dataloader (torch.utils.data.DataLoader): The input data
        steps (int): Number of training steps
        seed (int): Random seed
        max_workers (int): Maximum number of workers

    Returns:
        dict: A dictionary containing the trained model
    """
    instructor = Instructor(
        n_layers=n_layers,
        n_qubits=n_qubits,
        circuit_type=circuit_type,
        data_reupload=data_reupload,
        learning_rate=learning_rate,
        shots=shots,
        report_figure_every_n_steps=report_figure_every_n_steps,
        optimizer=optimizer,
        output_qubit=output_qubit,
        initialization=initialization,
        loss=loss,
        seed=seed,
        early_stopping=early_stopping,
    )

    model = instructor.train(dataloader, steps)

    logging.info("Logging Model to MlFlow")
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="qameraman",
    )

    return {"model": model}
