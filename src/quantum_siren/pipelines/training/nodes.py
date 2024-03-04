"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.12
"""

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

import math
import logging

import plotly.graph_objects as go

import mlflow
import logging

from .models import Model
from .optimizer import QNG, Adam

from typing import Dict

optimizers = {
    "QNG": QNG,
    "Adam": Adam,
}

log = logging.getLogger(__name__)


class Instructor:
    def __init__(
        self,
        n_layers: int,
        n_qubits: int,
        vqc_ansatz: str,
        iec_ansatz: str,
        data_reupload: bool,
        learning_rate: float,
        shots: int,
        report_figure_every_n_steps: int,
        optimizer: str,
        output_interpretation: str,
        loss: str,
        seed: int,
        max_workers: int,
    ) -> None:
        """
        Initializes the object with the given parameters.

        Args:
            n_layers (int): Number of layers.
            n_qubits (int): Number of qubits.
            vqc_ansatz (str): VQC ansatz.
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

        self.model = Model(
            n_qubits,
            shots,
            vqc_ansatz,
            iec_ansatz,
            n_layers,
            data_reupload,
            output_interpretation,
            max_workers,
        )

        if optimizer == "QNG":
            self.optim = QNG(
                params=self.model.parameters(),
                lr=learning_rate,
                qnode=self.model.qnode,
                argnum=None,
            )
        elif optimizer == "Adam":
            self.optim = Adam(params=self.model.parameters(), lr=learning_rate)
        else:
            raise KeyError(f"No optimizer {optimizer} in {optimizers}")
        # self.optim = torch.optim.Adam(lr=learning_rate, params=self.model.parameters())
        pass

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

        # del self.metrics[loss]

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
        val = ((pred - target) ** 2).mean()

        return val

    def calculate_sidelength(self, img: torch.Tensor):
        self.sidelength = int(math.sqrt(img.shape[0]))

    def train(self, model_input: torch.Tensor, ground_truth: torch.Tensor, steps: int):
        """
        Trains the model using the given input and ground truth data
        for the specified number of steps.

        Args:
            model_input (torch.Tensor): The input data for the model.
            ground_truth (torch.Tensor): The ground truth data for the model.
            steps (int): The number of steps to train the model.

        Returns:
            torch.nn.Module: The trained model.
        """
        self.calculate_sidelength(ground_truth)

        for step in range(steps):
            model_output = self.model(model_input)

            loss_val = self.cost(model_output, ground_truth)
            # mlflow.log_metric("Loss", loss_val.item(), step)

            for name, metric in self.metrics.items():
                val = metric(model_output, ground_truth)
                mlflow.log_metric(name, val, step)

            if not step % self.steps_till_summary:
                # print(self.params)
                # print(f"Step {step}:\t Loss: {loss_val.item()}\t SSIM: {ssim_val}")
                fig = go.Figure(
                    data=go.Heatmap(
                        z=model_output.cpu()
                        .view(self.sidelength, self.sidelength)
                        .detach()
                        .numpy(),
                        colorscale="RdBu",
                        zmid=0,
                    )
                )
                fig.update_layout(
                    yaxis=dict(scaleanchor="x", autorange="reversed"),
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                mlflow.log_figure(fig, f"prediction_step_{step}.html")
                # print(f"Params: {params}")
                # img_grad = gradient(model_output, coords)
                # img_laplacian = laplace(model_output, coords)

                # fig, axes = plt.subplots(1,3, figsize=(18,6))
                # axes[1].imshow(img_grad.norm(dim=-1).cpu().view(sidelength,sidelength).detach().numpy())
                # axes[2].imshow(img_laplacian.cpu().view(sidelength,sidelength).detach().numpy())
                # plt.show()
                log.debug(f"Step {step}:\t Loss: {loss_val.item()}")

            self.optim.zero_grad()
            loss_val.backward()
            self.optim.step()

        return self.model


def generate_instructor(
    n_layers: int,
    n_qubits: int,
    vqc_ansatz: str,
    iec_ansatz: str,
    data_reupload: bool,
    learning_rate: float,
    shots: int,
    report_figure_every_n_steps: int,
) -> Dict[str, Instructor]:
    instructor = Instructor(
        n_layers,
        n_qubits,
        vqc_ansatz,
        iec_ansatz,
        data_reupload,
        learning_rate,
        shots,
        report_figure_every_n_steps,
    )

    return {"instructor": instructor}


def training(
    n_layers: int,
    n_qubits: int,
    vqc_ansatz: str,
    iec_ansatz: str,
    data_reupload: bool,
    learning_rate: float,
    shots: int,
    report_figure_every_n_steps: int,
    optimizer: str,
    output_interpretation: int,
    loss: str,
    model_input: torch.Tensor,
    ground_truth: torch.Tensor,
    steps: int,
    seed: int,
    max_workers: int,
):
    """
    A function to train a model using an instructor,
    log the model to MlFlow, and return the trained model.

    Parameters:
        n_layers (int): Number of layers
        n_qubits (int): Number of qubits
        vqc_ansatz (str): VQC ansatz
        iec_ansatz (str): IEC ansatz
        data_reupload (bool): Data reupload flag
        learning_rate (float): Learning rate
        shots (int): Number of shots
        report_figure_every_n_steps (int): Report figure every n steps
        optimizer (str): Optimizer type
        output_interpretation (int): Output interpretation
        loss (str): Loss function
        model_input (torch.Tensor): Model input
        ground_truth (torch.Tensor): Ground truth
        steps (int): Number of training steps
        seed (int): Random seed
        max_workers (int): Maximum number of workers

    Returns:
        dict: A dictionary containing the trained model
    """
    instructor = Instructor(
        n_layers,
        n_qubits,
        vqc_ansatz,
        iec_ansatz,
        data_reupload,
        learning_rate,
        shots,
        report_figure_every_n_steps,
        optimizer,
        output_interpretation,
        loss,
        seed,
        max_workers,
    )

    model = instructor.train(model_input, ground_truth, steps)

    logging.info("Logging Model to MlFlow")
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="qameraman",
    )

    return {"model": model}
