"""
This is a boilerplate pipeline 'postprocessing'
generated using Kedro 0.18.12
"""

from quantum_siren.helpers.colors import hsv_to_rgb

import torch
import plotly.graph_objects as go
import math
import scipy
import numpy as np
import plotly.express as px

from typing import Dict, Union, Tuple

import logging

log = logging.getLogger(__name__)


def predict(model, coords):
    model_output = model(coords)

    return {"prediction": model_output.view(-1).detach()}


def upscaling(
    model: torch.nn.Module, coords: torch.Tensor, factor: float, shape: torch.Size
) -> Dict[str, Union[go.Figure, torch.Tensor]]:
    """
    Upscales the given coordinates using the given model.

    Args:
        model (torch.nn.Module): The model to use for upscaling.
        coords (torch.Tensor): The coordinates to upscale.
        factor (float): The factor by which to upscale the coordinates.
        shape (torch.Size): The shape of the input data.

    Returns:
        Dict[str, Union[go.Figure, torch.Tensor]]: A dictionary containing the
            upscaled figure, the upscaled image, and the upscaled coordinates.
    """
    # 1-D case
    if len(shape) == 2:
        pred_upscaled_fig = go.Figure()
        model_output = model(coords).detach()
        upscaled_coords = coords
    # 2-D case
    elif len(shape) == 3:
        sidelength = math.sqrt(coords.shape[0])
        upscaled_sidelength = int(sidelength * factor)

        upscaled_coords = torch.zeros(size=(upscaled_sidelength**2, 2))
        it = 0
        for x in torch.linspace(coords.min(), coords.max(), upscaled_sidelength):
            for y in torch.linspace(coords.min(), coords.max(), upscaled_sidelength):
                upscaled_coords[it] = torch.tensor([x, y])
                it += 1

        model_output = model(upscaled_coords).detach()

        pred_upscaled_fig = go.Figure(
            data=go.Heatmap(
                z=model_output.cpu()
                .view(upscaled_sidelength, upscaled_sidelength)
                .detach()
                .numpy(),
                colorscale="RdBu",
                zmid=0,
            )
        )
        pred_upscaled_fig.update_layout(
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        model_output = model(coords).detach()
        upscaled_coords = coords
        pred_upscaled_fig = go.Figure()

        log.warning(
            f"Dataset has {len(shape)} dimension(s).\
            No visualization possible"
        )

    return {
        "pred_upscaled_fig": pred_upscaled_fig,
        "upscaled_image": model_output,
        "upscaled_coordinates": upscaled_coords,
    }


def upscaling_ground_truth(ground_truth, factor, shape):
    # 1-D case
    if len(shape) == 2:
        log.warning(
            f"Dataset has {len(shape)} dimension(s).\
            No visualization possible"
        )

    # 2-D case
    elif len(shape) == 3:
        sidelength = int(math.sqrt(ground_truth.shape[0]))
        upscaled_sidelength = int(sidelength * factor)

        upscaled_gt = scipy.interpolate.RectBivariateSpline(
            np.linspace(0, 1, sidelength),
            np.linspace(0, 1, sidelength),
            ground_truth.cpu().view(sidelength, sidelength).detach().numpy(),
        )

        gt_upscaled_fig = go.Figure(
            data=go.Heatmap(
                z=upscaled_gt(
                    np.linspace(0, 1, upscaled_sidelength),
                    np.linspace(0, 1, upscaled_sidelength),
                ),
                colorscale="RdBu",
                zmid=0,
            )
        )
        gt_upscaled_fig.update_layout(
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        gt_upscaled_fig = go.Figure()
        upscaled_gt = ground_truth

        log.warning(
            f"Dataset has {len(shape)} dimension(s).\
            No visualization possible"
        )

    return {
        "gt_upscaled_fig": gt_upscaled_fig,
        "upscaled_gt_image": upscaled_gt,
    }


def pixelwise_difference(
    prediction: torch.Tensor,  # shape: (N,) or (N, M)
    target: torch.Tensor,  # shape: (N,) or (N, M)
    shape: Tuple[int, ...],  # shape of the input data
) -> Dict[str, go.Figure]:
    """
    Calculate the difference between the predicted and target values pixelwise.

    Args:
        prediction (torch.Tensor): Predicted values. Shape: (N,) or (N, M).
        target (torch.Tensor): Target values. Shape: (N,) or (N, M).
        shape (Tuple[int, ...]): Shape of the input data.

    Returns:
        Dict[str, go.Figure]: Dictionary containing the pixelwise difference figure.
    """

    if len(shape) == 2:
        pixelwise_diff_fig = go.Figure()
    elif len(shape) == 3:
        sidelength = int(math.sqrt(prediction.shape[0]))

        difference = prediction - target.view(sidelength**2)

        pixelwise_diff_fig = go.Figure(
            data=go.Heatmap(
                z=difference.cpu().view(sidelength, sidelength).detach().numpy(),
                colorscale="RdBu",
                zmid=0,
            )
        )

        pixelwise_diff_fig.update_layout(
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        pixelwise_diff_fig = go.Figure()
        log.warning(
            f"Dataset has {len(shape)} dimension(s).\
            No visualization possible"
        )

    return {"pixelwise_diff_fig": pixelwise_diff_fig}


def plot_gradients(
    model: torch.nn.Module,
    target: torch.Tensor,
    coords: torch.Tensor,
    shape: Tuple[int, ...],
) -> Dict[str, go.Figure]:
    """Plots the gradients of the model.
    Args:
        model (torch.nn.Module): The model to be plotted.
        target (torch.Tensor): The target values.
        coords (torch.Tensor): The coordinates.
        shape (Tuple[int, ...]): The shape of the input data.
    Returns:
        Dict[str, go.Figure]: A dictionary containing the plotted figures.
    """
    coords.requires_grad = True
    pred = model(coords)

    # ----------------------------------------------------------------
    # Gradient Prediction
    # same shape as coordinates
    pred_dc = torch.autograd.grad(
        outputs=pred,
        inputs=[coords],
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
    )[0]

    # 1-D case
    if len(shape) == 2:
        # ----------------------------------------------------------------
        # Gradient Prediction

        pred_gradient_fig = go.Figure(
            data=[
                go.Scatter(
                    x=coords.detach().flatten(),
                    y=pred_dc.detach().flatten(),
                    mode="lines",
                    name="Prediction",
                ),
            ]
        )
        pred_gradient_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # ----------------------------------------------------------------
        # Laplacian Prediction

        pred_dcdc = torch.autograd.grad(
            outputs=pred_dc,
            inputs=[coords],
            grad_outputs=torch.ones_like(pred_dc),
            create_graph=True,
        )[0]

        pred_laplacian_fig = go.Figure(
            data=[
                go.Scatter(
                    x=coords.detach().flatten(),
                    y=pred_dcdc.detach().flatten(),
                    mode="lines",
                    name="Prediction",
                ),
            ]
        )
        pred_laplacian_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # ----------------------------------------------------------------
        # Gradient Ground Truth

        gt_dc = scipy.ndimage.sobel(target)

        gt_gradient_fig = go.Figure(
            data=[
                go.Scatter(
                    x=coords.detach().flatten(),
                    y=gt_dc,
                    mode="lines",
                    name="Prediction",
                ),
            ]
        )
        gt_gradient_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # ----------------------------------------------------------------
        # Laplacian Ground Truth

        gt_dcdc = scipy.ndimage.laplace(target)

        gt_laplacian_fig = go.Figure(
            data=[
                go.Scatter(
                    x=coords.detach().flatten(),
                    y=gt_dcdc,
                    mode="lines",
                    name="Prediction",
                ),
            ]
        )
        gt_laplacian_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
        )
    # 2-D case
    elif len(shape) == 3:
        # ----------------------------------------------------------------
        # Gradient Prediction
        sidelength = int(math.sqrt(coords.shape[0]))

        pred_dc_img = grads2img(
            pred_dc[..., 0].view(sidelength, sidelength),
            pred_dc[..., 1].view(sidelength, sidelength),
            sidelength,
        )

        pred_gradient_fig = px.imshow(pred_dc_img)

        # ----------------------------------------------------------------
        # Laplacian Prediction

        pred_dcxdc = torch.autograd.grad(
            outputs=pred_dc[..., 0],
            inputs=[coords],
            grad_outputs=torch.ones_like(pred_dc[..., 0]),
            retain_graph=True,
        )[0]
        pred_dcydc = torch.autograd.grad(
            outputs=pred_dc[..., 1],
            inputs=[coords],
            grad_outputs=torch.ones_like(pred_dc[..., 1]),
            retain_graph=True,
        )[0]

        # select dcxdcx and dcydcy (we do not need dcxdcy and dcydcx)
        pred_dcdc = torch.stack((pred_dcxdc[..., 0], pred_dcydc[..., 1]), axis=-1)

        # calculate the sum so that it acutally becomes the laplace
        pred_laplace_dcdc = pred_dcdc.sum(dim=-1)

        pred_laplacian_fig = go.Figure(
            data=go.Heatmap(
                z=pred_laplace_dcdc.cpu().view(sidelength, sidelength).detach().numpy(),
                colorscale="RdBu",
                zmid=0,
            )
        )

        pred_laplacian_fig.update_layout(
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # ----------------------------------------------------------------
        # Gradient Ground Truth

        gt_image = target.reshape(sidelength, sidelength)

        gt_dcx = scipy.ndimage.sobel(gt_image, axis=0)
        gt_dcy = scipy.ndimage.sobel(gt_image, axis=1)

        gt_dc_img = grads2img(gt_dcx, gt_dcy, sidelength)

        gt_gradient_fig = px.imshow(gt_dc_img)

        # ----------------------------------------------------------------
        # Laplacian Ground Truth

        gt_dcdc = scipy.ndimage.laplace(gt_image)

        gt_laplacian_fig = go.Figure(
            data=go.Heatmap(z=gt_dcdc, colorscale="RdBu", zmid=0)
        )

        gt_laplacian_fig.update_layout(
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        pred_gradient_fig = go.Figure()
        pred_laplacian_fig = go.Figure()
        gt_gradient_fig = go.Figure()
        gt_laplacian_fig = go.Figure()
        log.warning(
            f"Dataset has {len(shape)} dimension(s).\
            No visualization possible"
        )

    return {
        "pred_gradient_fig": pred_gradient_fig,
        "pred_laplacian_fig": pred_laplacian_fig,
        "gt_gradient_fig": gt_gradient_fig,
        "gt_laplacian_fig": gt_laplacian_fig,
    }


def grads2img(
    grads_x: Union[torch.Tensor, np.ndarray],  # gradients along x-axis
    grads_y: Union[torch.Tensor, np.ndarray],  # gradients along y-axis
    sidelength: int,  # size of the resulting image
) -> np.ndarray:  # resulting image
    """
    Convert gradients along x and y-axis into an image.

    Args:
        grads_x: Gradients along x-axis.
        grads_y: Gradients along y-axis.
        sidelength: Size of the resulting image.

    Returns:
        Image represented by an array of RGB values.
    """
    if type(grads_x) == torch.Tensor:
        grads_x = grads_x.detach().numpy()
    if type(grads_y) == torch.Tensor:
        grads_y = grads_y.detach().numpy()

    grads_a = np.arctan2(grads_y, grads_x)
    grads_m = np.hypot(grads_y, grads_x)
    grads_hsv = np.zeros((sidelength, sidelength, 3), dtype=np.float32)
    grads_hsv[:, :, 0] = (grads_a + np.pi) / (2 * np.pi)
    grads_hsv[:, :, 1] = 1.0

    nPerMin = np.percentile(grads_m, 5)
    nPerMax = np.percentile(grads_m, 95)
    grads_m = (grads_m - nPerMin) / (nPerMax - nPerMin)
    grads_m = np.clip(grads_m, 0, 1)

    grads_hsv[:, :, 2] = grads_m
    grads_rgb = hsv_to_rgb(grads_hsv)

    return grads_rgb


def calculate_spectrum(
    values: Union[torch.Tensor, np.ndarray],  # input data
    shape: Tuple[int, ...],  # shape of the input data
) -> Dict[str, go.Figure]:  # dictionary of two go.Figure objects
    """
    Calculate the spectrum (absolute and phase) of the input data.

    Args:
        values: Input data.
        shape: Shape of the input data.

    Returns:
        Dictionary with two go.Figure objects:
            - spectrum_abs_fig: Heatmap of the logarithm of the absolute spectrum.
            - spectrum_phase_fig: Heatmap of the phase spectrum.
    """
    spectrum_abs_fig = go.Figure()
    spectrum_phase_fig = go.Figure()

    if len(shape) == 3:
        sidelength = int(math.sqrt(values.shape[0]))

        spectrum = torch.fft.fft2(values.view(sidelength, sidelength))
        spectrum = torch.fft.fftshift(spectrum)

        spectrum_abs_fig = go.Figure(
            data=go.Heatmap(z=torch.log(spectrum.abs()).numpy(), colorscale="gray")
        )
        spectrum_abs_fig.update_layout(
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
        )

        spectrum_phase_fig = go.Figure(
            data=go.Heatmap(z=spectrum.angle().numpy(), colorscale="gray")
        )
        spectrum_phase_fig.update_layout(
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
        )

    elif len(shape) == 2:
        spectrum = torch.fft.fft(values)
        spectrum = torch.fft.fftshift(spectrum)
        frequencies = [-len(spectrum) // 2 + i + 1 for i in range(len(spectrum))]

        spectrum_abs_fig = go.Figure(data=go.Bar(y=spectrum.abs()))
        spectrum_abs_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                tickvals=[i for i in range(len(spectrum))],
                ticktext=frequencies,
                tickmode="array",
            ),
        )

        spectrum_phase_fig = go.Figure(data=go.Bar(y=spectrum.angle()))
        spectrum_phase_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        log.warning(
            f"Dataset has {len(shape)} dimension(s).\
            No visualization possible"
        )

    return {
        "spectrum_abs_fig": spectrum_abs_fig,
        "spectrum_phase_fig": spectrum_phase_fig,
    }
