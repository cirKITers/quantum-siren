"""
This is a boilerplate pipeline 'postprocessing'
generated using Kedro 0.18.12
"""

from skimage.transform import rescale
import torch
import mlflow
import plotly.graph_objects as go
import math
import scipy
import numpy as np
import matplotlib.colors as colors
import plotly.express as px
import logging

log = logging.getLogger(__name__)


def predict(model, coords):
    model_output = model(coords)

    return {"prediction": model_output.view(-1).detach()}


def upscaling(model, coords, factor, shape):
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

    return {
        "pred_upscaled_fig": pred_upscaled_fig,
        "upscaled_image": model_output,
        "upscaled_coordinates": upscaled_coords,
    }


def pixelwise_difference(prediction, target, shape):
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

    return {"pixelwise_diff_fig": pixelwise_diff_fig}


def plot_gradients(model, target, coords, shape):
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

        pred_gradients_fig = go.Figure(
            data=[
                go.Scatter(
                    x=coords.detach().flatten(),
                    y=pred.detach(),
                    mode="lines",
                    name="Prediction",
                ),
            ]
        )
        pred_gradients_fig.update_layout(
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

        gt_gradients_fig = go.Figure(
            data=[
                go.Scatter(
                    x=coords.detach().flatten(),
                    y=gt_dc,
                    mode="lines",
                    name="Prediction",
                ),
            ]
        )
        gt_gradients_fig.update_layout(
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

        pred_gradients_fig = px.imshow(pred_dc_img)

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

        gt_gradients_fig = px.imshow(gt_dc_img)

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
        raise ("Unknown dataset type")
    return {
        "pred_gradients_fig": pred_gradients_fig,
        "pred_laplacian_fig": pred_laplacian_fig,
        "gt_gradients_fig": gt_gradients_fig,
        "gt_laplacian_fig": gt_laplacian_fig,
    }


def grads2img(grads_x, grads_y, sidelength):
    """
    Thankfully adapted from https://github.com/vsitzmann/siren/blob/master/dataio.py#L55
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
    grads_rgb = colors.hsv_to_rgb(grads_hsv)

    return grads_rgb


def calculate_spectrum(values, shape):
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
        log.warning("Unknown shape: " + str(shape))
    return {
        "spectrum_abs_fig": spectrum_abs_fig,
        "spectrum_phase_fig": spectrum_phase_fig,
    }
