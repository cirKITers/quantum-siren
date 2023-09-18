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

def upscaling(model, coordinates, factor):
    sidelength = math.sqrt(coordinates.shape[0])
    upscaled_sidelength = int(sidelength*factor)

    upscaled_coordinates = torch.zeros(size=(upscaled_sidelength**2, 2))
    it = 0
    for x in torch.linspace(0, coordinates.max(), upscaled_sidelength):
        for y in torch.linspace(0, coordinates.max(), upscaled_sidelength):
            upscaled_coordinates[it] = torch.tensor([x, y])
            it += 1

    model_output = model(upscaled_coordinates)

    fig = go.Figure(data =
                    go.Heatmap(z = model_output.cpu().view(upscaled_sidelength, upscaled_sidelength).detach().numpy())
                )
    fig.update_layout(
        yaxis=dict(
            scaleanchor='x',
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    mlflow.log_figure(fig, f"{factor}x_upscaled_prediction.html")

    return {
        "upscaled_image":model_output.detach().numpy(),
        "upscaled_coordinates":upscaled_coordinates
    }

def pixelwise_difference(model, coordinates, ground_truth):
    sidelength = int(math.sqrt(coordinates.shape[0]))

    model_output = model(coordinates)

    difference = model_output - ground_truth.view(sidelength**2)

    fig = go.Figure(data =
                    go.Heatmap(z = difference.cpu().view(sidelength, sidelength).detach().numpy())
                )

    fig.update_layout(
        yaxis=dict(
            scaleanchor='x',
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    mlflow.log_figure(fig, f"pixelwise_differences.html")

    return {
    }

def plot_gradients(model, coordinates, ground_truth):
    sidelength = int(math.sqrt(coordinates.shape[0]))

    coordinates.requires_grad = True
    model_output = model(coordinates)

    model_output_grad = torch.autograd.grad(outputs=model_output.sum(), inputs=coordinates, grad_outputs=None)[0] #same shape as coordinates

    pred_normed_grad = model_output_grad.norm(dim=-1)

    fig = go.Figure(data =
                    go.Heatmap(z = pred_normed_grad.cpu().view(sidelength, sidelength).detach().numpy())
                )

    fig.update_layout(
        yaxis=dict(
            scaleanchor='x',
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    mlflow.log_figure(fig, f"prediction_gradients.html")



    grads_x = scipy.ndimage.sobel(ground_truth.numpy(), axis=1).squeeze(0)[..., None]
    grads_y = scipy.ndimage.sobel(ground_truth.numpy(), axis=2).squeeze(0)[..., None]
    grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
            
    grads = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)
    normed_grad = grads.norm(dim=-1)

    fig = go.Figure(data =
                    go.Heatmap(z = normed_grad.cpu().view(sidelength, sidelength).detach().numpy())
                )

    fig.update_layout(
        yaxis=dict(
            scaleanchor='x',
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    mlflow.log_figure(fig, f"gradients.html")

    return {
    }