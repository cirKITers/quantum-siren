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
                    go.Heatmap(z = model_output.cpu().view(upscaled_sidelength, upscaled_sidelength).detach().numpy(), colorscale='RdBu', zmid=0)
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
                    go.Heatmap(z = difference.cpu().view(sidelength, sidelength).detach().numpy(), colorscale='RdBu', zmid=0)
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

    #----------------------------------------------------------------    
    # Gradient Prediction

    model_output_dc = torch.autograd.grad(outputs=model_output.sum(), inputs=coordinates, grad_outputs=None, create_graph=True)[0] #same shape as coordinates

    pred_normed_dc = model_output_dc.norm(dim=-1)

    fig = go.Figure(data =
                    go.Heatmap(z = pred_normed_dc.cpu().view(sidelength, sidelength).detach().numpy(), colorscale='RdBu', zmid=0)
                )

    fig.update_layout(
        yaxis=dict(
            scaleanchor='x',
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    mlflow.log_figure(fig, f"prediction_gradients.html")

    #----------------------------------------------------------------    
    # Laplacian Prediction

    model_output_dcdc = torch.autograd.grad(outputs=pred_normed_dc.sum(), inputs=coordinates, grad_outputs=None)[0]

    pred_normed_dcdc = model_output_dcdc.norm(dim=-1)
    
    fig = go.Figure(data =
                    go.Heatmap(z = pred_normed_dcdc.cpu().view(sidelength, sidelength).detach().numpy(), colorscale='RdBu', zmid=0)
                )

    fig.update_layout(
        yaxis=dict(
            scaleanchor='x',
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    mlflow.log_figure(fig, f"prediction_laplacian.html")


    #----------------------------------------------------------------    
    # Gradient Ground Truth

    gt_image = ground_truth.reshape(sidelength, sidelength)

    grads_x = scipy.ndimage.sobel(gt_image, axis=0)
    grads_y = scipy.ndimage.sobel(gt_image, axis=1)
    grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
            
    grads = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)
    normed_grad = grads.norm(dim=-1)

    fig = go.Figure(data =
                    go.Heatmap(z = normed_grad.cpu().view(sidelength, sidelength).detach().numpy(), colorscale='RdBu', zmid=0)
                )

    fig.update_layout(
        yaxis=dict(
            scaleanchor='x',
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    mlflow.log_figure(fig, f"gradients.html")

    #----------------------------------------------------------------    
    # Laplacian Ground Truth

    laplacian = scipy.ndimage.laplace(gt_image)

    fig = go.Figure(data =
                    go.Heatmap(z = laplacian, colorscale='RdBu', zmid=0)
                )

    fig.update_layout(
        yaxis=dict(
            scaleanchor='x',
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    mlflow.log_figure(fig, f"laplacian.html")

    return {
    }

    