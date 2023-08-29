"""
This is a boilerplate pipeline 'postprocessing'
generated using Kedro 0.18.12
"""

from skimage.transform import rescale
import torch
import mlflow
import plotly.graph_objects as go 
import math


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