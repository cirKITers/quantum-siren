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
    upscaled_coordinates = torch.tensor(rescale(coordinates.numpy(), [factor**2, 1]))
    sidelength = int(math.sqrt(upscaled_coordinates.shape[0]))

    model_output = torch.zeros(size=[upscaled_coordinates.shape[0],])
    
    for i, coord in enumerate(upscaled_coordinates):
        # out[i] = torch.mean(torch.stack(circuit(params, coord)), axis=0)
        model_output[i] = model.predict(coord)[-1]

    fig = go.Figure(data =
                    go.Heatmap(z = model_output.cpu().view(sidelength, sidelength).detach().numpy())
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
        "upscaled_image":model_output,
        "upscaled_coordinates":upscaled_coordinates
    }