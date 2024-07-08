import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express import colors

import logging

log = logging.getLogger(__name__)


def add_opacity(colorscale):
    for color in colorscale:
        rgb = colors.hex_to_rgb(color[1])
        color[1] = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {max(0.1, color[0])})"

    return colorscale


def generate_figure(shape, coords, values, sidelength=None):
    if len(shape) == 4:

        fig = go.Figure(
            data=go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=20 * values.abs() + 1.0,
                    color=values,
                    colorscale="Plasma",
                    opacity=1.0,
                ),
            )
        )
        fig.update_layout(
            template="simple_white",
        )
    elif len(shape) == 3:
        sidelength = sidelength
        if len(values.shape) == 1:
            values = values.view(-1, 1)
        fig = make_subplots(rows=1, cols=shape[2])
        for c in range(shape[2]):
            fig.add_trace(
                go.Heatmap(
                    z=values[:, c].view(sidelength, sidelength).detach().numpy(),
                    colorscale="RdBu",
                    zmid=0,
                ),
                1,
                c + 1,  # always start from 1
            )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
    elif len(shape) == 2:
        fig = go.Figure(
            data=go.Scatter(
                x=coords.detach().numpy(),
                y=values.detach().numpy(),
                mode="lines",
            )
        )
    else:
        log.warning(f"Dataset has {len(shape)} dimension(s). No visualization possible")

    return fig
