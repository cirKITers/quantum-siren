"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.12
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import skimage

import plotly.graph_objects as go
from plotly.express import colors

import logging

log = logging.getLogger(__name__)


def get_cameraman_tensor(sidelength):
    """
    Code from https://doi.org/10.48550/arXiv.2006.09661
    """
    img = Image.fromarray(skimage.data.camera())
    transform = Compose(
        [
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
        ]
    )
    img = transform(img)
    return img


def get_mgrid(domain, samples, dim=2):
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.

    Parameters
    ----------
    sidelen : int
        Side length of the grid
    dim : int, optional
        Dimensionality of the grid, by default 2

    Returns
    -------
    torch.Tensor
        Grid tensor of shape (sidelen^dim, dim)
    """
    tensors = tuple(dim * [torch.linspace(domain[0], domain[1], steps=samples)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class ImageFitting(Dataset):
    """
    Code from https://doi.org/10.48550/arXiv.2006.09661
    """

    def __init__(self, domain, sidelength, nonlinear_coords=False):
        super().__init__()
        self.sidelength = sidelength
        self.shape = (sidelength, sidelength, 1)
        img = get_cameraman_tensor(sidelength)
        values = img.permute(1, 2, 0).view(-1)
        self.coords = get_mgrid(domain, sidelength, 2)

        # scale the model input between 0..pi
        if nonlinear_coords:
            self.coords = (torch.asin(self.coords) + torch.pi / 2) / 2

        self.values = minmax_scaler(values, -1, 1)

    def __len__(self):
        assert len(self.coords) == len(self.values)
        assert self.coords.shape[1] == 2
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.values[idx]


class CosineFitting(Dataset):
    def __init__(self, domain, omega_d):
        if len(omega_d.shape) == 1:
            omega_d = omega_d.reshape(1, -1)

        # using the max of all dimensions because we want uniform sampling
        n_d = int(torch.ceil(2 * torch.max(torch.abs(domain)) * torch.max(omega_d)))
        self.shape = [n_d for _ in range(omega_d.shape[0])]
        self.shape.append(1)
        self.sidelength = n_d

        log.info(f"Using {n_d} data points on {omega_d.shape[0]} dimensions")

        # self.coords = torch.linspace(x_domain[0], x_domain[1], n_d)
        self.coords = get_mgrid(domain, n_d, dim=omega_d.shape[0])

        # Formula (4) in referenced paper 2309.03279
        def y(x):
            return (
                1
                / torch.linalg.norm(omega_d)
                * torch.sum(torch.cos(omega_d.T * x))  # transpose!
            )

        self.values = torch.stack([y(x) for x in self.coords])

    def __len__(self):
        assert len(self.coords) == len(self.values)
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.values[idx]


def generate_dataset(
    mode,
    domain,
    scale_domain_by_pi,
    sidelength,
    nonlinear_coords,
    omega,
):
    if scale_domain_by_pi:
        domain = torch.tensor(
            [domain[0] * torch.pi, domain[1] * torch.pi], dtype=torch.float
        )
    else:
        domain = torch.tensor([domain[0], domain[1]], dtype=torch.float)

    omega = torch.tensor(omega, dtype=torch.float)

    if mode == "image":
        dataset = ImageFitting(domain, sidelength, nonlinear_coords)
    elif mode == "cosine":
        dataset = CosineFitting(domain, omega)
    return {"dataset": dataset}


def construct_dataloader(dataset, batch_size):
    if batch_size > len(dataset.coords):
        log.warning(
            f"Adjusting batch size to {len(dataset.coords)} (was {batch_size} before)"
        )
    bs = min(batch_size, len(dataset.coords))
    dataloader = DataLoader(dataset, batch_size=bs, pin_memory=True)
    return {"dataloader": dataloader}


def minmax_scaler(data, min_norm, max_norm):
    return (data - data.min()) / (data.max() - data.min()) * (
        max_norm - min_norm
    ) + min_norm


def extract_data(dataset):
    return {
        "coords": dataset.coords,
        "target": dataset.values,
        "shape": dataset.shape,
    }


def gen_ground_truth_fig(dataset):
    if len(dataset.shape) == 4:

        def add_opacity(colorscale):
            for color in colorscale:
                rgb = colors.hex_to_rgb(color[1])
                color[1] = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {max(0.1, color[0])})"

            return colorscale

        fig = go.Figure(
            data=go.Scatter3d(
                x=dataset.coords[:, 0],
                y=dataset.coords[:, 1],
                z=dataset.coords[:, 2],
                mode="markers",
                marker=dict(
                    size=20 * dataset.values.abs() + 1.0,
                    color=dataset.values,  # set color to an array/list of desired values
                    colorscale=add_opacity(
                        colors.get_colorscale("Plasma")
                    ),  # choose a colorscale
                    opacity=1.0,
                ),
            )
        )
        fig.update_layout(
            template="simple_white",
        )
    elif len(dataset.shape) == 3:
        sidelength = dataset.sidelength
        fig = go.Figure(
            data=go.Heatmap(
                z=dataset.values.view(sidelength, sidelength).detach().numpy(),
                colorscale="RdBu",
                zmid=0,
            )
        )
        fig.update_layout(
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        fig = go.Figure(
            data=go.Scatter(
                x=dataset.coords.detach().numpy(),
                y=dataset.values.detach().numpy(),
                mode="lines",
            )
        )

    # mlflow.log_figure(fig, f"ground_truth.html")

    return {"ground_truth_fig": fig}
