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

import math

import plotly.graph_objects as go

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
    Code from https://doi.org/10.48550/arXiv.2006.09661
    """
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = tuple(dim * [torch.linspace(domain[0], domain[1], steps=samples)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class ImageFitting(Dataset):
    """
    Code from https://doi.org/10.48550/arXiv.2006.09661
    """

    def __init__(
        self, domain, sidelength, nonlinear_coords=False, img_val_min=0, img_val_max=1
    ):
        super().__init__()
        self.sidelength = sidelength
        img = get_cameraman_tensor(sidelength)
        values = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

        # scale the model input between 0..pi
        if nonlinear_coords:
            self.coords = (torch.asin(self.coords) + torch.pi / 2) / 2

        # scale the data between -1..1 (yeah, I know that's ugly)
        # values = 2/(torch.abs(values.max() - values.min())) * (-values.min() + values) - 1
        self.values = minmax_scaler(values, -1, 1)

    def __len__(self):
        assert len(self.coords) == len(self.values)
        assert self.coords.shape[1] == 2
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.values[idx]


class CosineFitting(Dataset):
    def __init__(self, omega_d, x_domain):
        n_d = int(torch.ceil(2 * torch.max(torch.abs(x_domain)) * torch.max(omega_d)))

        log.info(f"Using {n_d} data points")

        self.coords = torch.linspace(x_domain[0], x_domain[1], n_d)

        # Formula (4) in referenced paper 2309.03279
        y = lambda x: 1 / torch.linalg.norm(omega_d) * torch.sum(torch.cos(omega_d * x))

        self.pixels = torch.stack([y(x) for x in self.coords])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return self.coords, self.pixels


def generate_image(sidelength):
    img = ImageFitting(sidelength)
    return {"img": img}


def construct_dataloader(img, batch_size, pin_memory, num_workers):
    dataloader = DataLoader(
        img, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers
    )
    return {"dataloader": dataloader}


def minmax_scaler(data, min_norm, max_norm):
    return (data - data.min()) / (data.max() - data.min()) * (
        max_norm - min_norm
    ) + min_norm


def transform_data(dataloader, nonlinear_coords, img_val_min, img_val_max):
    coordinates, values = next(iter(dataloader))
    coordinates = coordinates.reshape(-1, 2)

    # scale the model input between 0..pi
    if nonlinear_coords:
        coordinates = (torch.asin(coordinates) + torch.pi / 2) / 2

    # scale the data between -1..1 (yeah, I know that's ugly)
    # values = 2/(torch.abs(values.max() - values.min())) * (-values.min() + values) - 1
    values = minmax_scaler(values, img_val_min, img_val_max)

    return {"coordinates": coordinates, "values": values.view(-1)}


def gen_ground_truth_fig(img):
    sidelength = int(math.sqrt(img.coords.shape[0]))
    ground_truth_fig = go.Figure(
        data=go.Heatmap(
            z=img.pixels.view(sidelength, sidelength).detach().numpy(),
            colorscale="RdBu",
            zmid=0,
        )
    )
    ground_truth_fig.update_layout(
        yaxis=dict(scaleanchor="x", autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)"
    )

    # mlflow.log_figure(fig, f"ground_truth.html")

    return {"ground_truth_fig": ground_truth_fig}
