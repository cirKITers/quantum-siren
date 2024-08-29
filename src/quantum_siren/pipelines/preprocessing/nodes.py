import scipy
import scipy.special
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, CenterCrop
import skimage

import plotly.graph_objects as go
from typing import Tuple, Union, Dict

from quantum_siren.helpers.visualization import generate_figure

import logging

log = logging.getLogger(__name__)


def get_cameraman_tensor(sidelength: int) -> torch.Tensor:
    """
    Code from https://doi.org/10.48550/arXiv.2006.09661

    Args:
        sidelength (int): The size of the side of the image.

    Returns:
        torch.Tensor: The camera image as a tensor.
    """
    img = Image.fromarray(skimage.data.camera())
    transform = Compose(
        [
            Resize(sidelength),
            ToTensor(),
        ]
    )
    img = transform(img)
    return img


def get_coffee_tensor(sidelength: int) -> torch.Tensor:
    """
    Args:
        sidelength (int): The size of the side of the image.

    Returns:
        torch.Tensor: The camera image as a tensor.
    """
    img = Image.fromarray(skimage.data.coffee()).convert("L")
    transform = Compose(
        [
            Resize(sidelength),
            CenterCrop(sidelength),
            ToTensor(),
        ]
    )
    img = transform(img)
    return img


def get_brick_tensor(sidelength: int) -> torch.Tensor:
    """
    Args:
        sidelength (int): The size of the side of the image.

    Returns:
        torch.Tensor: The camera image as a tensor.
    """
    img = Image.fromarray(skimage.data.brick())
    transform = Compose(
        [
            Resize(sidelength),
            ToTensor(),
        ]
    )
    img = transform(img)
    return img


def get_cat_tensor(sidelength: int) -> torch.Tensor:
    """
    Args:
        sidelength (int): The size of the side of the image.

    Returns:
        torch.Tensor: The camera image as a tensor.
    """
    img = Image.fromarray(skimage.data.cat())
    transform = Compose(
        [
            Resize(sidelength),
            CenterCrop(sidelength),
            ToTensor(),
        ]
    )
    img = transform(img)
    return img


def get_mgrid(domain, samples, dim=2):
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.

    Parameters
    ----------
    domain : Tuple[float, float]
        Domain of the grid in the format (min, max)
    samples : int
        Number of samples along each dimension
    dim : int, optional
        Dimensionality of the grid, by default 2

    Returns
    -------
    torch.Tensor
        Grid tensor of shape (sidelen^dim, dim)
    """
    tensors = tuple(dim * [torch.linspace(domain[0], domain[1], steps=samples)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)  # type: torch.Tensor
    mgrid = mgrid.reshape(-1, dim)  # type: torch.Tensor
    return mgrid


class ImageFitting(Dataset):
    """
    Code from https://doi.org/10.48550/arXiv.2006.09661
    """

    def __init__(
        self,
        domain: Tuple[float, float],
        sidelength: int,
        nonlinear_coords: bool = False,
        image: str = "cameraman",
    ) -> None:
        """
        Args:
            domain (Tuple[float, float]): (min, max) of data points.
            sidelength (int): Side length of the image.
            nonlinear_coords (bool, optional): Normalize the coordinates of the input.
            Defaults to False.
        """
        super().__init__()
        self.sidelength = sidelength
        self.shape = (sidelength, sidelength, 1)
        if image == "cameraman":
            img = get_cameraman_tensor(sidelength)
        elif image == "coffee":
            img = get_coffee_tensor(sidelength)
        elif image == "brick":
            img = get_brick_tensor(sidelength)
        else:
            raise ValueError(f"Unknown image: {image}")
        values = img.permute(1, 2, 0).view(-1)
        self.coords = get_mgrid(domain, sidelength, 2)

        # scale the model input between 0..pi
        if nonlinear_coords:
            self.coords = (torch.asin(self.coords) + torch.pi / 2) / 2

        self.values = minmax_scaler(values, -1, 1)

    def __len__(self) -> int:
        assert len(self.coords) == len(self.values)
        assert self.coords.shape[1] == 2
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.values[idx]


class FourierSeriesFitting(Dataset):
    def __init__(self, domain: torch.Tensor, omega_d: torch.Tensor) -> None:
        if len(omega_d.shape) == 1:
            omega_d = omega_d.reshape(1, -1)

        # using the max of all dimensions because we want uniform sampling
        n_d = int(torch.ceil(2 * torch.max(torch.abs(domain)) * torch.max(omega_d)))
        self.shape = [n_d for _ in range(omega_d.shape[0])]
        self.shape.append(1)
        self.sidelength = n_d

        log.info(f"Using {n_d} data points on {omega_d.shape[0]} dimensions")

        # self.coords = torch.linspace(x_domain[0], x_domain[1], n_d)
        self.coords: torch.Tensor = get_mgrid(domain, n_d, dim=omega_d.shape[0])

        # Formula (4) in referenced paper 2309.03279
        def f(x: torch.Tensor) -> torch.Tensor:
            return (
                1
                / torch.linalg.norm(omega_d)
                * torch.sum(torch.cos(omega_d.T * x))  # transpose!
            )

        self.values: torch.Tensor = torch.stack([f(x) for x in self.coords])

    def __len__(self) -> int:
        assert len(self.coords) == len(self.values)
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.values[idx]


class HelmholtzSource(Dataset):
    def __init__(self, domain: torch.Tensor, sidelength: int, wavenumber: int) -> None:
        self.sidelength = sidelength
        self.shape = (self.sidelength, self.sidelength, 2)

        self.coords: torch.Tensor = get_mgrid(domain, sidelength, dim=2)
        self.sigma = 1e-4
        self.wavenumber = wavenumber

        x0 = 0
        y0 = 0

        # Formula (4) in referenced paper 2309.03279
        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return scipy.special.hankel2(
                0, self.wavenumber * torch.sqrt((x - x0) ** 2 + (y - y0) ** 2) + 1e-6
            )

        field = torch.stack([f(x[0], x[1]) for x in self.coords])

        self.values: torch.Tensor = torch.stack([field.real, field.imag]).view(-1, 2)

    def __len__(self) -> int:
        assert len(self.coords) == len(self.values)
        assert self.coords.shape[1] == 2
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.values[idx]


def generate_dataset(
    mode: str,
    domain: Tuple[float, float],
    scale_domain_by_pi: bool,
    sidelength: int,
    nonlinear_coords: bool,
    omega: Tuple[float, ...],
    wavenumber: int,
) -> Dict[str, Dataset]:
    """
    Generate a dataset based on the input parameters.

    Args:
        mode (str): The type of dataset to generate.
        domain (Tuple[float, float]): The domain of the dataset.
        scale_domain_by_pi (bool): If true, scale the domain by pi.
        sidelength (int): The sidelength of the dataset.
        nonlinear_coords (bool): If true, use nonlinear coordinates.
        omega (Tuple[float, ...]): The frequency of the dataset.

    Returns:
        Dict[str, Dataset]: A dictionary with the dataset.

    """
    if scale_domain_by_pi:
        domain = torch.tensor(
            [domain[0] * torch.pi, domain[1] * torch.pi], dtype=torch.float
        )
    else:
        domain = torch.tensor([domain[0], domain[1]], dtype=torch.float)

    omega = torch.tensor(omega, dtype=torch.float)

    if mode == "fourierSeries":
        dataset = FourierSeriesFitting(domain, omega)
    elif mode == "helmholtz":
        dataset = HelmholtzSource(domain, sidelength=sidelength, wavenumber=wavenumber)
    elif mode.startswith("image_"):
        dataset = ImageFitting(
            domain,
            sidelength,
            nonlinear_coords=nonlinear_coords,
            image=mode.replace("image_", ""),
        )
    return {"dataset": dataset}


def construct_dataloader(dataset: Dataset, batch_size: int) -> Dict[str, DataLoader]:
    """
    Construct a dataloader from a dataset.

    Args:
        dataset (Dataset): The dataset to construct the dataloader from.
        batch_size (int): The size of each batch.

    Returns:
        Dict[str, DataLoader]: A dictionary with the constructed dataloader.
    """
    if batch_size > len(dataset.coords) or batch_size < 1:
        log.warning(
            f"Adjusting batch size to {len(dataset.coords)} (was {batch_size} before)"
        )
        batch_size = len(dataset.coords)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    return {"dataloader": dataloader}


def minmax_scaler(data: torch.Tensor, min_norm: float, max_norm: float) -> torch.Tensor:
    """
    Scales data between min_norm and max_norm using min-max normalization.

    Args:
        data (torch.Tensor): The data to be scaled.
        min_norm (float): The minimum value of the scaled data.
        max_norm (float): The maximum value of the scaled data.

    Returns:
        torch.Tensor: The scaled data.
    """
    return (data - data.min()) / (data.max() - data.min()) * (
        max_norm - min_norm
    ) + min_norm


def extract_data(
    dataset: Dataset,
) -> Dict[str, Union[torch.Tensor, Tuple[int, int, int]]]:
    """
    Extracts data from a dataset.

    Args:
        dataset (Dataset): The dataset to extract the data from.

    Returns:
        Dict[str, Union[torch.Tensor, Tuple[int, int, int]]]:
        A dictionary containing the extracted data.
            The keys are:
                - coords (torch.Tensor): The coordinates of the data.
                - target (torch.Tensor): The target values of the data.
                - shape (Tuple[int, int, int]): The shape of the data.
    """
    return {
        "coords": dataset.coords,
        "target": dataset.values,
        "shape": dataset.shape,
    }


def gen_ground_truth_fig(dataset: Dataset) -> Dict[str, go.Figure]:
    """
    Generates a ground truth figure based on the dataset.

    Args:
        dataset (Dataset): The dataset from which to generate the figure.

    Returns:
        Dict[str, go.Figure]: A dictionary containing the generated figure.
            The key is "ground_truth_fig".
    """
    return {
        "ground_truth_fig": generate_figure(
            shape=dataset.shape,
            coords=dataset.coords,
            values=dataset.values,
            sidelength=dataset.sidelength,
        )
    }
