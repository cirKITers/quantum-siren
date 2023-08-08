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



def get_cameraman_tensor(sidelength):
    '''
    Code from https://doi.org/10.48550/arXiv.2006.09661
    '''
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

def get_mgrid(sidelen, dim=2):
    '''
    Code from https://doi.org/10.48550/arXiv.2006.09661
    '''
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class ImageFitting(Dataset):
    '''
    Code from https://doi.org/10.48550/arXiv.2006.09661
    '''
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels

def generate_image(sidelength):
    img = ImageFitting(sidelength)
    return {
        "img":img
    }

def construct_dataloader(img, batch_size, pin_memory, num_workers):
    dataloader = DataLoader(img, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    return {
        "dataloader":dataloader
    }

def minmax_scaler(data, min_norm, max_norm):
    return (data - data.min())/(data.max() - data.min())*(max_norm - min_norm) + min_norm

def transform_data(dataloader, nonlinear_coords, img_val_min, img_val_max):
    coordinates, values = next(iter(dataloader))

    # scale the model input between 0..pi
    if nonlinear_coords:
        coordinates[0] = (torch.asin(coordinates[0])+torch.pi/2)/2

    # scale the data between -1..1 (yeah, I know that's ugly)
    # values = 2/(torch.abs(values.max() - values.min())) * (-values.min() + values) - 1
    values = minmax_scaler(values, img_val_min, img_val_max)

    return {
        "coordinates": coordinates,
        "values": values
    }