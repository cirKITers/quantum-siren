"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import generate_image, construct_dataloader, transform_data

def create_pipeline(**kwargs) -> Pipeline:
    nd_generate_img = node(
        generate_image,
        inputs={
            "sidelength":"params:sidelength"
        },
        outputs={
            "img":"img"
        }
    )
    nd_construct_dataloader = node(
        construct_dataloader,
        inputs={
            "img":"img",
            "batch_size":"params:batch_size",
            "pin_memory":"params:pin_memory",
            "num_workers":"params:num_workers"
        },
        outputs={
            "dataloader":"dataloader"
        }
    )
    nd_transform_data = node(
        transform_data,
        inputs={
            "dataloader":"dataloader",
            "nonlinear_coords":"params:nonlinear_coords",
            "img_val_min":"params:img_val_min",
            "img_val_max":"params:img_val_max"
        },
        outputs={
            "coordinates":"coordinates",
            "values":"values"
        }
    )

    return pipeline(
        [
            nd_generate_img,
            nd_construct_dataloader,
            nd_transform_data
        ],
        outputs={
            "coordinates":"coordinates",
            "values":"values"
        },
        namespace="preprocessing"
    )
