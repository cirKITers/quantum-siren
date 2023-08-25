"""
This is a boilerplate pipeline 'postprocessing'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import upscaling

def create_pipeline(**kwargs) -> Pipeline:

    nd_upscaling = node(
        upscaling,
        inputs={
            "model":"model",
            "coordinates":"coordinates",
            "factor":"params:upscale_factor"
        },
        outputs={
            "upscaled_image":"upscaled_image",
            "upscaled_coordinates":"upscaled_coordinates"
        }
    )

    return pipeline(
        [
            nd_upscaling
        ],
        inputs={
            "coordinates":"coordinates",
            "model":"model"
        },
        outputs={
            "upscaled_image":"upscaled_image"
        },
        namespace="postprocessing"
    )
