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
            "factor":"params:factor"
        },
        outputs={
            "image":"upscaled_image"
        }
    )

    return pipeline(
        [
            nd_upscaling
        ],
        inputs={
            "model":"model"
        },
        outputs={
            "upscaled_image":"upscaled_image"
        },
        namespace="postprocessing"
    )
