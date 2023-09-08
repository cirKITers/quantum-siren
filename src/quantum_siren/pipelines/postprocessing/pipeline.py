"""
This is a boilerplate pipeline 'postprocessing'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import upscaling, pixelwise_difference, plot_gradients

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

    nd_pixelwise_diff = node(
        pixelwise_difference,
        inputs={
            "model":"model",
            "coordinates":"coordinates",
            "ground_truth":"values"
        },
        outputs={
        }
    )

    nd_plot_gradients = node(
        plot_gradients,
        inputs={
            "model":"model",
            "coordinates":"coordinates",
            "ground_truth":"values"
        },
        outputs={
        }
    )

    return pipeline(
        [
            nd_upscaling,
            nd_pixelwise_diff,
            nd_plot_gradients
        ],
        inputs={
            "coordinates":"coordinates",
            "model":"model",
            "values":"values"
        },
        outputs={
            "upscaled_image":"upscaled_image"
        },
        namespace="postprocessing"
    )
