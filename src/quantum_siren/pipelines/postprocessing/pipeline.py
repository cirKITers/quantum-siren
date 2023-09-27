"""
This is a boilerplate pipeline 'postprocessing'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import upscaling, pixelwise_difference, plot_gradients, calculate_spectrum, predict

def create_pipeline(**kwargs) -> Pipeline:
    nd_predict = node(
        predict,
        inputs={
            "model":"model",
            "coordinates":"coordinates",
        },
        outputs={
            "prediction":"prediction"
        }
    )

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
            "prediction":"prediction",
            "ground_truth":"ground_truth"
        },
        outputs={
        }
    )

    nd_plot_gradients = node(
        plot_gradients,
        inputs={
            "model":"model",
            "ground_truth":"ground_truth",
            "coordinates":"coordinates"
        },
        outputs={
        }
    )

    return pipeline(
        [
            nd_predict,
            nd_upscaling,
            nd_pixelwise_diff,
            nd_plot_gradients,
            node(
                calculate_spectrum,
                inputs={
                    "values":"ground_truth"
                },
                outputs={
                }
            )
        ],
        inputs={
            "coordinates":"coordinates",
            "model":"model",
            "ground_truth":"values"
        },
        outputs={
            "upscaled_image":"upscaled_image"
        },
        namespace="postprocessing"
    )
