"""
This is a boilerplate pipeline 'postprocessing'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import (
    upscaling,
    pixelwise_difference,
    plot_gradients,
    calculate_spectrum,
    predict,
)


def create_pipeline(**kwargs) -> Pipeline:
    nd_predict = node(
        predict,
        inputs={
            "model": "model",
            "coords": "coords",
        },
        outputs={"prediction": "prediction"},
    )

    nd_upscaling = node(
        upscaling,
        inputs={
            "model": "model",
            "coords": "coords",
            "factor": "params:upscale_factor",
            "shape": "shape",
        },
        outputs={
            "pred_upscaled_fig": "pred_upscaled_fig",
            "upscaled_image": "upscaled_image",
            "upscaled_coordinates": "upscaled_coords",
        },
    )

    nd_pixelwise_diff = node(
        pixelwise_difference,
        inputs={"prediction": "prediction", "target": "target", "shape": "shape"},
        outputs={"pixelwise_diff_fig": "pixelwise_diff_fig"},
    )

    nd_plot_gradients = node(
        plot_gradients,
        inputs={
            "model": "model",
            "target": "target",
            "coords": "coords",
            "shape": "shape",
        },
        outputs={
            "pred_gradient_fig": "pred_gradient_fig",
            "pred_laplacian_fig": "pred_laplacian_fig",
            "gt_gradient_fig": "gt_gradient_fig",
            "gt_laplacian_fig": "gt_laplacian_fig",
        },
    )

    return pipeline(
        [
            nd_predict,
            nd_upscaling,
            nd_pixelwise_diff,
            nd_plot_gradients,
            node(
                calculate_spectrum,
                inputs={"values": "target", "shape": "shape"},
                outputs={
                    "spectrum_abs_fig": "gt_spectrum_abs_fig",
                    "spectrum_phase_fig": "gt_spectrum_phase_fig",
                },
            ),
            node(
                calculate_spectrum,
                inputs={"values": "prediction", "shape": "shape"},
                outputs={
                    "spectrum_abs_fig": "pred_spectrum_abs_fig",
                    "spectrum_phase_fig": "pred_spectrum_phase_fig",
                },
            ),
        ],
        inputs={
            "coords": "coords",
            "model": "model",
            "target": "target",
            "shape": "shape",
        },
        outputs={"upscaled_image": "upscaled_image"},
        namespace="postprocessing",
    )
