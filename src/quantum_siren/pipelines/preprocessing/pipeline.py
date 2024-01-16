"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    generate_dataset,
    construct_dataloader,
    extract_data,
    gen_ground_truth_fig,
)


def create_pipeline(**kwargs) -> Pipeline:
    nd_generate_dataset = node(
        generate_dataset,
        inputs={
            "mode": "params:mode",
            "domain": "params:domain",
            "scale_domain_by_pi": "params:scale_domain_by_pi",
            "sidelength": "params:sidelength",
            "nonlinear_coords": "params:nonlinear_coords",
            "omega": "params:omega",
        },
        outputs={"dataset": "dataset"},
    )
    nd_gen_ground_truth_fig = node(
        gen_ground_truth_fig,
        inputs={
            "dataset": "dataset",
        },
        outputs={"ground_truth_fig": "ground_truth_fig"},
    )
    nd_construct_dataloader = node(
        construct_dataloader,
        inputs={
            "dataset": "dataset",
            "batch_size": "params:batch_size",
        },
        outputs={"dataloader": "dataloader"},
    )
    nd_extract_data = node(
        extract_data,
        inputs={
            "dataset": "dataset",
        },
        outputs={"coordinates": "coordinates", "values": "values"},
    )

    return pipeline(
        [
            nd_generate_dataset,
            nd_gen_ground_truth_fig,
            nd_construct_dataloader,
            nd_extract_data,
        ],
        outputs={
            "coordinates": "coordinates",
            "values": "values",
            "dataloader": "dataloader",
        },
        namespace="preprocessing",
    )
