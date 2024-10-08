"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import training


def create_pipeline(**kwargs) -> Pipeline:
    nd_training = node(
        training,
        inputs={
            "n_layers": "params:n_layers",
            "n_qubits": "params:n_qubits",
            "circuit_type": "params:circuit_type",
            "data_reupload": "params:data_reupload",
            "learning_rate": "params:learning_rate",
            "shots": "params:shots",
            "report_figure_every_n_steps": "params:report_figure_every_n_steps",
            "optimizer": "params:optimizer",
            "output_qubit": "params:output_qubit",
            "initialization": "params:initialization",
            "loss": "params:loss",
            "dataloader": "dataloader",
            "steps": "params:steps",
            "seed": "params:seed",
            "early_stopping": "params:early_stopping",
        },
        outputs={"model": "model"},
    )
    return pipeline(
        [nd_training],
        inputs={"dataloader": "dataloader"},
        outputs={"model": "model"},
        namespace="training",
    )
