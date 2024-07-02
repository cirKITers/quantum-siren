"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from quantum_siren.pipelines.preprocessing.pipeline import (
    create_pipeline as create_preprocessing_pipeline,
)
from quantum_siren.pipelines.training.pipeline import (
    create_pipeline as create_training_pipeline,
)
from quantum_siren.pipelines.postprocessing.pipeline import (
    create_pipeline as create_postprocessing_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "__default__": create_preprocessing_pipeline()
        + create_training_pipeline()
        + create_postprocessing_pipeline(),
        "slurm": create_preprocessing_pipeline() + create_training_pipeline(),
    }
    return pipelines
