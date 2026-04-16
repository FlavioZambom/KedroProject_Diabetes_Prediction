"""Refit pipeline definition."""

from kedro.pipeline import Node, Pipeline

from .nodes import refit_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=refit_model,
                inputs=["master_table", "model_artifact", "params:refit"],
                outputs="production_model",
                name="refit_model",
            ),
        ]
    )
