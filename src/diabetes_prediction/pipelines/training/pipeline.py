"""Training pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["master_table", "params:training"],
                outputs="model_artifact",
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["master_table", "model_artifact", "params:training"],
                outputs="metrics",
                name="evaluate_model",
            ),
        ]
    )
