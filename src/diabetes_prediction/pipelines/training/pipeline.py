"""Training pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_all_models, select_best_model, train_all_models


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_all_models,
                inputs=["master_table", "params:training"],
                outputs="all_model_artifacts",
                name="train_all_models",
            ),
            node(
                func=evaluate_all_models,
                inputs=["master_table", "all_model_artifacts", "params:training"],
                outputs="metrics",
                name="evaluate_all_models",
            ),
            node(
                func=select_best_model,
                inputs=["all_model_artifacts", "metrics", "params:training"],
                outputs="model_artifact",
                name="select_best_model",
            ),
        ]
    )
