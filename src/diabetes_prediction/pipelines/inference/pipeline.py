"""Inference pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from diabetes_prediction.pipelines.data_engineering.nodes import (
    engineer_features,
    transform_cleaner,
    transform_encoders,
    transform_scaler,
)

from .nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=transform_cleaner,
                inputs=["raw_inference_data", "cleaner"],
                outputs="inference_cleaned",
                name="inference_transform_cleaner",
            ),
            node(
                func=engineer_features,
                inputs=["inference_cleaned", "params:data_engineering"],
                outputs="inference_featured",
                name="inference_engineer_features",
            ),
            node(
                func=transform_encoders,
                inputs=["inference_featured", "encoders"],
                outputs="inference_encoded",
                name="inference_transform_encoders",
            ),
            node(
                func=transform_scaler,
                inputs=["inference_encoded", "scaler"],
                outputs="inference_scaled",
                name="inference_transform_scaler",
            ),
            node(
                func=predict,
                inputs=["inference_scaled", "model_artifact"],
                outputs="inference_predictions",
                name="inference_predict",
            ),
        ]
    )
