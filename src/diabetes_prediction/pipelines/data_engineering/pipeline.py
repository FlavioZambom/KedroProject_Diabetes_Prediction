"""Data engineering pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    clean_data,
    engineer_features,
    fit_encoders,
    fit_scaler,
    split_data,
    transform_encoders,
    transform_scaler,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=["raw_modelling_data", "params:data_engineering"],
                outputs="cleaned_data",
                name="clean_data",
            ),
            node(
                func=engineer_features,
                inputs=["cleaned_data", "params:data_engineering"],
                outputs="featured_data",
                name="engineer_features",
            ),
            node(
                func=split_data,
                inputs=["featured_data", "params:data_engineering"],
                outputs="split_data",
                name="split_data",
            ),
            node(
                func=fit_encoders,
                inputs=["split_data", "params:data_engineering"],
                outputs="encoders",
                name="fit_encoders",
            ),
            node(
                func=transform_encoders,
                inputs=["split_data", "encoders"],
                outputs="encoded_data",
                name="transform_encoders",
            ),
            node(
                func=fit_scaler,
                inputs=["encoded_data", "params:data_engineering"],
                outputs="scaler",
                name="fit_scaler",
            ),
            node(
                func=transform_scaler,
                inputs=["encoded_data", "scaler"],
                outputs="master_table",
                name="transform_scaler",
            ),
        ]
    )
