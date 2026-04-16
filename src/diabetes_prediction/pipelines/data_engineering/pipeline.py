"""Data engineering pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    engineer_features,
    fit_cleaner,
    fit_encoders,
    fit_scaler,
    split_data,
    transform_cleaner,
    transform_encoders,
    transform_scaler,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["raw_modelling_data", "params:data_engineering"],
                outputs="split_data",
                name="split_data",
            ),
            node(
                func=fit_cleaner,
                inputs=["split_data", "params:data_engineering"],
                outputs="cleaner",
                name="fit_cleaner",
            ),
            node(
                func=transform_cleaner,
                inputs=["split_data", "cleaner"],
                outputs="cleaned_data",
                name="transform_cleaner",
            ),
            node(
                func=engineer_features,
                inputs=["cleaned_data", "params:data_engineering"],
                outputs="featured_data",
                name="engineer_features",
            ),
            node(
                func=fit_encoders,
                inputs=["featured_data", "params:data_engineering"],
                outputs="encoders",
                name="fit_encoders",
            ),
            node(
                func=transform_encoders,
                inputs=["featured_data", "encoders"],
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
