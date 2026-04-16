"""Project pipeline registry."""

from diabetes_prediction.pipelines.data_engineering.pipeline import (
    create_pipeline as create_data_engineering_pipeline,
)
from diabetes_prediction.pipelines.inference.pipeline import (
    create_pipeline as create_inference_pipeline,
)
from diabetes_prediction.pipelines.refit.pipeline import (
    create_pipeline as create_refit_pipeline,
)
from diabetes_prediction.pipelines.training.pipeline import (
    create_pipeline as create_training_pipeline,
)


def register_pipelines():
    """Register the project's pipelines."""
    data_engineering = create_data_engineering_pipeline()
    training = create_training_pipeline()
    inference = create_inference_pipeline()
    refit = create_refit_pipeline()

    return {
        "data_engineering": data_engineering,
        "training": training,
        "inference": inference,
        "refit": refit,
        "__default__": data_engineering + training,
    }
