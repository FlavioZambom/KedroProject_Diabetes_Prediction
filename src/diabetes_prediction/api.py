"""FastAPI application exposing the Diabetes Prediction Kedro pipelines."""

import json
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pydantic import BaseModel, Field

from diabetes_prediction.pipelines.data_engineering.nodes import (
    clean_data,
    engineer_features,
    transform_encoders,
    transform_scaler,
)
from diabetes_prediction.pipelines.inference.nodes import predict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
DATA_DIR = PROJECT_ROOT / "data"

ARTIFACTS: dict[str, Path] = {
    "model": DATA_DIR / "06_models" / "model_artifact.pkl",
    "production_model": DATA_DIR / "06_models" / "production_model.pkl",
    "encoders": DATA_DIR / "03_primary" / "encoders.pkl",
    "scaler": DATA_DIR / "03_primary" / "scaler.pkl",
    "metrics": DATA_DIR / "08_reporting" / "metrics.json",
}

DATA_ENGINEERING_PARAMS: dict[str, Any] = {
    "target_column": "OUTCOME",
    "zero_replacement_columns": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    "outlier_q1": 0.05,
    "outlier_q3": 0.95,
    "test_size": 0.30,
    "random_state": 17,
    "split_to_fit": ["train"],
}

state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _reload_pickle_artifacts() -> None:
    """(Re)load every pickle artifact that exists on disk into *state*."""
    for key, path in ARTIFACTS.items():
        if key == "metrics":
            continue
        if path.exists():
            state[key] = _load_pickle(path)
            logger.info("api: artifact loaded  key=%s  path=%s", key, path.name)
        else:
            logger.warning("api: artifact not found (run pipelines first)  path=%s", path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api: startup — loading artifacts from disk")
    _reload_pickle_artifacts()
    logger.info(
        "api: startup complete  loaded=%s",
        [k for k in state],
    )
    yield
    logger.info("api: shutdown — clearing state")
    state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Diabetes Prediction API",
    description=(
        "Exposes the Kedro pipelines for diabetes prediction. "
        "Run `data_engineering` + `training` before calling `/predict`."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DiabetesInput(BaseModel):
    Pregnancies: float = Field(..., ge=0, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, description="Plasma glucose concentration (mg/dL)")
    BloodPressure: float = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., ge=0, description="2-Hour serum insulin (µU/mL)")
    BMI: float = Field(..., ge=0, description="Body mass index (kg/m²)")
    DiabetesPedigreeFunction: float = Field(..., ge=0, description="Diabetes pedigree function")
    Age: float = Field(..., gt=0, description="Age in years")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Pregnancies": 6,
                    "Glucose": 148,
                    "BloodPressure": 72,
                    "SkinThickness": 35,
                    "Insulin": 0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 50,
                }
            ]
        }
    }


class DiabetesPrediction(BaseModel):
    prediction: int = Field(..., description="0 = no diabetes, 1 = diabetes")
    probability: float = Field(..., description="Probability of diabetes (0–1)")
    label: str = Field(..., description="Human-readable result")


class TuningOptions(BaseModel):
    """Hyperparameter tuning overrides for the training pipeline run."""
    enabled: bool = Field(
        default=False,
        description="Enable hyperparameter search (GridSearchCV / RandomizedSearchCV).",
    )
    method: str = Field(
        default="random",
        pattern="^(random|grid)$",
        description="Search strategy: ``random`` (RandomizedSearchCV) or ``grid`` (GridSearchCV).",
    )
    scoring: str = Field(
        default="f1",
        description="Sklearn scoring metric used to rank candidates (e.g. ``f1``, ``roc_auc``).",
    )
    n_iter: int = Field(
        default=20,
        ge=1,
        description="Number of candidates sampled in random search (ignored for grid search).",
    )
    cv: int = Field(default=5, ge=2, description="Number of cross-validation folds.")


class PipelineRunRequest(BaseModel):
    """Optional body for POST /pipelines/{pipeline_name}/run."""
    tuning: TuningOptions | None = Field(
        default=None,
        description=(
            "Hyperparameter tuning options. "
            "Only applied when ``pipeline_name`` is ``training`` or ``__default__``."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tuning": {
                        "enabled": True,
                        "method": "random",
                        "scoring": "f1",
                        "n_iter": 30,
                        "cv": 5,
                    }
                }
            ]
        }
    }


class PipelineRunResponse(BaseModel):
    pipeline_name: str
    status: str
    message: str
    tuning_applied: bool = Field(
        default=False,
        description="Whether hyperparameter tuning was applied in this run.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_kedro_pipeline(pipeline_name: str, extra_params: dict[str, Any] | None = None) -> None:
    """Bootstrap Kedro and run *pipeline_name*, optionally overriding parameters."""
    logger.info(
        "kedro: running pipeline  name=%s  extra_params=%s",
        pipeline_name,
        extra_params,
    )
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(project_path=PROJECT_ROOT, runtime_params=extra_params or {}) as session:
        session.run(pipeline_names=[pipeline_name])
    logger.info("kedro: pipeline finished  name=%s", pipeline_name)


def _infer(raw_input: DiabetesInput, model_key: str = "model") -> dict[str, Any]:
    """Run the full inference chain using preloaded artifacts.

    Args:
        raw_input: Validated patient data from the request body.
        model_key: Key in *state* to use (``"model"`` or ``"production_model"``).

    Returns:
        Dict with ``prediction``, ``probability``, and ``label``.

    Raises:
        HTTPException 503: When the required artifact is not loaded.
    """
    if model_key not in state:
        hint = (
            "Run the 'data_engineering' + 'training' pipelines first."
            if model_key == "model"
            else "Run the 'refit' pipeline first."
        )
        logger.warning("api: artifact missing  key=%s", model_key)
        raise HTTPException(
            status_code=503,
            detail=f"Artifact '{model_key}' not loaded. {hint}",
        )
    if "encoders" not in state or "scaler" not in state:
        logger.warning("api: encoder/scaler not loaded")
        raise HTTPException(
            status_code=503,
            detail="Encoder/scaler artifacts not loaded. Run the 'data_engineering' pipeline first.",
        )

    logger.debug("api: inference started  model_key=%s  input=%s", model_key, raw_input.model_dump())

    df = pd.DataFrame([raw_input.model_dump()])
    df_clean = clean_data(df, DATA_ENGINEERING_PARAMS)
    df_feat = engineer_features(df_clean, DATA_ENGINEERING_PARAMS)
    df_enc = transform_encoders(df_feat, state["encoders"])
    df_scaled = transform_scaler(df_enc, state["scaler"])
    results = predict(df_scaled, state[model_key])

    prediction = int(results["prediction"].iloc[0])
    probability = float(results["probability"].iloc[0])
    label = "Diabetes" if prediction == 1 else "No Diabetes"

    logger.info(
        "api: inference complete  model_key=%s  prediction=%d  probability=%.4f  label=%s",
        model_key, prediction, probability, label,
    )
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "label": label,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health() -> dict[str, Any]:
    loaded = [k for k in state]
    missing = [k for k, p in ARTIFACTS.items() if k != "metrics" and k not in state]
    logger.debug("api: health check  loaded=%s  missing=%s", loaded, missing)
    return {
        "status": "ok",
        "artifacts_loaded": {k: True for k in state if k != "metrics"},
        "artifacts_missing": missing,
    }


@app.get("/pipelines", tags=["Pipelines"])
def list_pipelines() -> dict[str, Any]:
    return {
        "pipelines": [
            {
                "name": "data_engineering",
                "description": "Clean raw data, engineer features, fit encoders & scaler, produce master_table.",
            },
            {
                "name": "training",
                "description": (
                    "Train all configured models (RandomForest, LogisticRegression, CatBoost), "
                    "evaluate each on the test split, select the best by selection_metric. "
                    "Supports optional hyperparameter tuning via the request body."
                ),
            },
            {
                "name": "inference",
                "description": "Run predictions on raw_inference_data using the saved model; writes inference_predictions.csv.",
            },
            {
                "name": "refit",
                "description": "Retrain the validated model on all data (train + test) for production use; saves production_model.",
            },
            {
                "name": "__default__",
                "description": "Run data_engineering + training end-to-end.",
            },
        ]
    }


@app.post(
    "/pipelines/{pipeline_name}/run",
    response_model=PipelineRunResponse,
    tags=["Pipelines"],
)
def run_pipeline(pipeline_name: str, body: PipelineRunRequest = PipelineRunRequest()) -> PipelineRunResponse:
    """Run a Kedro pipeline by name.

    Pass a ``tuning`` object in the request body to override hyperparameter
    search settings for the ``training`` or ``__default__`` pipelines.
    The options map directly to ``params:training.tuning.*`` and take
    precedence over the values in ``parameters.yml``.
    """
    valid = {"data_engineering", "training", "inference", "refit", "__default__"}
    if pipeline_name not in valid:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline '{pipeline_name}' not found. Valid options: {sorted(valid)}",
        )

    extra_params: dict[str, Any] | None = None
    tuning_applied = False
    tuning_pipelines = {"training", "__default__"}

    if body.tuning is not None and pipeline_name in tuning_pipelines:
        extra_params = {
            "training.tuning.enabled": body.tuning.enabled,
            "training.tuning.method": body.tuning.method,
            "training.tuning.scoring": body.tuning.scoring,
            "training.tuning.n_iter": body.tuning.n_iter,
            "training.tuning.cv": body.tuning.cv,
        }
        tuning_applied = body.tuning.enabled
        logger.info(
            "api: tuning override applied  enabled=%s  method=%s  scoring=%s  n_iter=%d  cv=%d",
            body.tuning.enabled,
            body.tuning.method,
            body.tuning.scoring,
            body.tuning.n_iter,
            body.tuning.cv,
        )
    elif body.tuning is not None:
        logger.warning(
            "api: tuning options ignored — pipeline '%s' does not support tuning",
            pipeline_name,
        )

    try:
        _run_kedro_pipeline(pipeline_name, extra_params=extra_params)
    except Exception as exc:
        logger.exception("api: pipeline failed  name=%s", pipeline_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("api: reloading artifacts after pipeline run  name=%s", pipeline_name)
    _reload_pickle_artifacts()

    return PipelineRunResponse(
        pipeline_name=pipeline_name,
        status="success",
        message=f"Pipeline '{pipeline_name}' completed successfully.",
        tuning_applied=tuning_applied,
    )


@app.post("/predict", response_model=DiabetesPrediction, tags=["Inference"])
def predict_single(body: DiabetesInput) -> dict[str, Any]:
    """Predict usando o modelo **validado** (treinado só no split train).

    Use para avaliação e comparação com as métricas reportadas.
    """
    return _infer(body, model_key="model")


@app.post("/predict/production", response_model=DiabetesPrediction, tags=["Inference"])
def predict_production(body: DiabetesInput) -> dict[str, Any]:
    """Predict usando o modelo de **produção** (refitted em train + test).

    Requer que o pipeline ``refit`` tenha sido executado após ``data_engineering`` + ``training``.
    """
    return _infer(body, model_key="production_model")


@app.get("/metrics", tags=["Metrics"])
def get_metrics() -> JSONResponse:
    """Return the last training metrics for all models."""
    metrics_path = ARTIFACTS["metrics"]
    if not metrics_path.exists():
        logger.warning("api: metrics file not found  path=%s", metrics_path)
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Run the 'training' pipeline first.",
        )
    with open(metrics_path) as f:
        metrics = json.load(f)
    logger.debug("api: metrics served  models=%s", list(metrics))
    return JSONResponse(content=metrics)


# ---------------------------------------------------------------------------

def run() -> None:
    uvicorn.run(
        "diabetes_prediction.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
