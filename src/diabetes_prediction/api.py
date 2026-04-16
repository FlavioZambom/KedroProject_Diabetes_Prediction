"""FastAPI application exposing the Diabetes Prediction Kedro pipelines."""

import json
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
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

ARTIFACTS = {
    "model": DATA_DIR / "06_models" / "model_artifact.pkl",
    "production_model": DATA_DIR / "06_models" / "production_model.pkl",
    "encoders": DATA_DIR / "03_primary" / "encoders.pkl",
    "scaler": DATA_DIR / "03_primary" / "scaler.pkl",
    "metrics": DATA_DIR / "08_reporting" / "metrics.json",
}

DATA_ENGINEERING_PARAMS = {
    "target_column": "OUTCOME",
    "zero_replacement_columns": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    "outlier_q1": 0.05,
    "outlier_q3": 0.95,
    "test_size": 0.30,
    "random_state": 17,
    "split_to_fit": ["train"],
}

state: dict[str, Any] = {}


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for key, path in ARTIFACTS.items():
        if key == "metrics":
            continue
        if path.exists():
            state[key] = _load_pickle(path)
            logger.info("Loaded artifact: %s", path.name)
        else:
            logger.warning("Artifact not found (run pipelines first): %s", path)
    yield
    state.clear()


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


class PipelineRunResponse(BaseModel):
    pipeline_name: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Helper: run a Kedro pipeline via KedroSession
# ---------------------------------------------------------------------------
def _run_kedro_pipeline(pipeline_name: str) -> None:
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(project_path=PROJECT_ROOT) as session:
        session.run(pipeline_name=pipeline_name)


# ---------------------------------------------------------------------------
# Helper: inference without Kedro session (uses preloaded artifacts)
# ---------------------------------------------------------------------------
def _infer(raw_input: DiabetesInput, model_key: str = "model") -> dict[str, Any]:
    if model_key not in state:
        hint = (
            "Run the 'data_engineering' + 'training' pipelines first."
            if model_key == "model"
            else "Run the 'refit' pipeline first."
        )
        raise HTTPException(status_code=503, detail=f"Artifact '{model_key}' not loaded. {hint}")
    if "encoders" not in state or "scaler" not in state:
        raise HTTPException(
            status_code=503,
            detail="Encoder/scaler artifacts not loaded. Run the 'data_engineering' pipeline first.",
        )

    df = pd.DataFrame([raw_input.model_dump()])
    df_clean = clean_data(df, DATA_ENGINEERING_PARAMS)
    df_feat = engineer_features(df_clean, DATA_ENGINEERING_PARAMS)
    df_enc = transform_encoders(df_feat, state["encoders"])
    df_scaled = transform_scaler(df_enc, state["scaler"])
    results = predict(df_scaled, state[model_key])

    prediction = int(results["prediction"].iloc[0])
    probability = float(results["probability"].iloc[0])
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "label": "Diabetes" if prediction == 1 else "No Diabetes",
    }


# ---------------------------------------------------------------------------


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "artifacts_loaded": {k: True for k in state if k != "metrics"},
        "artifacts_missing": [k for k, p in ARTIFACTS.items() if k != "metrics" and k not in state],
    }


@app.get("/pipelines", tags=["Pipelines"])
def list_pipelines():
    return {
        "pipelines": [
            {
                "name": "data_engineering",
                "description": "Clean raw data, engineer features, fit encoders & scaler, produce master_table.",
            },
            {
                "name": "training",
                "description": "Train all configured models (RandomForest, LogisticRegression, CatBoost), evaluate each on the test split, select the best by selection_metric; saves model_artifact and metrics.",
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


@app.post("/pipelines/{pipeline_name}/run", response_model=PipelineRunResponse, tags=["Pipelines"])
def run_pipeline(pipeline_name: str):
    valid = {"data_engineering", "training", "inference", "refit", "__default__"}
    if pipeline_name not in valid:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline '{pipeline_name}' not found. Valid options: {sorted(valid)}",
        )

    try:
        _run_kedro_pipeline(pipeline_name)
    except Exception as exc:
        logger.exception("Pipeline '%s' failed", pipeline_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    for key, path in ARTIFACTS.items():
        if key == "metrics":
            continue
        if path.exists():
            state[key] = _load_pickle(path)

    return PipelineRunResponse(
        pipeline_name=pipeline_name,
        status="success",
        message=f"Pipeline '{pipeline_name}' completed successfully.",
    )


@app.post("/predict", response_model=DiabetesPrediction, tags=["Inference"])
def predict_single(body: DiabetesInput):
    """
    Predict usando o modelo **validado** (treinado só no split train).

    Use para avaliação e comparação com as métricas reportadas.
    """
    return _infer(body, model_key="model")


@app.post("/predict/production", response_model=DiabetesPrediction, tags=["Inference"])
def predict_production(body: DiabetesInput):
    """
    Predict usando o modelo de **produção** (refitted em train + test).

    Requer que o pipeline `refit` tenha sido executado após `data_engineering` + `training`.
    """
    return _infer(body, model_key="production_model")


@app.get("/metrics", tags=["Metrics"])
def get_metrics():
    metrics_path = ARTIFACTS["metrics"]
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Run the 'training' pipeline first.",
        )
    with open(metrics_path) as f:
        metrics = json.load(f)
    return JSONResponse(content=metrics)


# ---------------------------------------------------------------------------
def run():
    uvicorn.run(
        "diabetes_prediction.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
