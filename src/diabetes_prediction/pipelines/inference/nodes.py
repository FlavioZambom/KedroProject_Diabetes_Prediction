"""Inference pipeline nodes."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def predict(
    df_in: pd.DataFrame,
    model_artifact: dict[str, Any],
) -> pd.DataFrame:
    """Generate predictions using the trained model."""
    df = df_in.copy()
    model = model_artifact["model"]
    feature_names = model_artifact["feature_names"]

    # Drop non-feature columns
    cols_to_drop = [c for c in ["split", "OUTCOME"] if c in df.columns]
    X = df.drop(columns=cols_to_drop, errors="ignore")

    # Align columns to match training features
    missing = [c for c in feature_names if c not in X.columns]
    for c in missing:
        X[c] = 0
    X = X[feature_names]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "prediction": predictions,
        "probability": probabilities,
    })

    logger.info("predict: generated %d predictions", len(result))
    return result
