"""Training pipeline nodes."""

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def train_model(
    master_table: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Train a RandomForestClassifier on the training split."""
    target = params["target_column"]

    df_train = master_table.loc[master_table["split"] == "train"].copy()
    df_train = df_train.drop(columns=["split"])

    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]

    model = RandomForestClassifier(
        random_state=params.get("random_state", 46),
        **params.get("model_params", {}),
    )
    model.fit(X_train, y_train)

    feature_names = list(X_train.columns)

    artifact = {
        "model": model,
        "feature_names": feature_names,
    }
    logger.info("train_model: trained on %d samples, %d features", len(X_train), len(feature_names))
    return artifact


def evaluate_model(
    master_table: pd.DataFrame,
    model_artifact: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, float]:
    """Evaluate the model on the test split and return metrics."""
    target = params["target_column"]
    model = model_artifact["model"]
    feature_names = model_artifact["feature_names"]

    df_test = master_table.loc[master_table["split"] == "test"].copy()
    df_test = df_test.drop(columns=["split"])

    X_test = df_test[feature_names]
    y_test = df_test[target]

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_pred), 4),
    }

    logger.info("evaluate_model: %s", metrics)
    return metrics
