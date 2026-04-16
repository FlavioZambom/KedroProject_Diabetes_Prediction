"""Training pipeline nodes."""

import logging
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

_MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "catboost": CatBoostClassifier,
}


def _build_model(model_type: str, model_kwargs: dict, random_state: int):
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(_MODEL_REGISTRY)}"
        )
    model_cls = _MODEL_REGISTRY[model_type]
    kwargs = dict(model_kwargs)
    if "random_state" in model_cls().get_params():
        kwargs.setdefault("random_state", random_state)
    return model_cls(**kwargs)


def train_all_models(
    master_table: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Train every model declared under params:training.models on the train split.

    Returns a dict keyed by model name, each value being
    {"model": <fitted estimator>, "feature_names": [...]}.
    """
    target = params["target_column"]
    random_state = params.get("random_state", 46)
    models_cfg: dict = params.get("models", {})

    df_train = master_table.loc[master_table["split"] == "train"].copy()
    df_train = df_train.drop(columns=["split"])
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    feature_names = list(X_train.columns)

    all_artifacts: dict[str, Any] = {}
    for name, model_kwargs in models_cfg.items():
        model = _build_model(name, model_kwargs, random_state)
        model.fit(X_train, y_train)
        all_artifacts[name] = {"model": model, "feature_names": feature_names}
        logger.info(
            "train_all_models: [%s] trained on %d samples, %d features",
            name,
            len(X_train),
            len(feature_names),
        )

    return all_artifacts


def evaluate_all_models(
    master_table: pd.DataFrame,
    all_model_artifacts: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every model on the test split.

    Returns a dict keyed by model name, each value being a metrics dict.
    """
    target = params["target_column"]

    df_test = master_table.loc[master_table["split"] == "test"].copy()
    df_test = df_test.drop(columns=["split"])
    y_test = df_test[target]

    all_metrics: dict[str, Any] = {}
    for name, artifact in all_model_artifacts.items():
        model = artifact["model"]
        feature_names = artifact["feature_names"]
        y_pred = model.predict(df_test[feature_names])
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_pred), 4),
        }
        all_metrics[name] = metrics
        logger.info("evaluate_all_models: [%s] %s", name, metrics)

    return all_metrics


def select_best_model(
    all_model_artifacts: dict[str, Any],
    all_metrics: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Pick the best model by params:training.selection_metric.

    Returns a single model_artifact dict (same format consumed by refit).
    """
    metric = params.get("selection_metric", "f1")
    best_name = max(all_metrics, key=lambda name: all_metrics[name][metric])
    best_score = all_metrics[best_name][metric]

    logger.info(
        "select_best_model: winner=[%s] %s=%.4f  (all scores: %s)",
        best_name,
        metric,
        best_score,
        {n: v[metric] for n, v in all_metrics.items()},
    )

    return all_model_artifacts[best_name]
