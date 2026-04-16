"""Training pipeline nodes."""

import logging
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logger = logging.getLogger(__name__)

# Model name -> class mapping; add new models here without touching pipeline logic
_MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "catboost": CatBoostClassifier,
}

# Type aliases
ModelArtifact = dict[str, Any]  # {"model": estimator, "feature_names": list[str]}
AllArtifacts = dict[str, ModelArtifact]  # model_name -> ModelArtifact
Metrics = dict[str, float]  # {"accuracy": ..., "f1": ..., ...}
AllMetrics = dict[str, Metrics]  # model_name -> Metrics


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_model(model_type: str, model_kwargs: dict[str, Any], random_state: int) -> BaseEstimator:
    """Instantiate a model from the registry with the given kwargs.

    Injects ``random_state`` automatically for models that support it.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {sorted(_MODEL_REGISTRY)}")
    model_cls = _MODEL_REGISTRY[model_type]
    kwargs: dict[str, Any] = dict(model_kwargs)
    if "random_state" in model_cls().get_params():
        kwargs.setdefault("random_state", random_state)
    return model_cls(**kwargs)


def _tune_model(  # noqa: PLR0913
    model: BaseEstimator,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuning_cfg: dict[str, Any],
    random_state: int,
) -> tuple[BaseEstimator, dict[str, Any], float]:
    """Run CV hyperparameter search and return *(best_estimator, best_params, best_cv_score)*.

    Args:
        model: Unfitted base estimator to search over.
        param_grid: Search space — keys are param names, values are candidate lists.
        X_train: Training features.
        y_train: Training target.
        tuning_cfg: Tuning section from ``params:training``.
        random_state: Seed forwarded to :class:`~sklearn.model_selection.RandomizedSearchCV`.

    Returns:
        Tuple of (fitted best estimator, best hyperparameters dict, best CV score).
    """
    method: str = tuning_cfg.get("method", "random")
    cv: int = tuning_cfg.get("cv", 5)
    scoring: str = tuning_cfg.get("scoring", "f1")

    if method == "grid":
        search: GridSearchCV | RandomizedSearchCV = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True
        )
        logger.debug("_tune_model: using GridSearchCV  cv=%d  scoring=%s", cv, scoring)
    else:
        n_iter: int = tuning_cfg.get("n_iter", 20)
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            random_state=random_state,
        )
        logger.debug(
            "_tune_model: using RandomizedSearchCV  cv=%d  scoring=%s  n_iter=%d",
            cv,
            scoring,
            n_iter,
        )

    search.fit(X_train, y_train)
    best_score = round(float(search.best_score_), 4)
    logger.debug("_tune_model: search complete  best_score=%.4f", best_score)
    return search.best_estimator_, dict(search.best_params_), best_score


def _compute_metrics(y_true: pd.Series, y_pred: Any) -> Metrics:
    """Compute classification metrics for a single model."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_pred)), 4),
    }


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------


def train_all_models(
    master_table: pd.DataFrame,
    params: dict[str, Any],
) -> AllArtifacts:
    """Train every model declared under ``params:training.models`` on the train split.

    When ``params:training.tuning.enabled`` is ``true``, runs CV hyperparameter
    search for each model that has a ``param_grid`` defined under
    ``params:training.tuning.param_grids``. Models without a param_grid are
    trained with the fixed hyperparameters from ``models``.

    Args:
        master_table: Full dataset with a ``split`` column (``"train"`` / ``"test"``).
        params: Contents of ``params:training``.

    Returns:
        Dict mapping model name to ``{"model": estimator, "feature_names": [...]}``.
    """
    target: str = params["target_column"]
    random_state: int = params.get("random_state", 46)
    models_cfg: dict[str, dict[str, Any]] = params.get("models", {})
    tuning_cfg: dict[str, Any] = params.get("tuning", {})
    tuning_enabled: bool = tuning_cfg.get("enabled", False)
    param_grids: dict[str, dict[str, list[Any]]] = tuning_cfg.get("param_grids", {})

    df_train = master_table.loc[master_table["split"] == "train"].copy()
    df_train = df_train.drop(columns=["split"])
    X_train: pd.DataFrame = df_train.drop(columns=[target])
    y_train: pd.Series = df_train[target]
    feature_names: list[str] = list(X_train.columns)

    logger.info(
        "train_all_models: starting  models=%s  tuning=%s  samples=%d  features=%d",
        list(models_cfg),
        tuning_enabled,
        len(X_train),
        len(feature_names),
    )

    all_artifacts: AllArtifacts = {}

    for name, model_kwargs in models_cfg.items():
        base_model = _build_model(name, model_kwargs, random_state)

        if tuning_enabled and name in param_grids:
            logger.info(
                "train_all_models: [%s] hyperparameter search started  method=%s  cv=%d  n_iter=%s  scoring=%s",
                name,
                tuning_cfg.get("method", "random"),
                tuning_cfg.get("cv", 5),
                tuning_cfg.get("n_iter", 20) if tuning_cfg.get("method", "random") != "grid" else "N/A (grid)",
                tuning_cfg.get("scoring", "f1"),
            )
            model, best_params, best_cv_score = _tune_model(
                base_model, param_grids[name], X_train, y_train, tuning_cfg, random_state
            )
            logger.info(
                "train_all_models: [%s] search complete  best_cv_%s=%.4f  best_params=%s",
                name,
                tuning_cfg.get("scoring", "f1"),
                best_cv_score,
                best_params,
            )
        else:
            model = base_model
            model.fit(X_train, y_train)
            if tuning_enabled and name not in param_grids:
                logger.warning(
                    "train_all_models: [%s] tuning is enabled but no param_grid found"
                    " — trained with fixed hyperparameters",
                    name,
                )
            else:
                logger.info(
                    "train_all_models: [%s] trained with fixed hyperparameters  samples=%d  features=%d",
                    name,
                    len(X_train),
                    len(feature_names),
                )

        all_artifacts[name] = {"model": model, "feature_names": feature_names}
        logger.debug("train_all_models: [%s] artifact saved", name)

    logger.info("train_all_models: done  trained=%d models", len(all_artifacts))
    return all_artifacts


def evaluate_all_models(
    master_table: pd.DataFrame,
    all_model_artifacts: AllArtifacts,
    params: dict[str, Any],
) -> AllMetrics:
    """Evaluate every trained model on the test split.

    Args:
        master_table: Full dataset with a ``split`` column.
        all_model_artifacts: Output of :func:`train_all_models`.
        params: Contents of ``params:training``.

    Returns:
        Dict mapping model name to a metrics dict
        ``{accuracy, recall, precision, f1, roc_auc}``.
    """
    target: str = params["target_column"]

    df_test = master_table.loc[master_table["split"] == "test"].copy()
    df_test = df_test.drop(columns=["split"])
    y_test: pd.Series = df_test[target]

    logger.info(
        "evaluate_all_models: evaluating %d models on %d test samples",
        len(all_model_artifacts),
        len(df_test),
    )

    all_metrics: AllMetrics = {}

    for name, artifact in all_model_artifacts.items():
        model: BaseEstimator = artifact["model"]
        feature_names: list[str] = artifact["feature_names"]

        logger.debug("evaluate_all_models: [%s] running predictions", name)
        y_pred = model.predict(df_test[feature_names])
        metrics = _compute_metrics(y_test, y_pred)
        all_metrics[name] = metrics

        logger.info("evaluate_all_models: [%s] %s", name, metrics)

    return all_metrics


def select_best_model(
    all_model_artifacts: AllArtifacts,
    all_metrics: AllMetrics,
    params: dict[str, Any],
) -> ModelArtifact:
    """Pick the best model by ``params:training.selection_metric``.

    Args:
        all_model_artifacts: Output of :func:`train_all_models`.
        all_metrics: Output of :func:`evaluate_all_models`.
        params: Contents of ``params:training``.

    Returns:
        Single ``{"model": estimator, "feature_names": [...]}`` artifact
        — the same format expected by the refit pipeline.
    """
    metric: str = params.get("selection_metric", "f1")

    scores: dict[str, float] = {name: m[metric] for name, m in all_metrics.items()}
    best_name: str = max(scores, key=lambda n: scores[n])
    best_score: float = scores[best_name]

    logger.info(
        "select_best_model: metric=%s  ranking=%s",
        metric,
        dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)),
    )
    logger.info(
        "select_best_model: winner=[%s]  %s=%.4f",
        best_name,
        metric,
        best_score,
    )

    return all_model_artifacts[best_name]
