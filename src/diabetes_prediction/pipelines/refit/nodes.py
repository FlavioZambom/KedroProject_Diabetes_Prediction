"""Refit pipeline nodes."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def refit_model(
    master_table: pd.DataFrame,
    model_artifact: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Retrain the validated model on all data splits for production use.

    Extracts the estimator class and hyperparameters from ``model_artifact``,
    then fits a fresh instance on the splits defined in ``params["train_splits"]``
    (typically train + test).

    Args:
        master_table: Fully-processed DataFrame with a ``split`` column.
        model_artifact: Artifact produced by ``train_model``, containing
            ``model`` (fitted estimator) and ``feature_names``.
        params: Refit configuration with keys:
            - ``target_column`` (str): Name of the target column.
            - ``train_splits`` (list[str]): Split labels to train on.

    Returns:
        Production model artifact with the same structure as ``model_artifact``.
    """
    source_model = model_artifact["model"]
    feature_names = model_artifact["feature_names"]
    target = params["target_column"]
    train_splits = params["train_splits"]

    df_train = master_table[master_table["split"].isin(train_splits)].copy()
    X_train = df_train[feature_names]
    y_train = df_train[target]

    production_model = type(source_model)(**source_model.get_params())
    production_model.fit(X_train, y_train)

    logger.info(
        "refit_model: %s refitted on %d samples (%d features) — splits=%s",
        type(production_model).__name__,
        len(X_train),
        len(feature_names),
        train_splits,
    )

    return {
        "model": production_model,
        "feature_names": feature_names,
    }
