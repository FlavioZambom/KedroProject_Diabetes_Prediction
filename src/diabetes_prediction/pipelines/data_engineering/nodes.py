"""Data engineering pipeline nodes."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. clean_data
# ---------------------------------------------------------------------------
def clean_data(raw_data: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Replace zeros with NaN in specific columns, impute with median,
    and clip outliers."""
    df = raw_data.copy()

    # --- Replace zeros with NaN (these cannot be zero physiologically) ---
    zero_cols = params["zero_replacement_columns"]
    for col in zero_cols:
        df[col] = np.where(df[col] == 0, np.nan, df[col])

    # --- Median imputation ---
    for col in zero_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # --- Outlier clipping at q1/q3 ---
    q1 = params.get("outlier_q1", 0.05)
    q3 = params.get("outlier_q3", 0.95)
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == params["target_column"]:
            continue
        quartile1 = df[col].quantile(q1)
        quartile3 = df[col].quantile(q3)
        iqr = quartile3 - quartile1
        low = quartile1 - 1.5 * iqr
        up = quartile3 + 1.5 * iqr
        df[col] = df[col].clip(lower=low, upper=up)

    logger.info("clean_data: shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 2. engineer_features
# ---------------------------------------------------------------------------
def engineer_features(df_in: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Create new features matching the original notebook."""
    df = df_in.copy()
    YOUNG = 21
    OLD = 50
    LOWINSULIN = 16
    HIGHINSULIN = 166

    # Age category
    df.loc[(df["Age"] >= YOUNG) & (df["Age"] < OLD), "NEW_AGE_CAT"] = "mature"
    df.loc[(df["Age"] >= OLD), "NEW_AGE_CAT"] = "senior"
    df["NEW_AGE_CAT"] = df["NEW_AGE_CAT"].fillna("mature")

    # BMI category
    df["NEW_BMI"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=["Underweight", "Healthy", "Overweight", "Obese"],
    )

    # Glucose category
    df["NEW_GLUCOSE"] = pd.cut(
        df["Glucose"],
        bins=[0, 140, 200, 300],
        labels=["Normal", "Prediabetes", "Diabetes"],
    )

    # Insulin score
    df["NEW_INSULIN_SCORE"] = df["Insulin"].apply(lambda x: "Normal" if LOWINSULIN <= x <= HIGHINSULIN else "Abnormal")

    # Interaction features (numeric)
    df["NEW_GLUCOSE_INSULIN"] = df["Glucose"] * df["Insulin"]
    df["NEW_GLUCOSE_PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

    # Uppercase column names (matching notebook)
    df.columns = [col.upper() for col in df.columns]

    logger.info("engineer_features: new shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 3. split_data
# ---------------------------------------------------------------------------
def split_data(df_in: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Add a 'split' column: train / test."""
    df = df_in.copy()
    np.random.seed(params.get("random_state", 17))
    test_size = params.get("test_size", 0.30)

    mask = np.random.rand(len(df)) < (1 - test_size)
    df["split"] = np.where(mask, "train", "test")

    logger.info(
        "split_data: train=%d  test=%d",
        (df["split"] == "train").sum(),
        (df["split"] == "test").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# 4. fit_encoders  (fit on train only)
# ---------------------------------------------------------------------------
def fit_encoders(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Fit LabelEncoders (binary cols) and OneHotEncoder (multi-class
    categorical cols) on the training split only."""
    target = params["target_column"]
    split_to_fit = params.get("split_to_fit", ["train"])
    df_fit = df.loc[df["split"].isin(split_to_fit)].copy()

    # Columns explicitly declared as multi-class in params take priority over
    # the nunique() heuristic.  This is necessary for columns like NEW_GLUCOSE
    # whose "Diabetes" bin (Glucose > 200) is never populated in this dataset
    # (max Glucose = 199), causing nunique() == 2 and a wrong LabelEncoder fit
    # that would raise ValueError if production data ever exceeds that limit.
    forced_multi: set[str] = set(params.get("multi_class_columns", []))

    # Identify categorical columns
    cat_cols = [c for c in df_fit.select_dtypes(include=["object", "category"]).columns if c not in [target, "split"]]
    multi_cols = [c for c in cat_cols if c in forced_multi or df_fit[c].nunique() > 2]  # noqa: PLR2004
    binary_cols = [c for c in cat_cols if c not in multi_cols]

    # Fit LabelEncoders for binary cols
    label_encoders = {}
    for col in binary_cols:
        le = LabelEncoder()
        le.fit(df_fit[col].astype(str))
        label_encoders[col] = le

    # Fit OneHotEncoder for multi-class categorical cols
    ohe = None
    if multi_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        # Convert categories to str to handle NaN
        ohe.fit(df_fit[multi_cols].astype(str))

    encoders = {
        "label_encoders": label_encoders,
        "binary_cols": binary_cols,
        "ohe": ohe,
        "multi_cols": multi_cols,
    }
    logger.info(
        "fit_encoders: binary=%s  multi=%s",
        binary_cols,
        multi_cols,
    )
    return encoders


# ---------------------------------------------------------------------------
# 5. transform_encoders
# ---------------------------------------------------------------------------
def transform_encoders(df_in: pd.DataFrame, encoders: dict[str, Any]) -> pd.DataFrame:
    """Apply fitted encoders to all splits."""
    df = df_in.copy()

    # Label encoding
    for col, le in encoders["label_encoders"].items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # One-hot encoding
    multi_cols = encoders["multi_cols"]
    ohe = encoders["ohe"]
    if ohe is not None and multi_cols:
        ohe_array = ohe.transform(df[multi_cols].astype(str))
        ohe_col_names = ohe.get_feature_names_out(multi_cols)
        ohe_df = pd.DataFrame(ohe_array, columns=ohe_col_names, index=df.index)
        df = df.drop(columns=multi_cols)
        df = pd.concat([df, ohe_df], axis=1)

    logger.info("transform_encoders: shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 6. fit_scaler  (fit on train only)
# ---------------------------------------------------------------------------
def fit_scaler(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Fit RobustScaler on numeric columns of the training split."""
    target = params["target_column"]
    split_to_fit = params.get("split_to_fit", ["train"])
    df_fit = df.loc[df["split"].isin(split_to_fit)].copy()

    num_cols = [c for c in df_fit.select_dtypes(include=[np.number]).columns if c not in [target]]

    scaler = RobustScaler()
    scaler.fit(df_fit[num_cols])

    scaler_artifact = {
        "scaler": scaler,
        "num_cols": num_cols,
    }
    logger.info("fit_scaler: columns=%s", num_cols)
    return scaler_artifact


# ---------------------------------------------------------------------------
# 7. transform_scaler  → master_table
# ---------------------------------------------------------------------------
def transform_scaler(df_in: pd.DataFrame, scaler_artifact: dict[str, Any]) -> pd.DataFrame:
    """Apply fitted scaler to all splits."""
    df = df_in.copy()
    scaler = scaler_artifact["scaler"]
    num_cols = scaler_artifact["num_cols"]

    # Only scale columns that exist in the dataframe
    cols_to_scale = [c for c in num_cols if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    logger.info("transform_scaler: shape=%s", df.shape)
    return df
