"""Adaptación de datos clínicos RAW al contrato de inferencia."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.feature_pipeline.transformers import CocientesClinicos
from pipelines.training_pipeline.train_pipeline import FEATURE_COLUMNS

RAW_INFERENCE_COLUMNS = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]
DERIVED_COLUMNS = ["Ratio_Bilirrubina_Directa", "Ratio_De_Ritis"]
NUMERIC_COLUMNS = [column for column in RAW_INFERENCE_COLUMNS if column != "Gender"]
VALID_GENDERS = frozenset({"Male", "Female"})


def _validate_raw_predictors(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(RAW_INFERENCE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas clínicas obligatorias: {missing}")
    raw = data[RAW_INFERENCE_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(raw[column], errors="coerce")
        invalid = raw[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"{column} contiene valores no numéricos")
        raw[column] = converted
    raw["Gender"] = raw["Gender"].astype("string").str.strip()
    invalid_genders = raw["Gender"].notna() & ~raw["Gender"].isin(VALID_GENDERS)
    if invalid_genders.any():
        raise ValueError("Gender solo puede contener Male o Female")
    if raw[NUMERIC_COLUMNS].lt(0).any().any():
        raise ValueError("Las variables clínicas no pueden ser negativas")
    if (raw["Age"].dropna() <= 0).any():
        raise ValueError("Age debe ser mayor que cero")
    inconsistent = (raw["Direct_Bilirubin"] > raw["Total_Bilirubin"]).fillna(False)
    if inconsistent.any():
        raise ValueError("Direct_Bilirubin no puede superar Total_Bilirubin")
    return raw


def prepare_inference_features(data: pd.DataFrame) -> pd.DataFrame:
    """Acepta features completas o predictores RAW y devuelve las 12 columnas del modelo."""
    if data.empty:
        raise ValueError("El archivo de entrada no contiene registros")
    if set(FEATURE_COLUMNS).issubset(data.columns):
        return data[FEATURE_COLUMNS].copy()
    raw = _validate_raw_predictors(data)
    ratios = CocientesClinicos().fit_transform(raw)
    features = pd.concat([raw, ratios], axis=1)
    numeric_values = features[NUMERIC_COLUMNS + DERIVED_COLUMNS].to_numpy(dtype=float)
    finite_values = numeric_values[~np.isnan(numeric_values)]
    if not np.isfinite(finite_values).all():
        raise ValueError("Las variables clínicas contienen valores infinitos")
    return features[FEATURE_COLUMNS].copy()
