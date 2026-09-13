"""Pruebas del adaptador de entradas RAW para inferencia."""

import pandas as pd
import pytest

from pipelines.inference_pipeline.input_adapter import prepare_inference_features
from pipelines.training_pipeline.train_pipeline import FEATURE_COLUMNS


def _raw_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [45],
            "Gender": [" Male "],
            "Total_Bilirubin": [1.0],
            "Direct_Bilirubin": [0.3],
            "Alkaline_Phosphotase": [200],
            "Alamine_Aminotransferase": [30],
            "Aspartate_Aminotransferase": [45],
            "Total_Protiens": [6.5],
            "Albumin": [3.2],
            "Albumin_and_Globulin_Ratio": [1.0],
        }
    )


def test_prepare_inference_features_calculates_ratios() -> None:
    features = prepare_inference_features(_raw_row())

    assert list(features.columns) == FEATURE_COLUMNS
    assert features.loc[0, "Gender"] == "Male"
    assert features.loc[0, "Ratio_Bilirrubina_Directa"] == pytest.approx(0.3)
    assert features.loc[0, "Ratio_De_Ritis"] == pytest.approx(1.5)


def test_prepare_inference_features_accepts_already_prepared_features() -> None:
    prepared = _raw_row()
    prepared["Ratio_Bilirrubina_Directa"] = 0.3
    prepared["Ratio_De_Ritis"] = 1.5

    features = prepare_inference_features(prepared)

    assert list(features.columns) == FEATURE_COLUMNS


def test_prepare_inference_features_rejects_missing_raw_column() -> None:
    with pytest.raises(ValueError, match="Faltan columnas clínicas"):
        prepare_inference_features(_raw_row().drop(columns=["Age"]))


def test_prepare_inference_features_rejects_invalid_raw_values() -> None:
    invalid = _raw_row()
    invalid.loc[0, "Gender"] = "Unknown"

    with pytest.raises(ValueError, match="Gender"):
        prepare_inference_features(invalid)


def test_prepare_inference_features_rejects_inconsistent_bilirubin() -> None:
    invalid = _raw_row()
    invalid.loc[0, "Direct_Bilirubin"] = 2.0

    with pytest.raises(ValueError, match="Direct_Bilirubin"):
        prepare_inference_features(invalid)
