from pathlib import Path

import pandas as pd
import pytest

from data.feature_store import (
    DERIVED_FEATURES,
    EXPECTED_COLUMNS,
    PRIMARY_KEY,
    _record_id,
    prepare_historical_features,
    validate_feature_frame,
)
from pipelines.feature_pipeline.transformers import CocientesClinicos

EXPECTED_PREPARED_ROWS = 3
EXPECTED_RECOVERED_BILIRUBIN = 0.4


def _write_csv(tmp_path: Path) -> Path:
    data = pd.DataFrame(
        [
            [40, "Male", 1.2, 0.4, 100, 20, 30, 6.0, 3.0, 1.0, "1"],
            [40, "Male", 1.2, None, 100, 20, 30, 6.0, 3.0, 1.0, "1"],
            [50, "Female", 2.0, 0.8, 120, 25, 40, 6.5, 3.2, 0.9, "2"],
            [60, "Male", 0.8, 0.5, 110, 30, 35, 6.1, 3.1, 1.0, "1"],
        ],
        columns=[
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
            "Dataset",
        ],
    )
    path = tmp_path / "historical.csv"
    data.to_csv(path, index=False)
    return path


def test_prepare_features_has_expected_schema_and_deduplicates(tmp_path: Path) -> None:
    features = prepare_historical_features(_write_csv(tmp_path))

    assert list(features.columns) == EXPECTED_COLUMNS
    assert len(features) == EXPECTED_PREPARED_ROWS
    assert features[PRIMARY_KEY].is_unique
    assert features["Direct_Bilirubin"].iloc[0] == EXPECTED_RECOVERED_BILIRUBIN
    assert features[DERIVED_FEATURES].notna().all().all()
    assert features["Ratio_Bilirrubina_Directa"].iloc[0] == pytest.approx(1 / 3)
    assert features["Ratio_De_Ritis"].iloc[0] == pytest.approx(1.5)
    assert features["Dataset"].dtype.name == "string"


def test_validate_feature_frame_rejects_duplicate_keys(tmp_path: Path) -> None:
    features = prepare_historical_features(_write_csv(tmp_path))
    features.loc[1, PRIMARY_KEY] = features.loc[0, PRIMARY_KEY]

    with pytest.raises(ValueError, match="duplicados"):
        validate_feature_frame(features)


@pytest.mark.parametrize("gender", ["Male", "Female"])
def test_validate_feature_frame_accepts_valid_gender(tmp_path: Path, gender: str) -> None:
    features = prepare_historical_features(_write_csv(tmp_path))
    features.loc[0, "Gender"] = gender

    validate_feature_frame(features)


def test_validate_feature_frame_allows_null_gender(tmp_path: Path) -> None:
    features = prepare_historical_features(_write_csv(tmp_path))
    features.loc[0, "Gender"] = None

    validate_feature_frame(features)


def test_validate_feature_frame_rejects_invalid_gender_with_nulls(tmp_path: Path) -> None:
    features = prepare_historical_features(_write_csv(tmp_path))
    features.loc[0, "Gender"] = "Male"
    features.loc[1, "Gender"] = None
    features.loc[2, "Gender"] = "Invalid"

    with pytest.raises(ValueError, match="Gender solo puede contener Male o Female"):
        validate_feature_frame(features)


def test_validate_feature_frame_rejects_direct_bilirubin_above_total(tmp_path: Path) -> None:
    features = prepare_historical_features(_write_csv(tmp_path))
    features.loc[0, "Direct_Bilirubin"] = features.loc[0, "Total_Bilirubin"] + 0.1

    with pytest.raises(ValueError, match="Direct_Bilirubin"):
        validate_feature_frame(features)


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_validate_feature_frame_rejects_infinite_values(tmp_path: Path, value: float) -> None:
    features = prepare_historical_features(_write_csv(tmp_path))
    features.loc[0, "Albumin"] = value

    with pytest.raises(ValueError, match="infinitos"):
        validate_feature_frame(features)


def test_clinical_ratios_turn_zero_denominators_into_nan() -> None:
    data = pd.DataFrame(
        {
            "Direct_Bilirubin": [1.0, 0.0],
            "Total_Bilirubin": [0.0, 1.0],
            "Aspartate_Aminotransferase": [10.0, 0.0],
            "Alamine_Aminotransferase": [0.0, 10.0],
        }
    )

    ratios = CocientesClinicos().fit_transform(data)

    assert ratios.loc[0, "Ratio_Bilirrubina_Directa"] != ratios.loc[0, "Ratio_Bilirrubina_Directa"]
    assert ratios.loc[0, "Ratio_De_Ritis"] != ratios.loc[0, "Ratio_De_Ritis"]
    assert ratios.loc[1, "Ratio_Bilirrubina_Directa"] == 0
    assert ratios.loc[1, "Ratio_De_Ritis"] == 0


def test_validate_feature_frame_rejects_missing_columns(tmp_path: Path) -> None:
    features = prepare_historical_features(_write_csv(tmp_path)).drop(columns=[DERIVED_FEATURES[0]])

    with pytest.raises(ValueError, match="Esquema inesperado"):
        validate_feature_frame(features)


def test_record_id_is_deterministic_and_does_not_use_index(tmp_path: Path) -> None:
    features = prepare_historical_features(_write_csv(tmp_path))
    row = features.iloc[0].copy()
    row.name = 9999

    assert _record_id(features.iloc[0]) == _record_id(row)


def test_prepare_rejects_invalid_target(tmp_path: Path) -> None:
    path = _write_csv(tmp_path)
    data = pd.read_csv(path)
    data.loc[0, "Dataset"] = 3
    data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="etiquetas 1 y 2"):
        prepare_historical_features(path)


def test_prepare_rejects_conflicting_labels_for_same_profile(tmp_path: Path) -> None:
    path = _write_csv(tmp_path)
    data = pd.read_csv(path)
    data.loc[1, "Dataset"] = 2
    data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicados"):
        prepare_historical_features(path)


def test_validate_rejects_negative_clinical_value(tmp_path: Path) -> None:
    features = prepare_historical_features(_write_csv(tmp_path))
    features.loc[0, "Albumin"] = -1

    with pytest.raises(ValueError, match="negativos"):
        validate_feature_frame(features)


def test_prepare_rejects_missing_required_column(tmp_path: Path) -> None:
    path = _write_csv(tmp_path)
    data = pd.read_csv(path).drop(columns=["Age"])
    data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        prepare_historical_features(path)
