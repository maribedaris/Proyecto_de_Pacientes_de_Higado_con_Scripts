from pathlib import Path

import pandas as pd
import pytest

from pipelines.feature_pipeline import feature_pipeline

RAW_COLUMNS = [
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
]
EXPECTED_RECORDS = 2
EXPECTED_DIRECT_BILIRUBIN = 0.4


def _write_raw_csv(tmp_path: Path) -> Path:
    rows = [
        [40, " Male ", 1.2, 0.4, 100, 20, 30, 6.0, 3.0, 1.0, 1],
        [40, " Male ", 1.2, 0.4, 100, 20, 30, 6.0, 3.0, 1.0, 1],
        [40, " Male ", 1.2, None, 100, 20, 30, 6.0, 3.0, 1.0, 1],
        [45, "Female", 1.0, 0.3, 110, 22, 35, 6.2, 3.1, 1.0, None],
        [55, "Male", 1.0, 1.2, 120, 25, 40, 6.5, 3.2, 0.9, 1],
        [50, "Female", 2.0, 0.8, 120, 25, 40, 6.5, 3.2, 0.9, 2],
    ]
    data = pd.DataFrame(rows, columns=RAW_COLUMNS)
    data["Unnamed: 0"] = range(len(data))
    path = tmp_path / "raw.csv"
    data.to_csv(path, index=False)
    return path


def test_prepare_features_cleans_raw_data_and_creates_ratios(tmp_path: Path) -> None:
    features = feature_pipeline.prepare_features(_write_raw_csv(tmp_path))

    assert list(features.columns) == [*RAW_COLUMNS, *feature_pipeline.DERIVED_COLUMNS]
    assert len(features) == EXPECTED_RECORDS
    assert not any(column.startswith("Unnamed:") for column in features.columns)
    assert features["Gender"].tolist() == ["Male", "Female"]
    assert features["Dataset"].tolist() == ["1", "2"]
    assert features.loc[0, "Direct_Bilirubin"] == EXPECTED_DIRECT_BILIRUBIN
    assert features.loc[0, "Ratio_Bilirrubina_Directa"] == pytest.approx(1 / 3)
    assert features.loc[0, "Ratio_De_Ritis"] == pytest.approx(1.5)
    assert features.loc[1, "Ratio_Bilirrubina_Directa"] == pytest.approx(0.4)
    assert features.loc[1, "Ratio_De_Ritis"] == pytest.approx(1.6)


def test_run_feature_pipeline_writes_parquet_without_absolute_paths(tmp_path: Path) -> None:
    output_path = tmp_path / "features" / "features.parquet"

    result = feature_pipeline.run_feature_pipeline(_write_raw_csv(tmp_path), output_path)

    assert result == output_path
    assert output_path.exists()
    saved = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(
        saved, feature_pipeline.prepare_features(_write_raw_csv(tmp_path))
    )


def test_main_accepts_data_and_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _write_raw_csv(tmp_path)
    output_path = tmp_path / "main-features.parquet"
    monkeypatch.setattr(
        "sys.argv",
        [
            "feature_pipeline.py",
            "--data-path",
            str(input_path),
            "--output-path",
            str(output_path),
        ],
    )

    feature_pipeline.main()

    assert output_path.is_file()


def test_rejects_non_numeric_value(tmp_path: Path) -> None:
    input_path = _write_raw_csv(tmp_path)
    data = pd.read_csv(input_path)
    data["Albumin"] = data["Albumin"].astype(object)
    data.loc[5, "Albumin"] = "not-a-number"
    data.to_csv(input_path, index=False)

    with pytest.raises(feature_pipeline.DataValidationError, match="Albumin"):
        feature_pipeline.prepare_features(input_path)


def test_rejects_missing_required_column(tmp_path: Path) -> None:
    input_path = _write_raw_csv(tmp_path)
    data = pd.read_csv(input_path).drop(columns=["Age"])
    data.to_csv(input_path, index=False)

    with pytest.raises(feature_pipeline.DataValidationError, match="Age"):
        feature_pipeline.prepare_features(input_path)


def test_rejects_invalid_dataset_label(tmp_path: Path) -> None:
    input_path = _write_raw_csv(tmp_path)
    data = pd.read_csv(input_path)
    data.loc[5, "Dataset"] = 3
    data.to_csv(input_path, index=False)

    with pytest.raises(feature_pipeline.DataValidationError, match="Dataset"):
        feature_pipeline.prepare_features(input_path)


def test_rejects_excessive_nulls(tmp_path: Path) -> None:
    input_path = _write_raw_csv(tmp_path)
    data = pd.read_csv(input_path)
    data.loc[5, "Albumin"] = None
    data.to_csv(input_path, index=False)

    with pytest.raises(feature_pipeline.DataValidationError, match="nulos"):
        feature_pipeline.prepare_features(input_path)


def test_rejects_invalid_gender_category(tmp_path: Path) -> None:
    input_path = _write_raw_csv(tmp_path)
    data = pd.read_csv(input_path)
    data.loc[5, "Gender"] = "Unknown"
    data.to_csv(input_path, index=False)

    with pytest.raises(feature_pipeline.DataValidationError, match="Gender"):
        feature_pipeline.prepare_features(input_path)


def test_rejects_invalid_age_without_persisting_features(tmp_path: Path) -> None:
    input_path = _write_raw_csv(tmp_path)
    output_path = tmp_path / "invalid-features.parquet"
    data = pd.read_csv(input_path)
    data.loc[5, "Age"] = -1
    data.to_csv(input_path, index=False)

    with pytest.raises(feature_pipeline.DataValidationError, match="Age"):
        feature_pipeline.run_feature_pipeline(input_path, output_path)

    assert not output_path.exists()


def test_rejects_age_equal_to_zero(tmp_path: Path) -> None:
    input_path = _write_raw_csv(tmp_path)
    data = pd.read_csv(input_path)
    data.loc[5, "Age"] = 0
    data.to_csv(input_path, index=False)

    with pytest.raises(feature_pipeline.DataValidationError, match="Age"):
        feature_pipeline.prepare_features(input_path)


def test_rejects_infinite_value(tmp_path: Path) -> None:
    input_path = _write_raw_csv(tmp_path)
    data = pd.read_csv(input_path)
    data.loc[5, "Albumin"] = float("inf")
    data.to_csv(input_path, index=False)

    with pytest.raises(feature_pipeline.DataValidationError, match="infinitos"):
        feature_pipeline.prepare_features(input_path)


def test_rejects_inconsistent_bilirubin_relation(tmp_path: Path) -> None:
    features = feature_pipeline.prepare_features(_write_raw_csv(tmp_path))
    features.loc[0, "Direct_Bilirubin"] = features.loc[0, "Total_Bilirubin"] + 1

    with pytest.raises(feature_pipeline.DataValidationError, match="Direct_Bilirubin"):
        feature_pipeline.validate_features(features)


def test_validate_features_rejects_duplicate_records(tmp_path: Path) -> None:
    features = feature_pipeline.prepare_features(_write_raw_csv(tmp_path))
    duplicated = pd.concat([features, features.iloc[[0]]], ignore_index=True)

    with pytest.raises(feature_pipeline.DataValidationError, match="duplicados"):
        feature_pipeline.validate_features(duplicated)


def test_validate_features_rejects_empty_dataset() -> None:
    columns = [*feature_pipeline.ORIGINAL_COLUMNS, *feature_pipeline.DERIVED_COLUMNS]

    with pytest.raises(feature_pipeline.DataValidationError, match="no quedan"):
        feature_pipeline.validate_features(pd.DataFrame(columns=columns))
