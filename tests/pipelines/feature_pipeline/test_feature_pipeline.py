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
