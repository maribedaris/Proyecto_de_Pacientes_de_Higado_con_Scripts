from pathlib import Path
from typing import TypedDict

import pandas as pd
import pytest

from pipelines.feature_pipeline import feature_pipeline

EXPECTED_RECORDS = 2


class PipelineCalls(TypedDict, total=False):
    path: str | Path
    features: pd.DataFrame


def _synthetic_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["record-1", "record-2"],
            "Age": [40, 50],
            "Gender": ["Male", "Female"],
            "Total_Bilirubin": [1.2, 2.0],
            "Direct_Bilirubin": [0.4, 0.8],
            "Alkaline_Phosphotase": [100, 120],
            "Alamine_Aminotransferase": [20, 25],
            "Aspartate_Aminotransferase": [30, 40],
            "Total_Protiens": [6.0, 6.5],
            "Albumin": [3.0, 3.2],
            "Albumin_and_Globulin_Ratio": [1.0, 0.9],
            "Ratio_Bilirrubina_Directa": [1 / 3, 0.4],
            "Ratio_De_Ritis": [1.5, 1.6],
            "Dataset": ["1", "2"],
        }
    )


def test_run_feature_pipeline_reuses_preparation_and_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _synthetic_features()
    calls: PipelineCalls = {}

    def fake_prepare(path: str | Path) -> pd.DataFrame:
        calls["path"] = path
        return features

    def fake_store(data: pd.DataFrame) -> int:
        calls["features"] = data
        return len(data)

    monkeypatch.setattr(feature_pipeline, "prepare_features", fake_prepare)
    monkeypatch.setattr(feature_pipeline, "store_features", fake_store)

    result = feature_pipeline.run_feature_pipeline("synthetic.csv")

    assert result == EXPECTED_RECORDS
    assert calls["path"] == "synthetic.csv"
    assert calls["features"] is features
    assert features["record_id"].is_unique
    assert {"Ratio_Bilirrubina_Directa", "Ratio_De_Ritis"}.issubset(features.columns)


def test_run_feature_pipeline_uses_default_data_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: PipelineCalls = {}

    def fake_prepare(path: str | Path) -> pd.DataFrame:
        captured["path"] = path
        return _synthetic_features()

    def fake_store(data: pd.DataFrame) -> int:
        return len(data)

    monkeypatch.setattr(feature_pipeline, "prepare_features", fake_prepare)
    monkeypatch.setattr(feature_pipeline, "store_features", fake_store)

    assert feature_pipeline.run_feature_pipeline() == EXPECTED_RECORDS
    assert captured["path"] == feature_pipeline.DEFAULT_DATA_PATH
