"""Pruebas unitarias del Inference Pipeline con datos y artifacts temporales."""

import json
from pathlib import Path

import pandas as pd
import pytest
from joblib import dump
from sklearn.pipeline import Pipeline

from pipelines.inference_pipeline.inference_pipeline import (
    FEATURE_COLUMNS,
    load_model,
    predict,
    read_inference_data,
    run_inference,
)
from pipelines.training_pipeline.train_pipeline import _build_model

PREDICTION_ROWS = 3


def _synthetic_features(rows: int = 6) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "Age": [20, 30, 40, 50, 60, 70][:rows],
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"][:rows],
            "Total_Bilirubin": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0][:rows],
            "Direct_Bilirubin": [0.3, 0.5, 0.8, 1.0, 1.2, 1.5][:rows],
            "Alkaline_Phosphotase": [100, 110, 120, 130, 140, 150][:rows],
            "Alamine_Aminotransferase": [20, 25, 30, 35, 40, 45][:rows],
            "Aspartate_Aminotransferase": [30, 35, 40, 45, 50, 55][:rows],
            "Total_Protiens": [6.0, 6.2, 6.4, 6.6, 6.8, 7.0][:rows],
            "Albumin": [3.0, 3.2, 3.4, 3.6, 3.8, 4.0][:rows],
            "Albumin_and_Globulin_Ratio": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3][:rows],
            "Ratio_Bilirrubina_Directa": [0.3, 0.25, 0.27, 0.25, 0.24, 0.25][:rows],
            "Ratio_De_Ritis": [1.5, 1.4, 1.33, 1.29, 1.25, 1.22][:rows],
        }
    )
    return data[FEATURE_COLUMNS]


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    model = _build_model().fit(_synthetic_features(), pd.Series([0, 1, 0, 1, 0, 1]))
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    dump(model, model_path)
    metadata_path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
    return model_path, metadata_path


def test_load_model_reads_saved_pipeline(tmp_path: Path) -> None:
    model_path, _ = _write_artifacts(tmp_path)

    loaded = load_model(model_path)

    assert isinstance(loaded, Pipeline)
    assert "preprocessor" in loaded.named_steps


def test_read_inference_data_reads_parquet_and_rejects_missing_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _synthetic_features(2).to_parquet(input_path, index=False)

    data = read_inference_data(input_path)

    assert list(data.columns) == FEATURE_COLUMNS
    missing_path = tmp_path / "missing.csv"
    pd.DataFrame({"Age": [20]}).to_csv(missing_path, index=False)
    with pytest.raises(ValueError, match="Faltan columnas"):
        read_inference_data(missing_path)


def test_predict_reuses_fitted_transformations_and_preserves_row_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, _ = _write_artifacts(tmp_path)
    model = load_model(model_path)
    before = (
        model.named_steps["preprocessor"].transformers_[0][1].named_steps["scaler"].mean_.copy()
    )
    monkeypatch.setattr(model, "fit", lambda *_args, **_kwargs: pytest.fail("No debe ajustar"))

    predictions = predict(model, _synthetic_features(PREDICTION_ROWS), threshold=0.5)

    after = model.named_steps["preprocessor"].transformers_[0][1].named_steps["scaler"].mean_
    assert len(predictions) == PREDICTION_ROWS
    assert list(predictions.columns) == ["record_id", "prediction", "positive_probability"]
    assert (before == after).all()


def test_run_inference_persists_predictions_and_ignores_target(tmp_path: Path) -> None:
    model_path, metadata_path = _write_artifacts(tmp_path)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.parquet"
    data = _synthetic_features(4)
    data["Dataset"] = [1, 2, 1, 2]
    data.to_csv(input_path, index=False)

    predictions = run_inference(input_path, model_path, metadata_path, output_path)

    saved = pd.read_parquet(output_path)
    assert len(predictions) == len(data) == len(saved)
    assert saved["prediction"].isin([0, 1]).all()
    assert saved["positive_probability"].between(0, 1).all()


def test_missing_model_and_input_raise_clear_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="modelo entrenado"):
        load_model(tmp_path / "missing.joblib")
    with pytest.raises(FileNotFoundError, match="archivo de entrada"):
        read_inference_data(tmp_path / "missing.parquet")
