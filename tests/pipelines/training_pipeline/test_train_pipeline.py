"""Pruebas unitarias del Training Pipeline con datos sintéticos."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import load
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from pipelines.training_pipeline.train_pipeline import (
    FEATURE_COLUMNS,
    read_feature_data,
    run_training_pipeline,
    select_threshold,
    split_features,
)

SYNTHETIC_ROWS = 100
TRAIN_ROWS = 80
TEST_ROWS = 20
RANDOM_STATE = 42


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    rows = SYNTHETIC_ROWS
    positive = np.arange(rows) % 2 == 0
    data = pd.DataFrame(
        {
            "Age": np.where(positive, 55, 30),
            "Gender": np.where(positive, "Male", "Female"),
            "Total_Bilirubin": np.where(positive, 4.0, 0.8),
            "Direct_Bilirubin": np.where(positive, 1.5, 0.2),
            "Alkaline_Phosphotase": np.where(positive, 250.0, 100.0),
            "Alamine_Aminotransferase": np.where(positive, 80.0, 25.0),
            "Aspartate_Aminotransferase": np.where(positive, 90.0, 30.0),
            "Total_Protiens": np.where(positive, 5.5, 7.0),
            "Albumin": np.where(positive, 2.5, 4.0),
            "Albumin_and_Globulin_Ratio": np.where(positive, 0.7, 1.3),
            "Ratio_Bilirrubina_Directa": np.where(positive, 0.375, 0.25),
            "Ratio_De_Ritis": np.where(positive, 1.125, 1.2),
            "Dataset": np.where(positive, "1", "2"),
        }
    )
    data.loc[0, "Albumin"] = np.nan
    data.loc[1, "Gender"] = None
    return data


def test_read_feature_data_validates_schema(
    tmp_path: Path, synthetic_features: pd.DataFrame
) -> None:
    path = tmp_path / "features.parquet"
    synthetic_features.to_parquet(path)
    loaded = read_feature_data(path)
    assert list(loaded.columns) == [*FEATURE_COLUMNS, "Dataset"]
    assert len(loaded) == SYNTHETIC_ROWS

    invalid = synthetic_features.drop(columns="Ratio_De_Ritis")
    invalid.to_parquet(path)
    with pytest.raises(ValueError, match="Faltan columnas"):
        read_feature_data(path)


def test_split_is_stratified_and_reproducible(synthetic_features: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = split_features(synthetic_features)
    other_split = split_features(synthetic_features)
    assert len(x_train) == TRAIN_ROWS
    assert len(x_test) == TEST_ROWS
    assert y_train.mean() == pytest.approx(y_test.mean())
    pd.testing.assert_frame_equal(x_train, other_split[0])
    pd.testing.assert_series_equal(y_test, other_split[3])


def test_threshold_uses_probabilities_and_returns_candidate() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.01, 0.02, 0.8, 0.9])

    candidates = np.quantile(probabilities, np.linspace(0.01, 0.99, 197))
    candidate_scores = [
        f1_score(y_true, (probabilities >= candidate).astype(int), average="macro")
        for candidate in candidates
    ]
    expected_threshold = float(candidates[int(np.argmax(candidate_scores))])

    threshold = select_threshold(y_true, probabilities)
    assert threshold == pytest.approx(expected_threshold)
    assert f1_score(y_true, (probabilities >= threshold).astype(int), average="macro") == max(
        candidate_scores
    )


def test_training_generates_metrics_model_and_metadata(
    tmp_path: Path, synthetic_features: pd.DataFrame
) -> None:
    input_path = tmp_path / "features.parquet"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    metadata_path = tmp_path / "metadata.json"
    synthetic_features.to_parquet(input_path)

    result = run_training_pipeline(input_path, model_path, metrics_path, metadata_path)

    assert model_path.exists()
    loaded_model = load(model_path)
    assert isinstance(loaded_model, Pipeline)
    assert isinstance(loaded_model.named_steps["model"], GaussianNB)
    assert metrics_path.exists()
    assert metadata_path.exists()
    assert set(result.metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "specificity",
        "roc_auc",
        "average_precision",
        "confusion_matrix",
    }
    assert result.metadata["random_state"] == RANDOM_STATE
    assert result.metadata["split"]["train_size"] == TRAIN_ROWS
    assert result.metadata["split"]["test_size_rows"] == TEST_ROWS
    assert result.metadata["methodology"]["test_used_only_for_final_evaluation"] is True
