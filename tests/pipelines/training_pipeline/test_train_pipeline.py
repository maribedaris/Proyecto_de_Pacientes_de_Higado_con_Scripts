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
    CV_FOLDS,
    FEATURE_COLUMNS,
    VALIDATION_THRESHOLD,
    _build_model,
    cross_validate_model,
    diagnose_generalization,
    read_feature_data,
    run_training_pipeline,
    select_threshold,
    split_features,
    validate_train_test_split,
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
            "Age": np.where(positive, 55, 30) + np.arange(rows) * 0.01,
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


def test_cross_validation_uses_stratified_folds(synthetic_features: pd.DataFrame) -> None:
    x_train, _, y_train, _ = split_features(synthetic_features)

    validation = cross_validate_model(_build_model(), x_train, y_train, threshold=0.5)

    assert len(validation["folds"]) == CV_FOLDS
    assert validation["threshold"] == VALIDATION_THRESHOLD
    assert set(validation["metrics"]) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "roc_auc",
    }
    for fold in validation["folds"]:
        assert sorted(fold["validation_class_counts"].values()) == [4, 4]


def test_generalization_diagnostic_identifies_overfitting() -> None:
    train_metrics = {"roc_auc": 0.95}
    cross_validation = {"metrics": {"roc_auc": {"mean": 0.7, "std": 0.1}}}
    test_metrics = {"roc_auc": 0.68}

    diagnosis = diagnose_generalization(train_metrics, cross_validation, test_metrics)

    assert diagnosis["diagnosis"] == "possible_overfitting"
    assert diagnosis["recommendations"]


def test_generalization_diagnostic_identifies_underfitting() -> None:
    train_metrics = {"roc_auc": 0.55}
    cross_validation = {"metrics": {"roc_auc": {"mean": 0.5, "std": 0.1}}}
    test_metrics = {"roc_auc": 0.5}

    diagnosis = diagnose_generalization(train_metrics, cross_validation, test_metrics)

    assert diagnosis["diagnosis"] == "possible_underfitting"
    assert diagnosis["recommendations"]


def test_generalization_diagnostic_identifies_consistent_generalization() -> None:
    train_metrics = {"roc_auc": 0.75}
    cross_validation = {"metrics": {"roc_auc": {"mean": 0.74, "std": 0.05}}}
    test_metrics = {"roc_auc": 0.7}

    diagnosis = diagnose_generalization(train_metrics, cross_validation, test_metrics)

    assert diagnosis["diagnosis"] == "consistent_generalization"
    assert diagnosis["recommendations"] == []


def test_generalization_diagnostic_identifies_cv_test_gap() -> None:
    train_metrics = {"roc_auc": 0.8}
    cross_validation = {"metrics": {"roc_auc": {"mean": 0.8, "std": 0.05}}}
    test_metrics = {"roc_auc": 0.65}

    diagnosis = diagnose_generalization(train_metrics, cross_validation, test_metrics)

    assert diagnosis["diagnosis"] == "possible_cv_test_gap"
    assert diagnosis["recommendations"]


def test_validate_train_test_split_accepts_valid_data() -> None:
    x_train = pd.DataFrame({"feature": [1, 2]})
    x_test = pd.DataFrame({"feature": [3, 4]})
    y_train = pd.Series([0, 1])
    y_test = pd.Series([0, 1])

    result = validate_train_test_split(x_train, x_test, y_train, y_test)

    assert result["passed"] is True
    assert result["shared_rows"] == 0
    assert result["target_distribution_warning"] is False


def test_validate_train_test_split_rejects_shared_rows() -> None:
    x_train = pd.DataFrame({"feature": [1, 2]})
    x_test = pd.DataFrame({"feature": [2, 3]})
    labels = pd.Series([0, 1])

    with pytest.raises(ValueError, match="idénticos compartidos"):
        validate_train_test_split(x_train, x_test, labels, labels)


def test_validate_train_test_split_rejects_incompatible_lengths() -> None:
    x_train = pd.DataFrame({"feature": [1, 2]})
    x_test = pd.DataFrame({"feature": [3, 4]})
    y_train = pd.Series([0])
    y_test = pd.Series([0, 1])

    with pytest.raises(ValueError):
        validate_train_test_split(x_train, x_test, y_train, y_test)


def test_validate_train_test_split_rejects_incompatible_columns() -> None:
    x_train = pd.DataFrame({"feature_a": [1]})
    x_test = pd.DataFrame({"feature_b": [1]})
    labels = pd.Series([0])

    with pytest.raises(ValueError, match="mismas columnas"):
        validate_train_test_split(x_train, x_test, labels, labels)


def test_validate_train_test_split_rejects_different_column_order() -> None:
    x_train = pd.DataFrame({"Age": [1], "Gender": ["Male"], "Total_Bilirubin": [1.0]})
    x_test = pd.DataFrame({"Gender": ["Female"], "Age": [2], "Total_Bilirubin": [2.0]})
    labels = pd.Series([0])

    with pytest.raises(ValueError):
        validate_train_test_split(x_train, x_test, labels, labels)


def test_validate_train_test_split_rejects_empty_sets() -> None:
    x_train = pd.DataFrame(columns=["feature"])
    x_test = pd.DataFrame({"feature": [1]})
    y_train = pd.Series(dtype=int)
    y_test = pd.Series([0])

    with pytest.raises(ValueError, match="contener registros"):
        validate_train_test_split(x_train, x_test, y_train, y_test)


def test_validate_train_test_split_warns_on_target_distribution() -> None:
    x_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    x_test = pd.DataFrame({"feature": [5, 6]})
    y_train = pd.Series([0, 0, 0, 1])
    y_test = pd.Series([1, 1])

    with pytest.warns(UserWarning, match="distribución"):
        result = validate_train_test_split(x_train, x_test, y_train, y_test)

    assert result["target_distribution_warning"] is True


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
        "train",
        "cross_validation",
        "test",
        "comparison",
        "generalization",
    }
    assert set(result.metrics["train"]) == {
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
    assert set(result.metrics["cross_validation"]["metrics"]) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "roc_auc",
    }
    assert result.metrics["comparison"]["threshold"] == VALIDATION_THRESHOLD
    assert result.metadata["random_state"] == RANDOM_STATE
    assert result.metadata["split"]["train_size"] == TRAIN_ROWS
    assert result.metadata["split"]["test_size_rows"] == TEST_ROWS
    assert result.metadata["methodology"]["test_used_only_for_final_evaluation"] is True
