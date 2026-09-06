"""Entrenamiento reproducible de GaussianNB sobre las features persistidas."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("data/04_feature/Pacientes_porblemas_higado_features.parquet")
DEFAULT_MODEL_PATH = Path("data/06_models/pacientes_higado_gaussiannb.joblib")
DEFAULT_METRICS_PATH = Path("data/07_model_output/pacientes_higado_metrics.json")
DEFAULT_METADATA_PATH = Path("data/07_model_output/pacientes_higado_metadata.json")
TARGET_COLUMN = "Dataset"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10
EXPECTED_CLASS_COUNT = 2
SMOOTHING_VALUES = np.logspace(-10, -2, 9)
FEATURE_COLUMNS = [
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
    "Ratio_Bilirrubina_Directa",
    "Ratio_De_Ritis",
]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column != "Gender"]
CATEGORICAL_COLUMNS = ["Gender"]


@dataclass
class TrainingResult:
    """Resultado completo del entrenamiento y de la evaluación final."""

    model: Pipeline
    metrics: dict[str, float | list[list[int]]]
    metadata: dict[str, Any]
    threshold: float


def read_feature_data(input_path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Lee y valida las features producidas por el Feature Pipeline."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el Parquet de features: {path}")
    data = pd.read_parquet(path, engine="pyarrow")
    expected = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [column for column in expected if column not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en las features: {missing}")
    data = data[expected].copy()
    if data.empty:
        raise ValueError("El Parquet de features está vacío")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Dataset no puede contener etiquetas nulas")
    labels = data[TARGET_COLUMN].astype("string").str.strip()
    if not labels.isin({"1", "2"}).all():
        raise ValueError("Dataset solo puede contener las etiquetas 1 y 2")
    if labels.nunique() != EXPECTED_CLASS_COUNT:
        raise ValueError("Dataset debe contener las dos clases para un entrenamiento binario")
    data[TARGET_COLUMN] = labels
    return data


def split_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Separa variables y target, recodifica la clase positiva y aplica el split."""
    features = data[FEATURE_COLUMNS]
    target = (data[TARGET_COLUMN] == "1").astype(int)
    return cast(
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
        train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            stratify=target,
            random_state=RANDOM_STATE,
        ),
    )


def _build_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
    )
    return Pipeline([("preprocessor", preprocessor), ("model", GaussianNB())])


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Selecciona el threshold que maximiza f1_macro sobre probabilidades OOF."""
    candidates = np.quantile(probabilities, np.linspace(0.01, 0.99, 197))
    scores = [
        f1_score(y_true, (probabilities >= threshold).astype(int), average="macro")
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def _calculate_metrics(
    y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | list[list[int]]]:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro")),
        "specificity": float(recall_score(y_true, predictions, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
    }


def _json_ready(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def run_training_pipeline(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> TrainingResult:
    """Ajusta GaussianNB con train, evalúa test una única vez y persiste resultados."""
    data = read_feature_data(input_path)
    x_train, x_test, y_train, y_test = split_features(data)

    grid = GridSearchCV(
        _build_model(),
        {"model__var_smoothing": SMOOTHING_VALUES},
        cv=_cv(),
        scoring="f1_macro",
        n_jobs=1,
        refit=True,
    )
    grid.fit(x_train, y_train)

    oof_probabilities = cross_val_predict(
        grid.best_estimator_,
        x_train,
        y_train,
        cv=_cv(),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold = select_threshold(y_train, oof_probabilities)

    model = grid.best_estimator_.fit(x_train, y_train)
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    metrics = _calculate_metrics(y_test, test_predictions, test_probabilities)
    metadata: dict[str, Any] = {
        "model": "GaussianNB",
        "best_params": grid.best_params_,
        "cv_best_score_f1_macro": float(grid.best_score_),
        "threshold": threshold,
        "random_state": RANDOM_STATE,
        "split": {
            "test_size": TEST_SIZE,
            "train_size": len(x_train),
            "test_size_rows": len(x_test),
            "stratified": True,
        },
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "positive_label": "1",
        "input_path": str(input_path),
        "methodology": {
            "grid_search_cv_folds": CV_FOLDS,
            "grid_search_scoring": "f1_macro",
            "oof_threshold": True,
            "test_used_only_for_final_evaluation": True,
        },
        "metrics": metrics,
    }

    model_output = Path(model_path)
    metrics_output = Path(metrics_path)
    metadata_output = Path(metadata_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_output, protocol=5)
    metrics_output.write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")
    metadata_output.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")
    LOGGER.info("Modelo guardado en %s; threshold OOF: %.16g", model_output, threshold)
    return TrainingResult(model=model, metrics=metrics, metadata=metadata, threshold=threshold)


def main() -> None:
    """Ejecuta el Training Pipeline desde la raíz del proyecto."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_training_pipeline(args.input_path, args.model_path, args.metrics_path, args.metadata_path)


if __name__ == "__main__":
    main()
