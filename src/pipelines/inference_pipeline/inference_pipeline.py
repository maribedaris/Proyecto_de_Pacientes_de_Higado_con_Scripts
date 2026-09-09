"""Genera predicciones con el modelo entrenado y sus transformaciones persistidas."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from pipelines.training_pipeline.train_pipeline import (
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("data/04_feature/Pacientes_porblemas_higado_features.parquet")
DEFAULT_METADATA_PATH = Path("data/07_model_output/pacientes_higado_metadata.json")
DEFAULT_OUTPUT_PATH = Path("data/07_model_output/pacientes_higado_predictions.parquet")


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Carga el Pipeline de scikit-learn guardado por el Training Pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado: {path}")

    model = load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"El artifact no contiene un sklearn Pipeline: {path}")
    LOGGER.info("Modelo cargado desde %s", path)
    return model


def read_inference_data(input_path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Lee un archivo de features y valida las columnas que necesita el modelo."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")

    if path.suffix.lower() == ".parquet":
        data = pd.read_parquet(path, engine="pyarrow")
    elif path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    else:
        raise ValueError("El archivo de entrada debe tener extensión .parquet o .csv")

    missing = sorted(set(FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas predictoras obligatorias: {missing}")
    if data.empty:
        raise ValueError("El archivo de entrada no contiene registros")
    return data


def load_threshold(metadata_path: str | Path = DEFAULT_METADATA_PATH) -> float:
    """Lee el threshold OOF guardado junto con el modelo."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe la metadata del modelo: {path}")

    metadata: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    threshold = metadata.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError(f"La metadata no contiene un threshold válido: {path}")
    return float(threshold)


def predict(model: Pipeline, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Transforma y predice sin ajustar el Pipeline con los datos nuevos."""
    features = data[FEATURE_COLUMNS]
    LOGGER.info("Aplicando transformaciones persistidas a %d registros", len(features))
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    LOGGER.info("Predicciones generadas: %d registros", len(predictions))

    result = pd.DataFrame(
        {
            "record_id": data.index.to_numpy(),
            "prediction": predictions,
            "positive_probability": probabilities,
        },
        index=data.index,
    )
    return result.reset_index(drop=True)


def save_predictions(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    """Persiste las predicciones en la capa de salida de modelos."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        predictions.to_parquet(path, index=False, engine="pyarrow")
    elif path.suffix.lower() == ".csv":
        predictions.to_csv(path, index=False)
    else:
        raise ValueError("La salida debe tener extensión .parquet o .csv")
    LOGGER.info("Predicciones guardadas en %s", path)
    return path


def run_inference(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Ejecuta lectura, carga, transformación, predicción y persistencia."""
    data = read_inference_data(input_path)
    model = load_model(model_path)
    threshold = load_threshold(metadata_path)
    predictions = predict(model, data, threshold)
    save_predictions(predictions, output_path)
    return predictions


def main() -> None:
    """Ejecuta el pipeline desde la raíz del proyecto."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_inference(args.input_path, args.model_path, args.metadata_path, args.output_path)


if __name__ == "__main__":
    main()
