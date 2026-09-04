"""Orquestación autónoma de la preparación y carga de features históricas."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from data.feature_store import prepare_historical_features, upload_to_hopsworks

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_PATH = Path("data/01_raw/Pacientes_porblemas_higado_india.csv")


def prepare_features(data_path: str | Path) -> pd.DataFrame:
    """Lee y transforma el histórico usando la preparación del Feature Store."""
    return prepare_historical_features(data_path)


def store_features(features: pd.DataFrame) -> int:
    """Almacena las features preparadas y valida su lectura posterior."""
    return int(upload_to_hopsworks(features))


def run_feature_pipeline(data_path: str | Path = DEFAULT_DATA_PATH) -> int:
    """Prepara el histórico y lo almacena en Hopsworks."""
    features = prepare_features(data_path)
    LOGGER.info("Iniciando carga de %d registros históricos", len(features))
    stored_records = store_features(features)
    LOGGER.info("Pipeline completado: %d registros disponibles", stored_records)
    return stored_records


def main() -> None:
    """Ejecuta el pipeline desde la raíz del proyecto."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Ruta al CSV histórico",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_feature_pipeline(args.data_path)


if __name__ == "__main__":
    main()
