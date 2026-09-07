"""Preparación y carga de features históricos en Hopsworks.

El módulo fusiona filas casi duplicadas antes de deduplicar para evitar fuga de
información y descarta filas sin etiqueta o con bilirrubinas inconsistentes. Los
cocientes clínicos se generan con el transformador compartido del pipeline de
características.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipelines.feature_pipeline.transformers import CocientesClinicos

LOGGER = logging.getLogger(__name__)

ORIGINAL_FEATURES = [
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
]
DERIVED_FEATURES = ["Ratio_Bilirrubina_Directa", "Ratio_De_Ritis"]
TARGET = "Dataset"
PRIMARY_KEY = "record_id"
EXPECTED_COLUMNS = [PRIMARY_KEY, *ORIGINAL_FEATURES, *DERIVED_FEATURES, TARGET]
NUMERIC_FEATURES = [feature for feature in ORIGINAL_FEATURES if feature != "Gender"]
MIN_GROUP_SIZE = 2


def _almost_duplicate_pairs(data: pd.DataFrame) -> list[tuple[int, int]]:
    """Encuentra filas compatibles que solo difieren en valores faltantes."""
    rows = list(data[ORIGINAL_FEATURES].itertuples(index=False, name=None))
    pairs: list[tuple[int, int]] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            differs_in_null = False
            compatible = True
            for first, second in zip(rows[left], rows[right], strict=True):
                if pd.isna(first) and pd.isna(second):
                    continue
                if pd.isna(first) or pd.isna(second):
                    differs_in_null = True
                elif first != second:
                    compatible = False
                    break
            if compatible and differs_in_null:
                pairs.append((data.index[left], data.index[right]))
    return pairs


def _recover_duplicate_values(data: pd.DataFrame) -> pd.DataFrame:
    """Rellena nulos desde la fila hermana antes de eliminar duplicados."""
    parent = {index: index for index in data.index}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in _almost_duplicate_pairs(data):
        parent[find(right)] = find(left)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for index in data.index:
        groups[find(index)].append(index)

    result = data.copy()
    for members in groups.values():
        if len(members) < MIN_GROUP_SIZE:
            continue
        merged = result.loc[members[0]].copy()
        for member in members[1:]:
            # La etiqueta no se imputa desde otra fila: un conflicto debe
            # permanecer visible y provocar una colisión de record_id.
            for column in ORIGINAL_FEATURES:
                if pd.isna(merged[column]) and not pd.isna(result.loc[member, column]):
                    merged[column] = result.loc[member, column]
        result.loc[members, ORIGINAL_FEATURES] = merged[ORIGINAL_FEATURES].to_numpy()
    return result


def _read_historical_data(path: Path) -> pd.DataFrame:
    """Lee el CSV raw y normaliza sus tipos para la preparación histórica."""
    data = pd.read_csv(path, low_memory=False, na_values=["", "NA", "NaN"])
    missing = set([*ORIGINAL_FEATURES, TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en los datos raw: {sorted(missing)}")
    data = data[[*ORIGINAL_FEATURES, TARGET]].copy()
    data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    data["Gender"] = data["Gender"].astype("string").str.strip()
    target = pd.to_numeric(data[TARGET], errors="coerce")
    data[TARGET] = target.astype("Int64").astype("string")
    return data


def _canonical_value(value: object) -> str:
    """Serializa un valor de entrada sin depender del índice del DataFrame."""
    if pd.isna(value):
        return "null"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return str(value)


def _record_id(row: pd.Series) -> str:
    """Construye una clave estable para un perfil clínico deduplicado.

    La granularidad de ``record_id`` es una combinación de los valores de las
    diez variables originales de una observación histórica, no una posición de
    fila ni una predicción futura. La etiqueta queda fuera de la clave para que
    no entre información objetivo en la identidad del registro; si dos etiquetas
    distintas comparten el mismo perfil, la validación rechaza el conflicto.
    """
    payload = [(column, _canonical_value(row[column])) for column in ORIGINAL_FEATURES]
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def prepare_historical_features(path: str | Path) -> pd.DataFrame:
    """Prepara el conjunto histórico que se enviará al Feature Store."""
    raw = _read_historical_data(Path(path))
    recovered = _recover_duplicate_values(raw)
    prepared = recovered.drop_duplicates().copy()
    prepared = prepared[prepared[TARGET].notna()]
    inconsistent = (prepared["Direct_Bilirubin"] > prepared["Total_Bilirubin"]).fillna(False)
    prepared = prepared.loc[~inconsistent].copy()

    ratios = CocientesClinicos().fit_transform(prepared)
    prepared[DERIVED_FEATURES] = ratios
    prepared.insert(0, PRIMARY_KEY, prepared.apply(_record_id, axis=1))
    prepared = prepared[EXPECTED_COLUMNS].reset_index(drop=True)
    validate_feature_frame(prepared)
    LOGGER.info("Features históricos preparados: %d registros", len(prepared))
    return prepared


def validate_feature_frame(data: pd.DataFrame) -> None:  # noqa: C901, PLR0912
    """Valida esquema, tipos, nulos permitidos y unicidad de la clave."""
    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Esquema inesperado. Se esperaba {EXPECTED_COLUMNS}, recibido {list(data.columns)}"
        )
    if data.empty:
        raise ValueError("El conjunto de features está vacío")
    if data[TARGET].isna().any():
        raise ValueError("La variable objetivo no puede contener nulos")
    if not (pd.api.types.is_string_dtype(data[TARGET]) or data[TARGET].dtype == object):
        raise TypeError("La variable objetivo debe ser texto con etiquetas 1 o 2")
    if not data[TARGET].isin(["1", "2"]).all():
        raise ValueError("La variable objetivo solo puede contener las etiquetas 1 y 2")
    if not (pd.api.types.is_string_dtype(data[PRIMARY_KEY]) or data[PRIMARY_KEY].dtype == object):
        raise TypeError("record_id debe ser texto")
    if data[PRIMARY_KEY].isna().any() or data[PRIMARY_KEY].duplicated().any():
        raise ValueError("La clave primaria contiene nulos o duplicados")
    if data[PRIMARY_KEY].astype("string").str.strip().eq("").any():
        raise ValueError("La clave primaria no puede estar vacía")
    valid_gender = data["Gender"].dropna().isin(["Male", "Female"])
    if not valid_gender.all():
        raise ValueError("Gender solo puede contener Male o Female")
    if data["Gender"].isna().any():
        LOGGER.warning("Gender contiene valores nulos; se conserva según la preparación existente")
    clinical_columns = NUMERIC_FEATURES + DERIVED_FEATURES
    if data[clinical_columns].select_dtypes(exclude=np.number).shape[1]:
        raise TypeError("Las variables clínicas y los cocientes deben ser numéricos")
    numeric_values = data[clinical_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values[~np.isnan(numeric_values)]).all():
        raise ValueError("Las variables clínicas no pueden contener infinitos")
    if (data[NUMERIC_FEATURES].lt(0).any()).any():
        raise ValueError("Las variables clínicas no pueden contener valores negativos")
    if data["Age"].notna().any() and (data["Age"].dropna() <= 0).any():
        raise ValueError("Age debe ser mayor que cero")
    inconsistent = (data["Direct_Bilirubin"] > data["Total_Bilirubin"]).fillna(False)
    if inconsistent.any():
        raise ValueError("Direct_Bilirubin no puede superar Total_Bilirubin")
    if data[DERIVED_FEATURES].isna().any().any():
        LOGGER.warning(
            "Hay cocientes nulos por pruebas clínicas faltantes; se conservarán para imputación posterior"
        )


def _configuration() -> dict[str, Any]:
    """Lee configuración de Hopsworks sin exponer valores sensibles."""
    project = os.getenv("HOPSWORKS_PROJECT", "Pacientes_con_problemas")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not project or not api_key:
        raise RuntimeError("Configure HOPSWORKS_PROJECT y HOPSWORKS_API_KEY antes de cargar")
    return {
        "project": project,
        "api_key_value": api_key,
        "host": os.getenv("HOPSWORKS_HOST"),
        "feature_group": os.getenv("HOPSWORKS_FEATURE_GROUP", "pacientes_higado_features"),
        "version": int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1")),
    }


def upload_to_hopsworks(data: pd.DataFrame) -> int:
    """Inserta los features y valida la lectura posterior desde Hopsworks."""
    validate_feature_frame(data)
    try:
        import hopsworks  # noqa: PLC0415  # carga opcional solo al conectar
    except ImportError as error:
        raise RuntimeError(
            "Instale la dependencia opcional feature-store para usar Hopsworks"
        ) from error

    settings = _configuration()
    login_options = {
        key: settings[key] for key in ("project", "api_key_value", "host") if settings[key]
    }
    project = hopsworks.login(**login_options)
    feature_store = project.get_feature_store()
    feature_group = feature_store.get_or_create_feature_group(
        name=settings["feature_group"],
        version=settings["version"],
        primary_key=[PRIMARY_KEY],
        description="Features históricos de pacientes con problemas de hígado",
        online_enabled=False,
    )
    feature_group.insert(data, write_options={"wait_for_job": True})
    stored = feature_group.select_all().read()
    normalized_columns = {str(column).lower(): column for column in stored.columns}
    if len(normalized_columns) != len(stored.columns):
        raise RuntimeError("Hopsworks devolvió columnas duplicadas tras normalizar nombres")
    expected_columns = {column.lower() for column in EXPECTED_COLUMNS}
    missing_stored = expected_columns - set(normalized_columns)
    if missing_stored:
        raise RuntimeError(f"Hopsworks no devolvió columnas esperadas: {sorted(missing_stored)}")
    stored = stored.rename(
        columns={normalized_columns[column.lower()]: column for column in EXPECTED_COLUMNS}
    )
    stored = stored[EXPECTED_COLUMNS]
    validate_feature_frame(stored)
    if len(stored) < len(data):
        raise RuntimeError(
            f"Hopsworks devolvió {len(stored)} registros; se esperaban al menos {len(data)}"
        )
    if not set(data[PRIMARY_KEY]).issubset(set(stored[PRIMARY_KEY])):
        raise RuntimeError("La lectura desde Hopsworks no contiene todos los registros insertados")
    LOGGER.info("Carga validada en Hopsworks: %d registros disponibles", len(stored))
    return len(stored)


def main() -> None:
    """Ejecuta preparación local y carga remota, salvo con --validate-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/01_raw/Pacientes_porblemas_higado_india.csv"),
    )
    parser.add_argument("--validate-only", action="store_true", help="No conecta a Hopsworks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    features = prepare_historical_features(args.data_path)
    if args.validate_only:
        LOGGER.info("Validación local completada; no se simuló una carga remota")
        return
    upload_to_hopsworks(features)


if __name__ == "__main__":
    main()
