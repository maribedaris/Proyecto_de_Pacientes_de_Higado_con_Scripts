"""Pipeline local de características: CSV RAW a features históricas en Parquet."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.feature_pipeline.transformers import CocientesClinicos

LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data/01_raw/Pacientes_porblemas_higado_india.csv")
DEFAULT_OUTPUT_PATH = Path("data/04_feature/Pacientes_porblemas_higado_features.parquet")
ORIGINAL_COLUMNS = [
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
NUMERIC_COLUMNS = [column for column in ORIGINAL_COLUMNS if column not in {"Gender", "Dataset"}]
DERIVED_COLUMNS = ["Ratio_Bilirrubina_Directa", "Ratio_De_Ritis"]
MIN_COMPATIBLE_GROUP_SIZE = 2
MAX_NULL_RATIO = 0.05
# El dataset real llega como máximo a 2.87%; 5% permite esa variación sin
# aceptar una degradación importante de la calidad de entrada.
VALID_GENDERS = frozenset({"Male", "Female"})
VALID_TARGETS = frozenset({1, 2})


class DataValidationError(ValueError):
    """Error de calidad o integridad que impide persistir las features."""


def _read_raw_data(data_path: str | Path) -> pd.DataFrame:
    """Lee las columnas relevantes y valida las condiciones básicas de entrada.

    Los tipos se comprueban antes de normalizarlos para no convertir un valor
    inválido en un nulo silenciosamente.
    """
    data = pd.read_csv(data_path, low_memory=False, na_values=["", "NA", "NaN"])
    missing = set(ORIGINAL_COLUMNS) - set(data.columns)
    if missing:
        raise DataValidationError(
            f"Validación de datos fallida: faltan columnas obligatorias {sorted(missing)}."
        )
    data = data[ORIGINAL_COLUMNS].copy()

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].notna() & converted.isna()
        if invalid.any():
            raise DataValidationError(
                f"Validación de datos fallida: {column} contiene valores no numéricos."
            )

    genders = data["Gender"].astype("string").str.strip()
    invalid_genders = genders.notna() & ~genders.isin(VALID_GENDERS)
    if invalid_genders.any():
        raise DataValidationError(
            "Validación de datos fallida: Gender solo puede contener Male o Female."
        )

    target = pd.to_numeric(data["Dataset"], errors="coerce")
    invalid_target = data["Dataset"].notna() & target.isna()
    if invalid_target.any() or not target.dropna().isin(VALID_TARGETS).all():
        raise DataValidationError(
            "Validación de datos fallida: Dataset solo puede contener las etiquetas 1 y 2."
        )

    data[NUMERIC_COLUMNS] = data[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data["Gender"] = genders
    data["Dataset"] = target.astype("Int64").astype("string")
    return data


def _compatible_pairs(data: pd.DataFrame) -> list[tuple[int, int]]:
    """Encuentra filas que solo difieren en valores faltantes."""
    rows = list(data[ORIGINAL_COLUMNS[:-1]].itertuples(index=False, name=None))
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


def _recover_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Recupera valores de filas compatibles antes de deduplicar."""
    parent = {index: index for index in data.index}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in _compatible_pairs(data):
        parent[find(right)] = find(left)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for index in data.index:
        groups[find(index)].append(index)

    result = data.copy()
    for members in groups.values():
        if len(members) < MIN_COMPATIBLE_GROUP_SIZE:
            continue
        merged = result.loc[members[0]].copy()
        for member in members[1:]:
            for column in ORIGINAL_COLUMNS[:-1]:
                if pd.isna(merged[column]) and not pd.isna(result.loc[member, column]):
                    merged[column] = result.loc[member, column]
        result.loc[members, ORIGINAL_COLUMNS[:-1]] = merged[ORIGINAL_COLUMNS[:-1]].to_numpy()
    return result


def prepare_features(data_path: str | Path) -> pd.DataFrame:
    """Prepara, transforma y valida el histórico antes de devolver sus features.

    El flujo recupera faltantes compatibles, elimina duplicados y descarta
    etiquetas o bilirrubinas inconsistentes. Después crea los ratios y ejecuta
    la validación final.
    """
    data = _recover_missing_values(_read_raw_data(data_path))
    data = data.drop_duplicates()
    data = data[data["Dataset"].notna()]
    # Las inconsistencias conocidas se eliminan antes de generar features; la
    # validación posterior garantiza que ninguna llegue al Parquet.
    inconsistent = (data["Direct_Bilirubin"] > data["Total_Bilirubin"]).fillna(False)
    data = data.loc[~inconsistent].copy()

    data[DERIVED_COLUMNS] = CocientesClinicos().fit_transform(data)
    features = data[[*ORIGINAL_COLUMNS, *DERIVED_COLUMNS]].reset_index(drop=True)
    validate_features(features)
    return features


def _validate_schema_and_nulls(features: pd.DataFrame) -> None:
    expected_columns = [*ORIGINAL_COLUMNS, *DERIVED_COLUMNS]
    if list(features.columns) != expected_columns:
        raise DataValidationError(
            "Validación de datos fallida: el esquema de features no coincide con el esperado."
        )
    null_ratios = features[ORIGINAL_COLUMNS + DERIVED_COLUMNS].isna().mean()
    exceeded = null_ratios[null_ratios > MAX_NULL_RATIO]
    if not exceeded.empty:
        columns = ", ".join(f"{column} ({ratio:.2%})" for column, ratio in exceeded.items())
        raise DataValidationError(
            f"Validación de datos fallida: porcentaje de nulos superior al 5% en {columns}."
        )


def _validate_categories(features: pd.DataFrame) -> None:
    invalid_genders = features["Gender"].dropna().isin(VALID_GENDERS).eq(False)
    if invalid_genders.any():
        raise DataValidationError(
            "Validación de datos fallida: Gender solo puede contener Male o Female."
        )
    if not features["Dataset"].isin({"1", "2"}).all():
        raise DataValidationError(
            "Validación de datos fallida: Dataset solo puede contener las etiquetas 1 y 2."
        )


def _validate_numeric_ranges(features: pd.DataFrame) -> None:
    numeric_columns = NUMERIC_COLUMNS + DERIVED_COLUMNS
    if features[numeric_columns].select_dtypes(exclude=np.number).shape[1]:
        raise DataValidationError(
            "Validación de datos fallida: las variables clínicas deben ser numéricas."
        )
    numeric_values = features[numeric_columns].to_numpy(dtype=float)
    finite_values = numeric_values[~np.isnan(numeric_values)]
    if not np.isfinite(finite_values).all():
        raise DataValidationError(
            "Validación de datos fallida: las variables clínicas contienen infinitos."
        )
    negative_columns = features[NUMERIC_COLUMNS].lt(0).any()
    if negative_columns.any():
        columns = ", ".join(negative_columns[negative_columns].index)
        raise DataValidationError(
            f"Validación de datos fallida: las variables clínicas no pueden ser negativas ({columns})."
        )
    if (features["Age"].dropna() <= 0).any():
        raise DataValidationError("Validación de datos fallida: Age debe ser mayor que cero.")


def _validate_integrity(features: pd.DataFrame) -> None:
    inconsistent = (features["Direct_Bilirubin"] > features["Total_Bilirubin"]).fillna(False)
    if inconsistent.any():
        raise DataValidationError(
            "Validación de datos fallida: Direct_Bilirubin no puede superar Total_Bilirubin."
        )
    if features.duplicated().any():
        raise DataValidationError(
            "Validación de datos fallida: persisten registros duplicados en las features."
        )


def validate_features(features: pd.DataFrame) -> None:
    """Actúa como última barrera antes de permitir la persistencia.

    Si una regla de esquema, nulos, categorías, rangos o integridad falla,
    lanza ``DataValidationError`` y el pipeline no llega a escribir el Parquet.
    """
    if features.empty:
        raise DataValidationError("Validación de datos fallida: no quedan registros válidos.")
    _validate_schema_and_nulls(features)
    _validate_categories(features)
    _validate_numeric_ranges(features)
    _validate_integrity(features)


def run_feature_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Genera el Parquet solo después de validar correctamente las features."""
    features = prepare_features(data_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False, engine="pyarrow")
    LOGGER.info("Features generadas: %d registros en %s", len(features), output)
    return output


def main() -> None:
    """Ejecuta el pipeline desde la raíz del proyecto."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_feature_pipeline(args.data_path, args.output_path)


if __name__ == "__main__":
    main()
