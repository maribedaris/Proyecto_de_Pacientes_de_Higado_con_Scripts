"""Feature pipeline local: CSV raw a features históricas en Parquet."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

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


def _read_raw_data(data_path: str | Path) -> pd.DataFrame:
    """Lee únicamente las columnas relevantes y normaliza sus tipos."""
    data = pd.read_csv(data_path, low_memory=False, na_values=["", "NA", "NaN"])
    data = data[ORIGINAL_COLUMNS].copy()
    data[NUMERIC_COLUMNS] = data[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data["Gender"] = data["Gender"].astype("string").str.strip()
    data["Dataset"] = (
        pd.to_numeric(data["Dataset"], errors="coerce").astype("Int64").astype("string")
    )
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
    """Limpia el histórico y crea las variables clínicas derivadas."""
    data = _recover_missing_values(_read_raw_data(data_path))
    data = data.drop_duplicates()
    data = data[data["Dataset"].notna()]
    inconsistent = (data["Direct_Bilirubin"] > data["Total_Bilirubin"]).fillna(False)
    data = data.loc[~inconsistent].copy()

    data[DERIVED_COLUMNS] = CocientesClinicos().fit_transform(data)
    return data[[*ORIGINAL_COLUMNS, *DERIVED_COLUMNS]].reset_index(drop=True)


def run_feature_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Ejecuta el pipeline y devuelve la ruta del Parquet generado."""
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
