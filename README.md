# Pacientes con problemas de hígado - India

Proyecto de ciencia de datos y machine learning para analizar pacientes con posibles
problemas hepáticos a partir del dataset ILPD. El proyecto utiliza Python 3.12, `uv`,
`pytest`, Ruff, Mypy y `pre-commit`.

## Arquitectura FTI

El proyecto mantiene una separación modular entre tres pipelines:

```text
Datos RAW
    ↓
Feature Pipeline: validación, limpieza y generación de features
    ↓
Features
    ↓
Training Pipeline: entrenamiento, evaluación y selección del modelo
    ↓
Modelo entrenado
    ↓
Inference Pipeline: datos nuevos y predicciones
```

El estado actual es:

- **Feature Pipeline:** implementado en `src/pipelines/feature_pipeline/`.
- **Training Pipeline:** implementado en `src/pipelines/training_pipeline/`, incluyendo la validación de la separación Train/Test.
- **Inference Pipeline:** pendiente de implementación productiva; los análisis actuales están en `notebooks/6-interpretation/`.
- **Orquestación FTI completa:** pendiente.

Los notebooks se utilizan para exploración, análisis y experimentación. La lógica
productiva de cada etapa debe implementarse en su pipeline correspondiente dentro de
`src/pipelines/`.

## Feature Pipeline

El Feature Pipeline transforma el archivo RAW
`data/01_raw/Pacientes_porblemas_higado_india.csv` en features y las guarda en
`data/04_feature/Pacientes_porblemas_higado_features.parquet`.

Actualmente realiza:

- validación de columnas, tipos, categorías, nulos y rangos;
- recuperación de valores faltantes entre filas compatibles;
- eliminación de duplicados y registros sin etiqueta;
- eliminación de inconsistencias de bilirrubina identificadas en los datos;
- creación de `Ratio_Bilirrubina_Directa` y `Ratio_De_Ritis`;
- validación final antes de persistir el Parquet.

Para ejecutarlo desde la raíz del proyecto:

```bash
PYTHONPATH=src uv run --no-sync python -m pipelines.feature_pipeline.feature_pipeline
```

Las rutas pueden personalizarse con `--data-path` y `--output-path`.

## Configuración del entorno

Crear o actualizar el entorno con:

```bash
uv sync
```

Activarlo manualmente cuando sea necesario:

```bash
source .venv/bin/activate
```

Las dependencias se declaran en `pyproject.toml` y las versiones resueltas se conservan
en `uv.lock`.

## Pruebas y calidad

Ejecutar la suite completa:

```bash
uv run --no-sync pytest
```

Ejecutar las validaciones de calidad:

```bash
uv run --no-sync pre-commit run --all-files
```

Las pruebas están organizadas por responsabilidad dentro de `tests/`. Las pruebas del
Feature Pipeline se encuentran en
`tests/pipelines/feature_pipeline/test_feature_pipeline.py`.

## Estructura principal

```text
.
├── data/
│   ├── 01_raw/          # Datos fuente e inmutables
│   ├── 02_intermediate/ # Datos tipados o intermedios
│   ├── 03_primary/      # Datos limpios del dominio
│   ├── 04_feature/      # Features generadas
│   └── 05_model_input/  # Datos preparados para entrenamiento
├── models/              # Modelos y artefactos serializados
├── notebooks/           # Exploración, análisis y experimentación
├── src/
│   ├── data/             # Lectura, validación y persistencia de datos
│   ├── inference/        # Componentes de inferencia
│   ├── model/            # Componentes de modelado
│   └── pipelines/        # Feature, Training e Inference Pipelines
├── tests/               # Pruebas unitarias organizadas por módulo
├── pyproject.toml       # Configuración y dependencias
└── uv.lock              # Versiones exactas de dependencias
```

## Referencias

- [Arquitectura FTI](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines)
- [Estructura de capas de datos](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
- [Dataset ILPD](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)
