# Notebooks del proyecto

Los notebooks siguen esta convención de nombres:

`[##.#]-[initials]-[short_description]-[yyyy_mm_dd].ipynb`

* `##.#`: número y versión del notebook.
* `initials`: iniciales de la persona que lo creó.
* `short_description`: descripción corta separada por `_`.
* `yyyy_mm_dd`: fecha de creación del notebook.

Ejemplos:

```text
01-jrz-data_exploration-2024_10_02.ipynb
02.1-jrz_data_raw_analysis-2024_10_08.ipynb
02.2-jrz_data_raw_analysis-2024_11_21.ipynb
```

## Estructura de carpetas

### Datos

```text
data/
├── 01_raw/                   # Datos originales, sin modificaciones
├── 02_intermediate/          # Datos de transformaciones intermedias
├── 03_primary/               # Datos preparados para análisis y modelado
└── 04_reporting/             # Datos y resultados para informes
```

### Notebooks

```text
notebooks/
├── 1-data                    # Extracción y limpieza de datos
├── 2-exploration             # Análisis exploratorio de datos (EDA)
├── 3-analysis                # Análisis estadístico y pruebas de hipótesis
├── 4-feat_eng                # Creación, selección y transformación de features
├── 5-models                  # Entrenamiento, evaluación y ajuste de hiperparámetros
├── 6-interpretation          # Interpretación del modelo
├── 7-deploy                  # Empaquetado y estrategias de despliegue
├── 8-reports                 # Informes, resultados y conclusiones
└── notebook_template.ipynb
```
