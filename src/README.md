# Pipelines de características, entrenamiento e inferencia

La estructura del proyecto se basa en:

<https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines>

## Estructura de carpetas

- `src`: código fuente
    - `data`: extracción, validación, procesamiento, transformación y exportación de datos.
    - pipelines:
        - `feature_pipeline`: transforma datos RAW en features y etiquetas.
        - `training_pipeline`: transforma features y etiquetas en un modelo.
        - `inference_pipeline`: recibe datos nuevos y un modelo entrenado para generar predicciones.

La separación FTI del proyecto es:

- **Feature Pipeline:** RAW → validación → limpieza → feature engineering → features.
- **Training Pipeline:** features → entrenamiento → evaluación → modelo.
- **Inference Pipeline:** modelo + datos nuevos → transformación → predicciones.

Los tres pipelines están implementados. Los notebooks se reservan para exploración,
análisis, interpretación y experimentación histórica; la lógica productiva vive en
`src/pipelines/`.
