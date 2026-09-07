# Pipelines de características, entrenamiento e inferencia

La estructura del proyecto se basa en:

<https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines>

## Estructura de carpetas

- `src`: código fuente
    - `data`: extracción, validación, procesamiento, transformación y exportación de datos.
    - `model`: entrenamiento, evaluación, validación y exportación de modelos.
    - `inference`: predicción, publicación y monitorización de modelos.
    - pipelines:
        - `feature_pipeline`: transforma datos RAW en features y etiquetas.
        - `training_pipeline`: transforma features y etiquetas en un modelo.
        - `inference_pipeline`: recibe datos nuevos y un modelo entrenado para generar predicciones.

La separación FTI del proyecto es:

- **Feature Pipeline:** RAW → validación → limpieza → feature engineering → features.
- **Training Pipeline:** features → entrenamiento → evaluación → modelo.
- **Inference Pipeline:** modelo + datos nuevos → transformación → predicciones.

Actualmente el Feature Pipeline y el Training Pipeline están implementados. El
Inference Pipeline aún no está implementado. Los notebooks se reservan para
exploración, análisis y experimentación; la lógica productiva debe vivir en
`src/pipelines/`.
