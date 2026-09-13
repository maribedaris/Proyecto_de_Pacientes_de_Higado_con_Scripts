# Inference Pipeline

## Qué hace

Lee datos clínicos RAW o features nuevas y genera una predicción por registro usando el modelo entrenado.
No entrena el modelo ni ajusta transformaciones con los datos de entrada.

## Entrada y modelo

Por defecto lee `data/04_feature/Pacientes_porblemas_higado_features.parquet`.
También acepta CSV. La entrada puede contener las 10 variables clínicas originales,
en cuyo caso el adaptador calcula `Ratio_Bilirrubina_Directa` y `Ratio_De_Ritis`, o
las 12 columnas de `FEATURE_COLUMNS` ya preparadas. `Dataset` es opcional y, si existe,
no se usa como predictor.

El artifact es `data/06_models/pacientes_higado_gaussiannb.joblib`. Es un `Pipeline`
de scikit-learn que contiene imputación, escalado, codificación y `GaussianNB`. El
threshold OOF utilizado para convertir probabilidades en clases se lee desde
`data/07_model_output/pacientes_higado_metadata.json`.

## Salida y ejecución

Las predicciones se guardan por defecto en
`data/07_model_output/pacientes_higado_predictions.parquet`. Incluyen el índice de
entrada (`record_id`), la clase (`prediction`) y la probabilidad positiva.

Desde la raíz del proyecto:

```bash
PYTHONPATH=src python -m pipelines.inference_pipeline.inference_pipeline
```

Las rutas se pueden cambiar con `--input-path`, `--model-path`, `--metadata-path` y
`--output-path`. La inferencia reutiliza directamente las transformaciones aprendidas
dentro del artifact mediante `predict_proba`; nunca ejecuta `fit` sobre datos nuevos.

## Pruebas

```bash
pytest tests/pipelines/inference_pipeline/test_inference_pipeline.py
```
