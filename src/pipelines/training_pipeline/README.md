# Training Pipeline

## Propósito

Entrena y evalúa el modelo GaussianNB para clasificar pacientes con problemas de
hígado a partir de las features generadas por el Feature Pipeline.

## Entrada

La entrada productiva es:

`data/04_feature/Pacientes_porblemas_higado_features.parquet`

Contiene las variables clínicas, `Gender`, los dos cocientes derivados y la etiqueta
`Dataset`. El pipeline no utiliza `data/05_model_input`.

## Proceso

1. Valida el esquema, las etiquetas `1` y `2` y la presencia de ambas clases.
2. Recodifica `Dataset == "1"` como clase positiva.
3. Divide los datos en train/test con proporción 80/20, estratificación y `random_state=42`.
4. Ajusta imputación, escalado y codificación categórica dentro de un pipeline de
   scikit-learn.
5. Selecciona `var_smoothing` con `GridSearchCV`, 10 folds estratificados,
   `random_state=42`, `f1_macro` y `np.logspace(-10, -2, 9)`.
6. Obtiene probabilidades OOF sobre train y selecciona el threshold que maximiza
   `f1_macro`.
7. Reentrena el mejor pipeline con todo train y evalúa test una sola vez.

## Modelo y evaluación

El modelo final es `GaussianNB`. Se generan `accuracy`, `precision`, `recall`, `f1`,
`f1_macro`, `specificity`, `roc_auc`, `average_precision` y matriz de confusión.

Por defecto se persisten:

- `data/06_models/pacientes_higado_gaussiannb.joblib`
- `data/07_model_output/pacientes_higado_metrics.json`
- `data/07_model_output/pacientes_higado_metadata.json`

## Ejecución

Desde la raíz del proyecto:

```bash
PYTHONPATH=src python -m pipelines.training_pipeline.train_pipeline
```

En PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m pipelines.training_pipeline.train_pipeline
```

También pueden cambiarse las rutas con `--input-path`, `--model-path`,
`--metrics-path` y `--metadata-path`.

Las pruebas se ejecutan con:

```bash
pytest tests/pipelines/training_pipeline/test_train_pipeline.py
```

## Controles contra data leakage

- La entrada es el Parquet del Feature Pipeline, no los splits históricos de
  `data/05_model_input`.
- El preprocesamiento se ajusta dentro del pipeline y de cada fold de validación.
- GridSearchCV solo observa train.
- El threshold se calcula exclusivamente con probabilidades OOF de train.
- Test no participa en hiperparámetros ni threshold; se usa una única vez para la
  evaluación final.
