# Training Pipeline

## Qué hace

Entrena un `GaussianNB` con las features generadas por el Feature Pipeline y evalúa
su capacidad de generalización sin usar el test para tomar decisiones de entrenamiento.

La entrada es:

`data/04_feature/Pacientes_porblemas_higado_features.parquet`

El pipeline valida las columnas y etiquetas, separa train/test de forma estratificada
(80/20, `random_state=42`) y comprueba que ambos conjuntos estén correctamente
separados antes de entrenar.

## Entrenamiento y validación

`GridSearchCV` selecciona `var_smoothing` con `f1_macro` usando únicamente train.
Después, `StratifiedKFold` comprueba el modelo en distintas particiones de train,
manteniendo una proporción similar de las dos clases en cada fold.

En cada fold se clona el pipeline y se ajustan el imputador, el escalado, la
codificación y el modelo solo con la parte de entrenamiento. El fold de validación
solo se transforma y evalúa. El test permanece separado hasta la evaluación final.

La validación cruzada usa threshold fijo `0.5` para que el threshold no se seleccione
con las etiquetas del mismo fold. El threshold optimizado con probabilidades OOF de
train se utiliza después para las métricas finales de train y test.

Se comparan resultados de train, cross-validation (`mean +/- std`) y test. Se generan:
`accuracy`, `precision`, `recall`, `f1`, `f1_macro`, `specificity`, `roc_auc`,
`average_precision` y matriz de confusión.

El diagnóstico usa `roc_auc`: una brecha train-CV superior a `0.10` indica posible
overfitting; valores de train y CV inferiores a `0.60` indican posible underfitting;
y una diferencia CV-test superior a `0.10` indica una posible brecha de generalización.
El diagnóstico solo informa y recomienda revisar el modelo; no cambia sus parámetros
automáticamente.

## Salidas y ejecución

Se guardan por defecto:

- `data/06_models/pacientes_higado_gaussiannb.joblib`
- `data/07_model_output/pacientes_higado_metrics.json`
- `data/07_model_output/pacientes_higado_metadata.json`

La metadata y las métricas contienen los resultados de train, los folds y su resumen,
el test y el diagnóstico de generalización.

Desde la raíz del proyecto:

```bash
PYTHONPATH=src python -m pipelines.training_pipeline.train_pipeline
```

Las rutas se pueden cambiar con `--input-path`, `--model-path`, `--metrics-path` y
`--metadata-path`. Las pruebas se ejecutan con:

```bash
pytest tests/pipelines/training_pipeline/test_train_pipeline.py
```
