# Etapa 5 — Modelos históricos: resumen ejecutivo

Resultado consolidado de la selección y validación de modelos. Todos los números de esta
tabla provienen de la evaluación sobre el mismo conjunto de prueba (`x_test_v2`, 114
pacientes, deduplicado por paciente).

## Tabla comparativa final (test v2, ordenada por f1_macro)

| Modelo | f1_macro | recall clase 1 (enfermos) | recall clase 0 (sanos) | roc_auc | Notebook |
|---|---|---|---|---|---|
| **GaussianNB + umbral 0.0011** ⭐ | **0.741** | 0.741 | 0.818 | 0.861 | [02](02-basic_algorithms_model_selection-mba-2026-08-19.ipynb) |
| svc + umbral 0.284 | 0.692 | 0.716 | 0.727 | 0.831 | [02](02-basic_algorithms_model_selection-mba-2026-08-19.ipynb) |
| Baseline heurístico clínico | 0.659 | 0.926 | 0.588 | 0.815 | [01](01-heuristic_model-mba-2026-08-19.ipynb) |
| Random Forest (tuneado, mejor de 02–05) | 0.556 | 0.951 | 0.182 | 0.790 | [03](03-first_model-mba-2026-08-19.ipynb) |
| k-NN (AutoML FLAML, umbral 0.26) | 0.545 | 0.889 | 0.212 | 0.676 | [06](06-automl_model_selection-mba-2026-08-20.ipynb) |

*Nota: svc con umbral aparece con f1_macro 0.692 en segunda posición de la tabla por
métrica, pero su selección requería el paso extra de calibración de umbral; el
GaussianNB fue validado además con tuning de `var_smoothing` y prueba de permutación.
Los notebooks 04 y 05 re-entrenan variantes de RF con f1_macro 0.515–0.556 y no
aportan un modelo distinto a la tabla.*

## Modelo recomendado en la etapa histórica

**GaussianNB con umbral de decisión 0.0011**, entrenado sobre `x_train_v2` en el
notebook 02. Artefacto: `models/pacientes_higado_clasificacion-gaussiannb-v2.joblib`
(+ `.json` con el umbral y su checksum sha256). Ver
[models/README.md](../../models/README.md) para las instrucciones de uso — en
particular, **nunca llamar `predict()` directo**: el umbral no está empaquetado en el
pipeline y aplicar el corte 0.5 por defecto degrada el f1_macro de 0.741 a 0.631.

Este resultado pertenece al proceso histórico de selección. El modelo operativo actual
lo genera `src/pipelines/training_pipeline/` y lo consume
`src/pipelines/inference_pipeline/`.

## Consideraciones metodológicas

Durante la revisión del pipeline se identificó data leakage porque registros del mismo
paciente podían aparecer simultáneamente en entrenamiento y prueba. Esto contaminaba
la evaluación y producía métricas optimistas, afectando la comparación y selección del
modelo.

La corrección consistió en separar adecuadamente los pacientes antes de ajustar el
preprocesamiento y los modelos, y en evaluar después sobre un conjunto de prueba
independiente. Los resultados posteriores a esta corrección son los que deben tomarse
como referencia válida. La preparación de features, los splits y la validación del
proceso se describen en los notebooks enlazados a continuación.

## Dónde está el detalle

| Notebook | Contenido | Características del proceso |
|---|---|---|
| [01-heuristic_model](01-heuristic_model-mba-2026-08-19.ipynb) | Línea base heurística clínica (regla de 4 pruebas hepáticas) | Evaluación con el conjunto de prueba v2 |
| [02-basic_algorithms_model_selection](02-basic_algorithms_model_selection-mba-2026-08-19.ipynb) | Comparación de 7 familias, ajuste, umbrales, **GaussianNB ganador** | Permutación y selección por `f1_macro` |
| [03-first_model](03-first_model-mba-2026-08-19.ipynb) | Primer modelo ML (RF) | Ajuste por `f1_macro` |
| [04-experiment-track-model](04-experiment-track-model-mba-2026-08-19.ipynb) | Seguimiento de experimentos (MLflow) | Registro de parámetros y métricas |
| [05-model_validation](05-model_validation-mba-2026-08-19.ipynb) | Validación de datos y proceso | Revisión de particiones y métricas |
| [06-automl_model_selection](06-automl_model_selection-mba-2026-08-20.ipynb) | AutoML (FLAML) | Comparación automática de modelos |

Los splits y sus checksums viven en `data/05_model_input/` (`metadata_v2.json`). La
descripción de la preparación de features está en el notebook de
[4-feat_eng](../4-feat_eng/01-basic-feature-engineering-pipeline-mba-2026-08-19.ipynb).
