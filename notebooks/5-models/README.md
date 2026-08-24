# Etapa 5 — Modelos: resumen ejecutivo

Resultado consolidado de la selección y validación de modelos, tras la corrección de
data leakage de la etapa de feature engineering (PR #24). Todos los números de esta
tabla provienen de la evaluación sobre el mismo conjunto de prueba corregido
(`x_test_v2`, 114 pacientes, deduplicado por paciente).

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

## Modelo recomendado

**GaussianNB con umbral de decisión 0.0011**, entrenado sobre `x_train_v2` en el
notebook 02. Artefacto: `models/pacientes_higado_clasificacion-gaussiannb-v2.joblib`
(+ `.json` con el umbral y su checksum sha256). Ver
[models/README.md](../../models/README.md) para las instrucciones de uso — en
particular, **nunca llamar `predict()` directo**: el umbral no está empaquetado en el
pipeline y aplicar el corte 0.5 por defecto degrada el f1_macro de 0.741 a 0.631.

## El hallazgo más importante del proceso

El leakage original — pacientes duplicados entre entrenamiento y prueba que la
limpieza no detectaba — no solo inflaba las métricas: **cambiaba qué modelo parecía
ganador**. En la selección manual, el filtro que rankeaba por recall de la clase
mayoritaria (inflado por el mismo leakage y el desbalance 71/29) descartó a
GaussianNB, que resultó ser el mejor modelo del proyecto cuando finalmente se le dio
oportunidad. En el AutoML, la misma búsqueda FLAML con el mismo presupuesto y espacio
eligió un extra_tree con el split contaminado y un k-NN con el split corregido — el
ganador cambió por completo, y las conclusiones de la corrida contaminada no se
sostuvieron. La lección: la limpieza de datos y el split correcto importan más que la
elección del algoritmo. Ningún tuning rescata un modelo entrenado y evaluado sobre
datos equivocados, y ninguna métrica es confiable hasta que la validación lo sea.

## Dónde está el detalle

| Notebook | Contenido | Corrección aplicada |
|---|---|---|
| [01-heuristic_model](01-heuristic_model-mba-2026-08-19.ipynb) | Baseline heurístico clínico (regla de 4 pruebas hepáticas) | Splits v2 + advertencia de baseline informado por dominio |
| [02-basic_algorithms_model_selection](02-basic_algorithms_model_selection-mba-2026-08-19.ipynb) | Comparación de 7 familias, tuning, umbrales, **GaussianNB ganador** | Splits v2 + permutación + re-tuning con f1_macro + hallazgo del filtro sesgado |
| [03-first_model](03-first_model-mba-2026-08-19.ipynb) | Primer modelo ML (RF) | Splits v2 + re-tuning con f1_macro |
| [04-experiment-track-model](04-experiment-track-model-mba-2026-08-19.ipynb) | Tracking de experimentos (MLflow) | Splits v2 + scoring f1_macro |
| [05-model_validation](05-model_validation-mba-2026-08-19.ipynb) | Validación de datos y proceso | Splits v2 + re-verificación de fugas |
| [06-automl_model_selection](06-automl_model_selection-mba-2026-08-20.ipynb) | AutoML (FLAML) | Splits v2 — el ganador cambió de extra_tree a k-NN |

Los splits corregidos y sus checksums viven en `data/05_model_input/`
(`metadata_v2.json`); la causa raíz del leakage está documentada en el notebook de
[4-feat_eng](../4-feat_eng/01-basic-feature-engineering-pipeline-mba-2026-08-19.ipynb)
y en el PR #24.
