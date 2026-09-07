# Evidencia del proceso de selección de modelos

Los artefactos de esta carpeta **no son candidatos finales**: son modelos evaluados
durante la etapa 5 que no superaron al GaussianNB del notebook 02 (ni, en algunos
casos, al baseline heurístico del notebook 01). Se conservan como evidencia del
proceso de selección y de la corrección de data leakage, para que la comparación
documentada en los notebooks sea reproducible.

| Artefacto | Qué es | Por qué se conserva |
|---|---|---|
| `pacientes_higado_clasificacion-gaussiannb-v2.joblib` está en `models/` (raíz) — ver [models/README.md](../README.md) | Entregable final | Ganador de la comparación (f1_macro 0,741) |
| `pacientes_higado_random_forest-v1.joblib` | Random forest (profundidad 10) que persisten los notebooks 03, 04 y 05 sobre los splits v2; el 04 lo registra además como artefacto de su run de MLflow | Evidencia del primer modelo ML (03), del tracking de experimentos (04) y de la validación (05). Su f1_macro (0,505–0,556) nunca alcanzó al baseline |
| `pacientes_higado_clasificacion-random_forest-v1.joblib` | Random forest (profundidad 4) del notebook 02 | Referencia histórica del modelo que el filtro sesgado —que rankeaba por recall de la clase mayoritaria— favorecía antes de la corrección metodológica |
| `pacientes_higado_clasificacion-kneighbor-v1.joblib` | K-NN (k=4) con umbral 0,26, ganador de la búsqueda FLAML del notebook 06 **sobre los splits corregidos** | Evidencia del AutoML: con el split contaminado la misma búsqueda elegía un extra_tree con métricas que parecían buenas y eran espejismo del leakage (f1_macro 0,545 vs 0,741 del ganador) |

| `pacientes_higado_clasificacion-extra_tree-v1.joblib` | Extra trees (11 árboles, `class_weight="balanced"`) que la búsqueda FLAML eligió **con el split contaminado** (antes de la corrección) | Evidencia directa de que el leakage cambiaba al ganador del AutoML: sus métricas parecían buenas (recall 1,000) y eran espejismo — especificidad 0,000, comportamiento de dummy |

**Nota sobre la ubicación:** estos artefactos se conservan en esta carpeta como
evidencia de los modelos evaluados. El modelo seleccionado se encuentra en `models/`,
separado de las alternativas experimentales; las celdas de guardado de los notebooks
03–06 apuntan a esta ubicación.
