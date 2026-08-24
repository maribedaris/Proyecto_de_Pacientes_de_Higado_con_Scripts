# Modelos guardados del proyecto

Esta carpeta contiene los artefactos de modelos generados por los notebooks de la
etapa 5 (`notebooks/5-models/`). Estado al cierre de la corrección de data leakage
(splits v2, ver PR #24):

## `pacientes_higado_clasificacion-gaussiannb-v2.joblib` (+ `.json`) — modelo recomendado

Pipeline completo (imputación por mediana + escalado + one-hot + GaussianNB con
`var_smoothing=1e-10`), entrenado sobre `x_train_v2` en el notebook 02. Es el primer
modelo de ML que supera al baseline heurístico del notebook 01 (f1_macro 0,741 vs
0,659; roc_auc 0,861 vs 0,815 en test v2).

**Importante — cómo usarlo:** el umbral de decisión óptimo (0,0011) **no** está
empaquetado dentro del pipeline; viaja en el JSON adjunto junto con su checksum
sha256 y la fecha de generación. Nunca llamar `predict()` directo (aplicaría el corte
0,5 por defecto y degradaría el f1_macro de 0,741 a 0,631):

```python
import json
from pathlib import Path

from joblib import load

pipe = load("models/pacientes_higado_clasificacion-gaussiannb-v2.joblib")
meta = json.loads(Path("models/pacientes_higado_clasificacion-gaussiannb-v2.json").read_text())
umbral = meta["umbral_decision"]  # 0.0011

probabilidades = pipe.predict_proba(X_nuevos)[:, 1]
predicciones = (probabilidades >= umbral).astype(int)  # nunca pipe.predict(X_nuevos)
```

## `pacientes_higado_clasificacion-random_forest-v1.joblib` — referencia histórica

Random forest (profundidad 4) generado por el notebook 02. Se conserva como
referencia histórica del modelo que el filtro sesgado —que rankeaba por recall de la
clase mayoritaria— favorecía antes de la corrección metodológica; el propio notebook
lo documenta así.

## `pacientes_higado_random_forest-v1.joblib` — artefacto activo de los notebooks 03/04/05

Random forest (profundidad 10) que persisten activamente los notebooks 03, 04 y 05 en
sus celdas de guardado, y que el notebook 04 registra además como artefacto de su run
de MLflow. Al re-ejecutar esos notebooks sobre los splits v2, este archivo se
sobrescribe con la última versión re-entrenada.

**Nota sobre el nombre:** no sigue la convención `pacientes_higado_clasificacion-*`
de los otros dos por una inconsistencia histórica entre notebooks. Se documenta aquí
en lugar de renombrarlo, para no reabrir notebooks ya cerrados y validados; cualquier
unificación de nombres debería hacerse en un PR dedicado que actualice también las
celdas de guardado y carga de 03/04/05.


## `data/06_models/pacientes_higado_clasificacion-kneighbor-v1.joblib` — evidencia del AutoML (NO recomendado)

K-NN (k=4) con umbral 0,26, ganador de la búsqueda FLAML del notebook 06 **sobre los
splits corregidos v2**. **No es el modelo recomendado del proyecto**: perdió contra el
GaussianNB del notebook 02 (f1_macro 0,545 vs 0,741) y contra el baseline heurístico
(0,659). Se conserva como evidencia del proceso de AutoML y de un hallazgo importante:
con el split anterior (contaminado), la misma búsqueda elegía un extra_tree con
métricas que parecían buenas y eran un espejismo del leakage.

**Nota sobre la ubicación:** este artefacto vive en `data/06_models/` (no en `models/`)
porque sigue la convención de capas de datos del proyecto, tipo Kedro, documentada en
`data/README.md`: `06_models` es la capa designada para modelos serializados. Los
artefactos en `models/` provienen de los notebooks 02–05, que usan esa otra ruta; la
coexistencia de ambas convenciones es histórica y se documenta aquí tal cual.
