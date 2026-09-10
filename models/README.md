# Modelos guardados del proyecto

## Modelo histórico: `pacientes_higado_clasificacion-gaussiannb-v2.joblib`

Este es el modelo histórico entrenado en formato `.joblib` y el único artefacto en la raíz de
`models/`. Es un pipeline completo (imputación por mediana +
escalado + one-hot + GaussianNB con `var_smoothing=1e-10`), entrenado sobre
`x_train_v2` en el notebook `02-basic_algorithms_model_selection`, y es el primer
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

El modelo operativo actual lo genera `src/pipelines/training_pipeline/` en
`data/06_models/` y lo consume `src/pipelines/inference_pipeline/`. Este directorio
conserva el modelo anterior y su metadata como evidencia de la selección realizada en
notebooks.

## Evidencia del proceso de selección

Los artefactos de los modelos que participaron en la comparación —y perdieron— viven
en [`models/experiments/`](experiments/README.md), documentados ahí como evidencia
del proceso de selección, no como candidatos finales.
