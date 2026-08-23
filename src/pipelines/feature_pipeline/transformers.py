"""Transformadores de características reutilizables del proyecto.

Contiene los transformadores personalizados compatibles con scikit-learn que
usan los pipelines de feature engineering y los notebooks de modelado, de modo
que los modelos persistidos con joblib puedan recargarse desde cualquier
proceso Python (la clase debe ser importable desde un módulo estable para que
la deserialización funcione fuera del notebook que la creó).
"""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CocientesClinicos(BaseEstimator, TransformerMixin):
    """Transformador compatible con scikit-learn que crea variables clínicas derivadas.

    Genera, por paciente, dos cocientes a partir de las pruebas hepáticas:
    - Ratio_Bilirrubina_Directa: bilirrubina directa / bilirrubina total.
    - Ratio_De_Ritis: AST / ALT (cociente de De Ritis).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cocientes = pd.DataFrame(
            {
                "Ratio_Bilirrubina_Directa": X["Direct_Bilirubin"] / X["Total_Bilirubin"],
                "Ratio_De_Ritis": X["Aspartate_Aminotransferase"] / X["Alamine_Aminotransferase"],
            },
            index=X.index,
        )
        # por robustez ante datos nuevos: una división por cero se trata como faltante
        return cocientes.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features: np.ndarray | None = None) -> np.ndarray:
        return np.array(["Ratio_Bilirrubina_Directa", "Ratio_De_Ritis"])
