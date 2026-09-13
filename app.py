"""Interfaz Streamlit para predicción online y batch."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from pipelines.inference_pipeline.inference_pipeline import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    load_model,
    load_threshold,
    predict,
    prepare_inference_features,
)


@st.cache_resource
def get_model() -> Pipeline:
    return load_model(DEFAULT_MODEL_PATH)


@st.cache_data
def get_threshold() -> float:
    return float(load_threshold(DEFAULT_METADATA_PATH))


def _predict_frame(data: pd.DataFrame) -> pd.DataFrame:
    features = prepare_inference_features(data)
    return predict(get_model(), features, get_threshold())


def _online_form() -> pd.DataFrame | None:
    st.subheader("Predicción online")
    st.caption("Ingresa las variables clínicas originales. Los ratios se calculan automáticamente.")
    values: dict[str, object] = {
        "Age": st.number_input("Edad", min_value=1.0, value=45.0),
        "Gender": st.selectbox("Género", ["Male", "Female"]),
        "Total_Bilirubin": st.number_input("Bilirrubina total", min_value=0.0, value=1.0),
        "Direct_Bilirubin": st.number_input("Bilirrubina directa", min_value=0.0, value=0.3),
        "Alkaline_Phosphotase": st.number_input("Fosfatasa alcalina", min_value=0.0, value=200.0),
        "Alamine_Aminotransferase": st.number_input("ALT", min_value=0.0, value=30.0),
        "Aspartate_Aminotransferase": st.number_input("AST", min_value=0.0, value=35.0),
        "Total_Protiens": st.number_input("Proteínas totales", min_value=0.0, value=6.5),
        "Albumin": st.number_input("Albúmina", min_value=0.0, value=3.2),
        "Albumin_and_Globulin_Ratio": st.number_input(
            "Ratio albúmina/globulina", min_value=0.0, value=1.0
        ),
    }
    if st.button("Predecir", type="primary"):
        return pd.DataFrame([values])
    return None


def _batch_form() -> None:
    st.subheader("Predicción batch")
    st.caption("Carga un CSV con las 10 variables clínicas originales y Gender.")
    uploaded = st.file_uploader("CSV de pacientes", type="csv")
    if uploaded is None:
        return
    try:
        data = pd.read_csv(uploaded)
        predictions = _predict_frame(data)
    except (TypeError, ValueError, FileNotFoundError) as error:
        st.error(str(error))
        return
    st.dataframe(predictions, use_container_width=True)
    st.download_button(
        "Descargar predicciones",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="pacientes_higado_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Predicción de problemas hepáticos", page_icon="🧪")
    st.title("Predicción de problemas hepáticos")
    st.write("Aplicación de inferencia del modelo GaussianNB del proyecto FTI.")
    mode = st.radio("Modo", ["Online", "Batch"], horizontal=True)
    if mode == "Online":
        try:
            data = _online_form()
            if data is not None:
                predictions = _predict_frame(data)
                prediction = int(predictions.loc[0, "prediction"])
                probability = float(predictions.loc[0, "positive_probability"])
                st.metric(
                    "Predicción", "Problemas hepáticos" if prediction else "Sin problemas hepáticos"
                )
                st.write(f"Probabilidad positiva: {probability:.2%}")
        except (TypeError, ValueError, FileNotFoundError) as error:
            st.error(str(error))
    else:
        _batch_form()


if __name__ == "__main__":
    main()
