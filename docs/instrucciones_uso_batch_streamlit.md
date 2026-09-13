# Procesamiento Batch — Aplicación Streamlit

## 1. Objetivo

La aplicación permite cargar múltiples registros clínicos mediante un archivo CSV y
obtener las predicciones generadas por el modelo.

## 2. Acceso

La aplicación está disponible en:

<https://pacientes-higado-upb.streamlit.app/>

## 3. Formato del archivo de entrada

El archivo debe estar en formato CSV y contener las siguientes 10 variables clínicas
RAW:

- Edad.
- Género.
- Bilirrubina total.
- Bilirrubina directa.
- Fosfatasa alcalina.
- ALT.
- AST.
- Proteínas totales.
- Albúmina.
- Ratio albúmina/globulina.

El archivo puede contener múltiples registros. Se incluye un archivo de ejemplo en:

`data/examples/pacientes_higado_input_example.csv`

## 4. Uso del modo Batch

1. Accede a la aplicación mediante la URL pública.
2. Selecciona el modo `Batch`.
3. Carga el archivo CSV.
4. Ejecuta el procesamiento.
5. Revisa la tabla de predicciones.
6. Descarga el resultado.

## 5. Resultado

La aplicación genera un archivo CSV con estas columnas:

| Columna | Descripción |
| --- | --- |
| `record_id` | Identificador técnico generado para cada registro procesado. |
| `prediction` | Clase predicha por el modelo: `0` o `1`. |
| `positive_probability` | Score o probabilidad calculada por el modelo para la clase positiva. |

El archivo de salida se descarga en formato CSV. Se incluye un ejemplo en:

`data/examples/pacientes_higado_output_example.csv`

## 6. Ejemplo de salida

```csv
record_id,prediction,positive_probability
0,0,0.0006410318589396604
1,1,0.9972031913656323
2,0,0.00014109461155054997
3,0,9.312336547025598e-05
4,0,0.00011932326631504232
```

## 7. Validaciones

La aplicación valida el archivo antes de realizar la inferencia. Los registros deben
contener las columnas requeridas y cumplir el formato esperado para las variables
clínicas.

## 8. Consideraciones y limitaciones

- La aplicación utiliza un modelo previamente entrenado.
- El procesamiento Batch realiza inferencia y no entrenamiento.
- La predicción no constituye un diagnóstico médico.
- El resultado depende de las características y limitaciones del conjunto de datos
  utilizado para entrenar el modelo.
