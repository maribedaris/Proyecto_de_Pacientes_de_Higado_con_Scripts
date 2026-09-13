# Instrucciones de uso — Aplicación Streamlit

## 1. Objetivo de la aplicación

La aplicación permite realizar una **predicción online** utilizando el modelo GaussianNB desarrollado en el proyecto de clasificación de pacientes con posibles problemas hepáticos.

La aplicación recibe los datos clínicos de un paciente y muestra la clase predicha junto con el resultado asociado al modelo.

## 2. Acceso a la aplicación

La aplicación se encuentra publicada en Streamlit Community Cloud y puede accederse mediante la siguiente URL:

**<https://pacientes-higado-upb.streamlit.app/>**

No es necesario instalar el proyecto ni ejecutar código para utilizar la versión publicada.

## 3. Realizar una predicción online

Para realizar una predicción:

1. Ingresa a la URL de la aplicación.
2. Selecciona el modo **Online**.
3. Completa los datos clínicos solicitados.
4. Selecciona **Predecir**.
5. Consulta el resultado generado por el modelo.

### Datos solicitados

| Variable                 | Descripción                                                   |
| ------------------------ | ------------------------------------------------------------- |
| Edad                     | Edad del paciente en años.                                    |
| Género                   | Género registrado para el paciente.                           |
| Bilirrubina total        | Concentración de bilirrubina total en sangre (mg/dL).         |
| Bilirrubina directa      | Concentración de bilirrubina directa en sangre (mg/dL).       |
| Fosfatasa alcalina       | Nivel de fosfatasa alcalina (U/L).                            |
| ALT                      | Alanina aminotransferasa, también conocida como SGPT (U/L).   |
| AST                      | Aspartato aminotransferasa, también conocida como SGOT (U/L). |
| Proteínas totales        | Concentración de proteínas séricas totales (g/dL).            |
| Albúmina                 | Concentración de albúmina sérica (g/dL).                      |
| Ratio albúmina/globulina | Relación entre albúmina y globulina.                          |

> **Nota:** La aplicación calcula automáticamente los **ratios clínicos derivados utilizados internamente por el modelo** a partir de las variables originales. El usuario no necesita realizar estos cálculos.

## 4. Validación de los datos

Antes de realizar la predicción, la aplicación valida los valores ingresados.

Entre las validaciones realizadas se encuentra la relación entre las variables de bilirrubina:

* La **bilirrubina directa no debe ser superior a la bilirrubina total**.

Si se identifica un valor no válido, la aplicación muestra un mensaje de error y solicita corregir los datos antes de realizar la predicción.

## 5. Resultado de la predicción

Después de seleccionar **Predecir**, la aplicación muestra el resultado correspondiente al paciente ingresado.

El resultado incluye:

* **Clase predicha:** clasificación generada por el modelo.
* **Probabilidad positiva o score:** valor asociado a la predicción del modelo.

La clasificación utiliza automáticamente el **umbral definido durante la evaluación del modelo**, por lo que el usuario no necesita realizar cálculos adicionales.

### Interpretación

El resultado debe entenderse como una **predicción del modelo**, no como un diagnóstico médico.

El valor numérico mostrado por la aplicación no corresponde a una probabilidad clínica calibrada. Por esta razón, un valor más alto representa una mayor evidencia relativa para la clase positiva dentro del modelo, pero **no debe interpretarse como un porcentaje de probabilidad de que una persona tenga una enfermedad hepática**.

## 6. Consideraciones y limitaciones

La aplicación corresponde a un proyecto académico y educativo.
El modelo realiza inferencia a partir de los datos ingresados y no se entrena durante la consulta.
La predicción no constituye un diagnóstico médico ni sustituye la valoración de un profesional de la salud.
El desempeño del modelo depende de las características y limitaciones del conjunto de datos utilizado durante su desarrollo.
Los resultados deben interpretarse dentro del contexto del proyecto y no utilizarse como único criterio para tomar decisiones clínicas.
