# Estructura de datos

## Fuente del proyecto

El dataset utilizado en este proyecto es público, fue modificado por el profesor y se
descargó localmente. La fuente efectiva y única es
`data/01_raw/Pacientes_porblemas_higado_india.csv`; los notebooks y procesos deben leer
siempre desde esa ruta y no desde una fuente externa.

## Convención de capas de datos

![Capas de datos](https://docs.kedro.org/en/stable/_images/data_layers.png)

| Carpeta en `data` | Descripción |
| --- | --- |
| `01_raw` | Fuente inicial e inmutable del pipeline. Es la única fuente de verdad y contiene los datos originales, normalmente sin tipar, como el CSV del proyecto. |
| `02_intermediate` | Capa opcional para corregir y tipar los datos RAW. |
| `03_primary` | Datos limpiados y transformados según el dominio, antes de generar features. |
| `04_feature` | Features resultantes del Feature Pipeline, organizadas para el análisis. |
| `05_model_input` | Datos preparados para el entrenamiento y sus artefactos asociados. |
| `06_models` | Modelos de machine learning serializados. |
| `07_model_output` | Resultados generados por los modelos. |
| `08_reporting` | Datos y resultados preparados para informes y visualizaciones. |

## Referencias

* <https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention>
* <https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71>
