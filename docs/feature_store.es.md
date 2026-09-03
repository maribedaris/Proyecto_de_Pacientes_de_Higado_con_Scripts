# Feature Store: histórico clínico

El objetivo de este módulo es conservar en Hopsworks el histórico procesado que alimenta las etapas posteriores del proyecto. No crea un pipeline autónomo de entrenamiento o inferencia: prepara, valida, almacena y vuelve a leer el Feature Group.

La fuente es `data/01_raw/Pacientes_porblemas_higado_india.csv`. El archivo raw se lee seleccionando explícitamente las diez columnas originales y `Dataset`; por tanto, columnas auxiliares como `Unnamed: 0` no se convierten en features.

## Datos almacenados

El Feature Group contiene las diez variables predictoras originales (`Age`, `Gender` y las ocho pruebas clínicas), los cocientes `Ratio_Bilirrubina_Directa` y `Ratio_De_Ritis`, y `Dataset` como etiqueta. `record_id` es una clave técnica estable calculada con SHA-256 sobre los valores canónicos de las diez variables originales. No depende del índice ni de la etiqueta, no inventa un identificador de paciente y no se utiliza como predictor. Si el mismo perfil aparece con etiquetas diferentes, la validación detecta la colisión y rechaza la carga.

La preparación histórica recupera valores faltantes compatibles entre filas casi duplicadas antes de deduplicar. Después elimina duplicados exactos, descarta filas sin target y aplica `Direct_Bilirubin <= Total_Bilirubin`. Los cocientes se generan mediante `CocientesClinicos`, el transformador compartido en `src/pipelines/feature_pipeline/transformers.py`. Una división por cero se convierte en `NaN`; los valores faltantes de pruebas clínicas y cocientes se conservan para su tratamiento posterior.

## Configuración y ejecución

Instala la dependencia opcional y ejecuta desde la raíz del proyecto:

```bash
uv sync --all-extras --dev
PYTHONPATH=src .venv/bin/python -m data.feature_store
```

La preparación y sus validaciones pueden ejecutarse sin conectarse a Hopsworks:

```bash
PYTHONPATH=src .venv/bin/python -m data.feature_store --validate-only
```

Configura estas variables de entorno antes de una carga real:

- `HOPSWORKS_PROJECT`
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_HOST` (opcional)
- `HOPSWORKS_FEATURE_GROUP` (opcional)
- `HOPSWORKS_FEATURE_GROUP_VERSION` (opcional)

La API key debe proporcionarse mediante el gestor de secretos o el entorno de ejecución. No debe escribirse en el repositorio.

## Feature Group y carga

Por defecto se utiliza el Feature Group `pacientes_higado_features`, versión `1`, con `record_id` como `primary_key` y `online_enabled=False`. Se puede cambiar el nombre y la versión mediante variables de entorno. El módulo crea el grupo si no existe o reutiliza el grupo indicado, inserta esperando la finalización del job y lee de nuevo el grupo para verificar esquema, claves y presencia de los registros enviados.

La clave determinística permite repetir la preparación con el mismo resultado. El comportamiento exacto de una segunda inserción de las mismas claves depende de la configuración y versión de Hopsworks; debe verificarse durante la primera carga real para confirmar si actualiza, ignora o duplica filas. La validación posterior no presenta una carga local como evidencia remota.

## Validaciones

Antes de insertar se comprueban las columnas, tipos, etiquetas `1`/`2`, género no nulo válido (los nulos se permiten), valores finitos, valores clínicos no negativos, la consistencia de bilirrubinas, los nulos permitidos y la unicidad de `record_id`. Tras insertar, el proceso vuelve a leer el Feature Group y verifica su esquema, cantidad de registros, ausencia de duplicados y presencia de todos los identificadores enviados. Si Hopsworks no está disponible, `--validate-only` deja evidencia únicamente de la validación local y no simula una carga exitosa.
