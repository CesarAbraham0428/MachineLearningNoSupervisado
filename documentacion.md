# Documentación técnica — ClusterLab

> **Versión:** `dev-marco` · **Stack:** Python 3.12 + Streamlit · **Fecha:** Julio 2026

---

## Tabla de contenidos

1. [Descripción general](#1-descripción-general)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [Cómo ejecutar la aplicación](#3-cómo-ejecutar-la-aplicación)
4. [Arquitectura y flujo de datos](#4-arquitectura-y-flujo-de-datos)
5. [Session State](#5-session-state)
6. [Funcionalidades implementadas](#6-funcionalidades-implementadas)
7. [Módulos en desarrollo (pendientes)](#7-módulos-en-desarrollo-pendientes)
8. [API de servicios](#8-api-de-servicios)
9. [Dependencias](#9-dependencias)

---

## 1. Descripción general

**ClusterLab** es una aplicación web construida con Streamlit que implementa un flujo completo de **aprendizaje no supervisado**. Permite a estudiantes e investigadores cargar un dataset, prepararlo, analizarlo estadísticamente y (en versiones futuras) entrenar un modelo K-Means, todo desde una única interfaz web sin necesidad de escribir código.

---

## 2. Estructura del proyecto

```
clusterLab/
│
├── app.py                    # Punto de entrada principal
│
├── views/                    # Capa de presentación (UI)
│   ├── data_view.py          # Pestaña "Datos cargados" (RF-01 a RF-08) ✅
│   ├── statistics_view.py    # Pestaña "Estadística descriptiva" (RF-09) ✅
│   ├── training_view.py      # Pestaña "Entrenamiento" 🔜
│   ├── results_view.py       # Pestaña "Resultados" 🔜
│   └── models_view.py        # Pestaña "Modelos guardados" 🔜
│
├── services/                 # Capa de lógica de negocio
│   ├── dataset_service.py    # Carga, validación, diagnóstico y limpieza ✅
│   ├── statistics_service.py # Estadísticas descriptivas Big Five ✅
│   ├── report_service.py     # Generación de reportes PDF ✅
│   ├── training_service.py   # Entrenamiento K-Means 🔜
│   └── model_service.py      # Persistencia de modelos 🔜
│
├── utils/
│   ├── session_state.py      # Gestión del estado de sesión
│   └── validators.py         # Validadores reutilizables
│
├── assets/
│   └── styles.css            # Sistema visual completo de ClusterLab
│
├── database/                 # Manejo de base de datos (pendiente)
├── storage/                  # Almacenamiento de modelos y reportes
├── tests/                    # Pruebas unitarias
└── requirements.txt
```

---

## 3. Cómo ejecutar la aplicación

```bash
# Activar entorno virtual (Windows)
venv\Scripts\activate

# Ejecutar la aplicación
streamlit run app.py
```

La aplicación abre automáticamente en `http://localhost:8501`.

---

## 4. Arquitectura y flujo de datos

La app sigue un patrón de **capas desacopladas**:

```
Usuario
  │
  ▼
app.py  ──── navegación segmentada (5 pestañas)
  │
  ├── views/          ← renderizado UI con Streamlit
  │     └── llama a services/ para lógica de negocio
  │
  └── services/       ← funciones/clases puras (sin Streamlit)
        └── retornan dataclasses tipados a las vistas
```

**Regla de oro:** los servicios nunca importan `streamlit`.
Toda interacción con `session_state` ocurre exclusivamente en las vistas.

---

## 5. Session State

Claves globales gestionadas por la aplicación:

| Clave | Tipo | Descripción |
|---|---|---|
| `dataframe_cargado` | `pd.DataFrame \| None` | Dataset tal como fue leído del archivo |
| `dataset_original` | `pd.DataFrame \| None` | Alias inmutable de `dataframe_cargado` (nunca se modifica) |
| `dataset_limpio` | `pd.DataFrame \| None` | Dataset tras la limpieza automática; `None` si no se ha limpiado |
| `resultado_limpieza` | `ResultadoLimpieza \| None` | Métricas de la última limpieza ejecutada |
| `nombre_archivo` | `str \| None` | Nombre del archivo cargado |
| `fecha_carga` | `str \| None` | Timestamp de la última carga (`DD/MM/YYYY HH:MM`) |
| `pagina_actual` | `int` | Página activa en la tabla paginada (base 1) |
| `mostrar_carga` | `bool` | Controla la visibilidad del uploader |
| `modelo_entrenado` | `bool` | Indica si existe un modelo activo |
| `modelos_guardados` | `list` | Lista de modelos persistidos |
| `tab_activa` | `int` | Índice de la pestaña activa (0–4) |

**Regla de prioridad para las vistas que consumen datos:**

```
dataset_limpio  →  dataframe_filtrado  →  dataframe_cargado
```

---

## 6. Funcionalidades implementadas

### RF-01 · Carga de archivos

**Archivo:** `views/data_view.py` → `_renderizar_carga()`

- Acepta archivos **CSV** y **Excel** (`.xlsx` / `.xls`).
- Valida que el archivo no esté vacío antes de aceptarlo.
- Al cargar un nuevo archivo invalida automáticamente `dataset_limpio` y `resultado_limpieza`.
- Muestra un spinner durante la carga y un toast de confirmación al terminar.
- El uploader se oculta automáticamente tras una carga exitosa.

---

### RF-02 · Tabla de datos con paginación

**Archivo:** `views/data_view.py` → `_renderizar_tabla_paginada()`

- Muestra el dataset en una tabla nativa de Streamlit.
- Selector de filas por página: 10, 25, 50 o 100.
- Controles de paginación: primera, anterior, número de página actual, siguiente, última.
- El índice de fila es 1-indexado y relativo al dataset completo.

---

### RF-03 · Filtrado por columna categórica

**Archivo:** `views/data_view.py` → `renderizar_vista_datos()`

- Detecta automáticamente columnas categóricas con entre 2 y 60 valores únicos.
- El usuario selecciona columna y valor; la tabla se filtra en tiempo real.
- Un banner muestra el filtro activo y el conteo de registros resultantes.
- Si no existen columnas categóricas se ocultan los controles de filtro.

---

### RF-04 · Exportación de datos

**Archivo:** `views/data_view.py` → `_preparar_descarga()`

- Botón **Exportar** disponible siempre que haya un dataset cargado.
- Genera un archivo **Excel** (`.xlsx`) si `openpyxl` está instalado.
- Como fallback genera **CSV** con encoding `utf-8-sig`.
- Exporta el dataset completo (no el filtrado).

---

### RF-05 · Diagnóstico de calidad del dataset

**Archivos:**
- Lógica: `services/dataset_service.py` → `diagnosticar_calidad()`, `DiagnosticoCalidad`
- Vista: `views/data_view.py` → `_renderizar_validacion_limpieza()`

Aparece automáticamente debajo de la tabla cada vez que hay un dataset cargado. Muestra:

| Métrica | Descripción |
|---|---|
| Filas | Total de registros |
| Columnas | Total de variables |
| Columnas numéricas | Variables con dtype numérico |
| Columnas categóricas | Variables no numéricas |
| Valores nulos | Suma de celdas vacías (resaltado en ámbar si > 0) |
| Registros duplicados | Filas completamente repetidas (resaltado en ámbar si > 0) |
| Outliers detectados (IQR) | Valores fuera de `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]` (resaltado en ámbar si > 0) |

Dos expanders opcionales muestran el **detalle por columna** de nulos y de outliers.

**Indicador visual de estado:**
- `✓ Dataset listo` — verde, cuando no hay problemas o ya se limpió.
- `⚠ Requiere limpieza` — ámbar, cuando hay nulos, duplicados u outliers sin tratar.

**API:**

```python
def diagnosticar_calidad(df: pd.DataFrame) -> DiagnosticoCalidad: ...
```

`DiagnosticoCalidad` expone las propiedades calculadas `total_nulos`, `total_outliers` y `requiere_limpieza`.

---

### RF-06 · Limpieza automática del dataset

**Archivos:**
- Lógica: `services/dataset_service.py` → `limpiar_dataset()`, `ResultadoLimpieza`
- Vista: `views/data_view.py` → botón `LIMPIAR DATASET`

El botón aparece **únicamente** cuando `requiere_limpieza = True` y aún no se ha limpiado el dataset.

La limpieza aplica **4 pasos secuenciales y automáticos** sin configuración del usuario:

| Paso | Acción |
|---|---|
| 1 | Eliminar filas completamente duplicadas |
| 2 | Imputar nulos en columnas **numéricas** con la **mediana** de la columna |
| 3 | Imputar nulos en columnas **categóricas** con la **moda** de la columna |
| 4 | **Winsorización IQR**: `clip(Q1 - 1.5·IQR, Q3 + 1.5·IQR)` en columnas numéricas |

> El DataFrame original (`dataset_original`) nunca se modifica. La limpieza opera sobre una copia.

**API:**

```python
def limpiar_dataset(df: pd.DataFrame) -> ResultadoLimpieza: ...
```

---

### RF-07 · Reporte de limpieza

**Archivo:** `views/data_view.py` → sección inferior de `_renderizar_validacion_limpieza()`

Tras ejecutar la limpieza se muestra un panel con fondo verde que incluye:

- Título: **Limpieza completada correctamente**
- Tarjetas numéricas: Duplicados eliminados, Nulos corregidos, Outliers tratados, Filas finales, Columnas finales.
- Badge final: **✓ Dataset listo para entrenamiento**

---

### RF-08 · Propagación del dataset limpio

**Archivos:** `views/data_view.py` (escritura) + `views/statistics_view.py` (lectura)

- `dataset_limpio` se guarda en `session_state` tras la limpieza.
- Las pestañas **Estadística Descriptiva** y **Entrenamiento** consumen `dataset_limpio` si existe; de lo contrario usan `dataframe_cargado`.
- Al cargar un nuevo archivo, `dataset_limpio` y `resultado_limpieza` se resetean a `None`.

---

### RF-09 · Estadística descriptiva

**Archivos:**
- Vista: `views/statistics_view.py` → `renderizar_vista_estadisticas()`
- Servicio: `services/statistics_service.py` → `ServicioEstadisticas`

> Esta vista fue diseñada específicamente para datasets con el esquema del cuestionario **Big Five** (25 preguntas Likert numeradas). Si el dataset no sigue ese esquema, la vista muestra un error descriptivo en lugar de fallar silenciosamente.

Contenido de la sección:

| Elemento | Descripción |
|---|---|
| KPIs | Registros analizados, variables, respuestas evaluadas, datos faltantes |
| Tabla de estadísticas | Media, mediana, moda, desviación estándar, mínimo y máximo por pregunta |
| Gráfica de frecuencias | Distribución global de respuestas Likert (barras, Plotly) |
| Gráfica de dimensiones | Promedio de las 5 dimensiones Big Five (barras de color, Plotly) |
| Histograma por pregunta | Distribución de la pregunta seleccionada por el usuario (Plotly) |
| Expander de detalle | Tabla de frecuencias y tabla de faltantes por pregunta |
| Botón Descargar PDF | Reporte estadístico en PDF generado con `ReportLab` |

---

## 7. Módulos en desarrollo (pendientes)

| Pestaña | Archivo(s) | Estado |
|---|---|---|
| Entrenamiento K-Means | `views/training_view.py` + `services/training_service.py` | 🔜 Stub vacío |
| Resultados | `views/results_view.py` | 🔜 Stub vacío |
| Modelos guardados | `views/models_view.py` + `services/model_service.py` | 🔜 Stub vacío |

---

## 8. API de servicios

### `dataset_service.py`

#### Clase `ServicioConjuntoDatos`

| Método | Firma | Descripción |
|---|---|---|
| `cargar_datos` | `(ruta: str) → pd.DataFrame` | Carga CSV o Excel desde ruta de archivo |
| `identificar_columnas_preguntas` | `(datos) → list[str]` | Encuentra y ordena preguntas numeradas 1–25 |
| `identificar_columna_temporal` | `(datos) → str \| None` | Localiza columna de timestamp si existe |
| `convertir_likert` | `(datos, columnas) → pd.DataFrame` | Convierte respuestas de texto a enteros 1–5 |
| `calcular_dimensiones` | `(respuestas, columnas) → pd.DataFrame` | Promedia las 5 dimensiones Big Five |
| `preprocesar` | `(datos) → ResultadoPreprocesamiento` | Orquesta la validación completa |

#### Funciones libres (RF-05 a RF-08)

| Símbolo | Tipo | Descripción |
|---|---|---|
| `diagnosticar_calidad(df)` | función | Retorna `DiagnosticoCalidad` con nulos, duplicados y outliers |
| `DiagnosticoCalidad` | dataclass | Campos: `num_filas`, `num_columnas`, `num_columnas_numericas`, `num_columnas_categoricas`, `nulos_por_columna`, `num_duplicados`, `outliers_por_columna`. Propiedades: `total_nulos`, `total_outliers`, `requiere_limpieza` |
| `limpiar_dataset(df)` | función | Retorna `ResultadoLimpieza` con el df limpio y métricas |
| `ResultadoLimpieza` | dataclass | Campos: `duplicados_eliminados`, `nulos_corregidos`, `outliers_tratados`, `filas_finales`, `columnas_finales`, `dataset_limpio` |
| `_detectar_outliers_iqr(serie)` | función interna | Cuenta outliers por IQR en una Serie numérica |

---

### `statistics_service.py`

#### Clase `ServicioEstadisticas`

| Método | Firma | Descripción |
|---|---|---|
| `calcular_resumen` | `(datos) → ResumenEstadistico` | Orquesta todos los cálculos estadísticos |

#### Dataclass `ResumenEstadistico`

| Campo | Tipo | Descripción |
|---|---|---|
| `cantidad_registros` | `int` | Total de filas válidas |
| `cantidad_variables` | `int` | Número de preguntas analizadas |
| `estadisticas_por_pregunta` | `pd.DataFrame` | Media, mediana, moda, desv. estándar, mín y máx |
| `faltantes_por_pregunta` | `pd.Series` | Conteo de nulos por columna |
| `frecuencia_respuestas` | `pd.DataFrame` | Distribución global de valores Likert |
| `promedio_dimensiones` | `pd.Series` | Promedio de cada dimensión Big Five |
| `respuestas_numericas` | `pd.DataFrame` | Respuestas convertidas a int |
| `dimensiones_por_registro` | `pd.DataFrame` | Promedio por dimensión por fila |

---

### `report_service.py`

#### Clase `ServicioReportes`

| Método | Firma | Descripción |
|---|---|---|
| `generar_reporte_estadistico` | `(resumen, nombre_dataset, fecha_generacion) → bytes` | PDF con estadísticas descriptivas |

---

## 9. Dependencias

| Paquete | Uso principal |
|---|---|
| `streamlit` | Framework web de la aplicación |
| `pandas` | Manipulación y análisis de datos |
| `numpy` | Operaciones numéricas y vectoriales |
| `scikit-learn` | Algoritmos de ML (K-Means, métricas de clustering) |
| `joblib` | Serialización de modelos entrenados |
| `plotly` | Gráficas interactivas en la UI |
| `matplotlib` | Gráficas estáticas para reportes |
| `openpyxl` | Lectura y escritura de archivos Excel |
| `fpdf2` | Generación de PDF (alternativa) |
| `reportlab` | Generación de PDF (motor principal) |
| `pytest` | Ejecución de pruebas unitarias |
