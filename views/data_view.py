"""Vista para cargar, filtrar y consultar conjuntos de datos."""

import io
from datetime import date, datetime
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from services.dataset_service import (
    DIMENSIONES_BIG_FIVE,
    ErrorDatos,
    crear_perfiles_big_five,
    crear_perfiles_con_contexto,
)
from services.synthetic_data_service import (
    ErrorDatosSinteticos,
    generar_dataset_sintetico,
)


_SIN_FILTRO = "Sin filtro"
_TODOS = "Todos"


def _tiene_minimo_rasgos_big_five(columnas: list[str]) -> bool:
    """Indica si la selección permite entrenar K-Means."""
    rasgos = set(DIMENSIONES_BIG_FIVE)
    return sum(columna in rasgos for columna in columnas) >= 2


def _validar_seleccion_columnas(todas_columnas: list[str]) -> None:
    """Conserva la última selección válida de al menos dos rasgos Big Five."""
    seleccion = [
        columna
        for columna in st.session_state.get("ms_columnas_filtro", [])
        if columna in todas_columnas
    ]
    if _tiene_minimo_rasgos_big_five(seleccion):
        st.session_state.columnas_seleccionadas = seleccion
        return

    anterior = [
        columna
        for columna in st.session_state.get("columnas_seleccionadas", [])
        if columna in todas_columnas
    ]
    seleccion_segura = anterior if _tiene_minimo_rasgos_big_five(anterior) else todas_columnas
    st.session_state["ms_columnas_filtro"] = seleccion_segura
    st.session_state["aviso_minimo_rasgos"] = True

def _etiqueta_columna(columna: object) -> str:
    """Devuelve un nombre visible sin modificar el encabezado original."""
    nombre = str(columna).strip()
    if nombre.casefold() in {"marca temporal", "timestamp"}:
        return "Fecha y hora de respuesta"
    return nombre


def _convertir_fechas(serie: pd.Series) -> pd.Series:
    """Convierte fechas comunes, incluidas marcas de Google Forms en español."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return pd.to_datetime(serie, errors="coerce")

    texto = serie.astype("string").str.strip()
    texto = texto.str.replace(r"(?i)a\s*\.\s*m\s*\.", "AM", regex=True)
    texto = texto.str.replace(r"(?i)p\s*\.\s*m\s*\.", "PM", regex=True)
    texto = texto.str.replace(
        r"(?i)\s+GMT\s*[+-]\s*\d{1,2}(?::?\d{2})?\s*$",
        "",
        regex=True,
    )
    return pd.to_datetime(texto, errors="coerce", format="mixed")


def _detectar_columnas_temporales(df: pd.DataFrame) -> list[str]:
    """Detecta columnas cuyos valores representan fechas y horas."""
    temporales = []
    for columna in df.columns:
        serie = df[columna]
        if pd.api.types.is_numeric_dtype(serie):
            continue
        valores_presentes = int(serie.notna().sum())
        if valores_presentes == 0:
            continue
        fechas_validas = int(_convertir_fechas(serie).notna().sum())
        if fechas_validas / valores_presentes >= 0.8:
            temporales.append(columna)
    return temporales


def _limites_rango_fechas(
    valor: object,
    fecha_minima: date,
    fecha_maxima: date,
) -> tuple[date, date]:
    """Normaliza selecciones completas o parciales del selector de fechas."""
    if isinstance(valor, (tuple, list)) and valor:
        fechas = [pd.Timestamp(item).date() for item in valor[:2]]
        if len(fechas) == 1:
            fechas.append(fechas[0])
        inicio, fin = sorted(fechas)
        return inicio, fin
    return fecha_minima, fecha_maxima


def _asegurar_rango_fechas(serie: pd.Series) -> tuple[date, date]:
    """Inicializa o corrige el rango persistido para el dataset actual."""
    fechas = _convertir_fechas(serie).dropna()
    fecha_minima = fechas.min().date()
    fecha_maxima = fechas.max().date()
    rango_guardado = st.session_state.get("rango_fechas_filtro")

    try:
        inicio, fin = _limites_rango_fechas(
            rango_guardado,
            fecha_minima,
            fecha_maxima,
        )
    except (TypeError, ValueError):
        inicio, fin = fecha_minima, fecha_maxima

    if inicio < fecha_minima or fin > fecha_maxima:
        st.session_state["rango_fechas_filtro"] = (fecha_minima, fecha_maxima)
    elif rango_guardado is None:
        st.session_state["rango_fechas_filtro"] = (fecha_minima, fecha_maxima)
    return fecha_minima, fecha_maxima




def _inicializar_estado() -> None:
    """Garantiza que la vista pueda arrancar sin un dataset cargado."""
    valores_iniciales = {
        "dataframe_cargado": None,
        "dataset_fuente_original": None,
        "dataset_original": None,
        "dataset_original_visualizacion": None,
        "nombre_archivo_original": None,
        "fecha_carga_original": None,
        "dataset_sintetico_activo": False,
        "dataset_limpio": None,
        "resultado_limpieza": None,
        "nombre_archivo": None,
        "fecha_carga": None,
        "pagina_actual": 1,
        "mostrar_carga": False,
        "modelo_entrenado": False,
        "resultado_entrenamiento": None,
        "modelos_guardados": [],
        "columnas_seleccionadas": [],
        "dataframe_filtrado": None,
        "indices_filas_filtradas": None,
        "firma_filtro_activo": None,
        "filtro_calidad": None,
        "dataframe_entrenamiento": None,
        "firma_entrenamiento": None,
        "dataset_likert": None,
        "mapeo_likert": {},
        "firma_likert": None,
        "columnas_likert": [],
        "rango_fechas_filtro": None,
        "datos_sinteticos_generados": None,
        "datos_combinados_sinteticos": None,
    }
    for clave, valor in valores_iniciales.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


def _limpiar_estado_dataset() -> None:
    """Elimina el dataset activo del estado y vuelve a habilitar la opción de carga."""
    st.session_state.dataframe_cargado = None
    st.session_state.dataset_fuente_original = None
    st.session_state.dataset_original = None
    st.session_state.dataset_original_visualizacion = None
    st.session_state.nombre_archivo_original = None
    st.session_state.fecha_carga_original = None
    st.session_state.dataset_sintetico_activo = False
    st.session_state.dataset_limpio = None
    st.session_state.resultado_limpieza = None
    st.session_state.nombre_archivo = None
    st.session_state.fecha_carga = None
    st.session_state.pagina_actual = 1
    st.session_state.mostrar_carga = True
    st.session_state.modelo_entrenado = False
    st.session_state.resultado_entrenamiento = None
    st.session_state.pop("variable_perfil_resultados", None)
    st.session_state.columnas_seleccionadas = []
    st.session_state.dataframe_filtrado = None
    st.session_state.indices_filas_filtradas = None
    st.session_state.firma_filtro_activo = None
    st.session_state.filtro_calidad = None
    st.session_state.dataframe_entrenamiento = None
    st.session_state.firma_entrenamiento = None
    st.session_state.dataset_likert = None
    st.session_state.mapeo_likert = {}
    st.session_state.firma_likert = None
    st.session_state.columnas_likert = []
    st.session_state.rango_fechas_filtro = None
    _limpiar_resultado_sintetico()


def _limpiar_resultado_sintetico() -> None:
    """Descarta el resultado y restaura el original si estaba activo el combinado."""
    if st.session_state.get("dataset_sintetico_activo"):
        dataset_original = st.session_state.get("dataset_original_visualizacion")
        if not isinstance(dataset_original, pd.DataFrame):
            dataset_original = st.session_state.get("dataset_original")
        if isinstance(dataset_original, pd.DataFrame):
            st.session_state.dataframe_cargado = dataset_original.copy(deep=True)
        st.session_state.nombre_archivo = st.session_state.get(
            "nombre_archivo_original"
        )
        st.session_state.fecha_carga = st.session_state.get("fecha_carga_original")
        st.session_state.dataset_sintetico_activo = False
        _invalidar_resultados_dataset()

    st.session_state.datos_sinteticos_generados = None
    st.session_state.datos_combinados_sinteticos = None


def _invalidar_resultados_dataset() -> None:
    """Limpia filtros y cálculos derivados al cambiar el dataset activo."""
    st.session_state.dataset_limpio = None
    st.session_state.resultado_limpieza = None
    st.session_state.pagina_actual = 1
    st.session_state.columnas_seleccionadas = []
    st.session_state.dataframe_filtrado = None
    st.session_state.indices_filas_filtradas = None
    st.session_state.firma_filtro_activo = None
    st.session_state.filtro_calidad = None
    st.session_state.dataframe_entrenamiento = None
    st.session_state.firma_entrenamiento = None
    st.session_state.dataset_likert = None
    st.session_state.mapeo_likert = {}
    st.session_state.firma_likert = None
    st.session_state.columnas_likert = []
    st.session_state.modelo_entrenado = False
    st.session_state.resultado_entrenamiento = None
    st.session_state.pop("evaluaciones_k", None)
    st.session_state.pop("firma_evaluaciones_k", None)
    st.session_state.rango_fechas_filtro = None
    st.session_state.pop("variable_perfil_resultados", None)


def _invalidar_resultados_filtro() -> None:
    """Descarta resultados calculados con una selección de filas anterior."""
    st.session_state.dataset_limpio = None
    st.session_state.resultado_limpieza = None
    st.session_state.filtro_calidad = None
    st.session_state.dataframe_entrenamiento = None
    st.session_state.firma_entrenamiento = None
    st.session_state.dataset_likert = None
    st.session_state.mapeo_likert = {}
    st.session_state.firma_likert = None
    st.session_state.columnas_likert = []
    st.session_state.modelo_entrenado = False
    st.session_state.resultado_entrenamiento = None
    st.session_state.pop("evaluaciones_k", None)
    st.session_state.pop("firma_evaluaciones_k", None)
    st.session_state.pop("variable_perfil_resultados", None)


def _activar_datos_combinados() -> None:
    """Convierte el resultado sintético en el dataset que consumen las vistas."""
    datos_combinados = st.session_state.get("datos_combinados_sinteticos")
    if not isinstance(datos_combinados, pd.DataFrame):
        return

    st.session_state.dataframe_cargado = datos_combinados.copy(deep=True)
    st.session_state.dataset_sintetico_activo = True
    st.session_state.mostrar_carga = False
    _invalidar_resultados_dataset()
    st.session_state.nombre_archivo = _nombre_descarga_sinteticos(
        st.session_state.get("nombre_archivo_original"),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.session_state.fecha_carga = datetime.now().strftime("%d/%m/%Y %H:%M")


def _restaurar_datos_originales() -> None:
    """Vuelve a usar el original y conserva el combinado para reutilizarlo."""
    if not st.session_state.get("dataset_sintetico_activo"):
        return

    datos_originales = st.session_state.get("dataset_original_visualizacion")
    if not isinstance(datos_originales, pd.DataFrame):
        datos_originales = st.session_state.get("dataset_original")
    if not isinstance(datos_originales, pd.DataFrame):
        return

    st.session_state.dataframe_cargado = datos_originales.copy(deep=True)
    st.session_state.dataset_sintetico_activo = False
    st.session_state.nombre_archivo = st.session_state.get(
        "nombre_archivo_original"
    )
    st.session_state.fecha_carga = st.session_state.get("fecha_carga_original")
    _invalidar_resultados_dataset()


def _detectar_columnas_categoricas(df: pd.DataFrame) -> list[str]:
    """Devuelve columnas categóricas con cardinalidad útil para filtrar filas."""
    candidatas = []
    for columna in df.columns:
        valores_unicos = df[columna].nunique(dropna=True)
        es_numerica = pd.api.types.is_numeric_dtype(df[columna])
        if not es_numerica and 2 <= valores_unicos <= 60:
            candidatas.append(columna)
    return candidatas


def _crear_firma_filtro(
    columnas: list[str], columna_filtro: str, valor_filtro: object
) -> tuple[tuple[str, ...], str, str]:
    """Identifica el subconjunto que se está mostrando y diagnosticando."""
    return tuple(map(str, columnas)), str(columna_filtro), str(valor_filtro)


def _obtener_subconjunto(
    datos: pd.DataFrame,
    columnas: list[str],
    columna_filtro: str = _SIN_FILTRO,
    valor_filtro: object = _TODOS,
) -> pd.DataFrame:
    """Aplica los filtros de filas y columnas al dataset sin modificarlo."""
    columnas_validas = [columna for columna in columnas if columna in datos.columns]
    resultado = datos
    if (
        columna_filtro in datos.columns
        and columna_filtro != _SIN_FILTRO
        and valor_filtro != _TODOS
    ):
        if isinstance(valor_filtro, (tuple, list)) and valor_filtro:
            fechas = _convertir_fechas(datos[columna_filtro])
            fechas_validas = fechas.dropna()
            if not fechas_validas.empty:
                inicio, fin = _limites_rango_fechas(
                    valor_filtro,
                    fechas_validas.min().date(),
                    fechas_validas.max().date(),
                )
                dias = fechas.dt.date
                resultado = datos.loc[dias.between(inicio, fin)]
        else:
            resultado = datos.loc[
                datos[columna_filtro].astype(str).eq(str(valor_filtro))
            ]
    return resultado.loc[:, columnas_validas].copy()



def _cargar_archivo(archivo_subido) -> tuple[pd.DataFrame | None, str]:
    """Valida y carga un archivo CSV o Excel."""
    nombre = archivo_subido.name.lower()
    try:
        if nombre.endswith(".csv"):
            df = pd.read_csv(archivo_subido)
        elif nombre.endswith((".xlsx", ".xls")):
            df = pd.read_excel(archivo_subido, engine="openpyxl")
        else:
            return None, "Formato no compatible. Usa CSV o Excel (.xlsx / .xls)."

        if df.empty:
            return None, "El archivo está vacío o no contiene datos válidos."

        return df, ""
    except Exception as error:  # pragma: no cover - depende del archivo elegido
        return None, f"Error al leer el archivo: {error}"



def _exportar_excel(df: pd.DataFrame) -> bytes:
    """Serializa el DataFrame en un buffer Excel."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as escritor:
        df.to_excel(escritor, index=False, sheet_name="Datos")
    return buffer.getvalue()


def _preparar_descarga(df: pd.DataFrame) -> tuple[bytes, str, str]:
    """Prepara Excel o un CSV compatible si falta el motor opcional."""
    try:
        import openpyxl  # noqa: F401 - comprobación explícita del motor.
    except ImportError:
        return (
            df.to_csv(index=False).encode("utf-8-sig"),
            "datos_exportados.csv",
            "text/csv",
        )

    return (
        _exportar_excel(df),
        "datos_exportados.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _renderizar_carga() -> None:
    """Renderiza la zona de carga cuando no hay dataset o se solicita reemplazo."""
    df_cargado = st.session_state.dataframe_cargado
    # Sin dataset, el uploader debe estar disponible desde el primer render.
    # Cuando ya existe uno, `mostrar_carga` conserva el flujo para reemplazarlo.
    mostrar = df_cargado is None or st.session_state.mostrar_carga
    if not mostrar:
        return

    with st.container(border=True, key="upload-panel"):
        st.markdown(":material/cloud_upload:", text_alignment="center")
        st.subheader("Cargar archivo de datos", text_alignment="center")
        st.caption(
            "Selecciona un CSV o Excel para comenzar tu análisis de clustering.",
            text_alignment="center",
        )
        archivo = st.file_uploader(
            ":material/upload_file: Selecciona un archivo",
            type=["csv", "xlsx", "xls"],
            key="file_uploader",
            help="Formatos admitidos: CSV, XLSX y XLS. Tamaño máximo: 200 MB.",
            width="stretch",
        )

        if archivo is None:
            if df_cargado is not None:
                if st.button(
                    "Cancelar",
                    key="btn_cancelar_carga",
                    type="secondary",
                    icon=":material/close:",
                ):
                    st.session_state.mostrar_carga = False
                    st.rerun()
            return


        with st.spinner("Cargando y validando archivo…"):
            df_nuevo, error = _cargar_archivo(archivo)

        if error:
            st.error(error)
            return

        try:
            perfiles_big_five = crear_perfiles_big_five(df_nuevo)
            perfiles_con_contexto = crear_perfiles_con_contexto(df_nuevo)
        except (ErrorDatos, TypeError) as error:
            st.error(
                "El archivo no pudo transformarse a perfiles Big Five: " + str(error)
            )
            return

        st.session_state.dataset_fuente_original = df_nuevo.copy(deep=True)
        st.session_state.dataframe_cargado = perfiles_con_contexto.copy(deep=True)
        st.session_state.dataset_original = perfiles_big_five.copy(deep=True)
        st.session_state.dataset_original_visualizacion = (
            perfiles_con_contexto.copy(deep=True)
        )
        st.session_state.dataset_sintetico_activo = False
        # RF-08: al cargar un nuevo archivo se invalida la limpieza anterior
        st.session_state.dataset_limpio = None
        st.session_state.resultado_limpieza = None
        st.session_state.nombre_archivo = archivo.name
        st.session_state.nombre_archivo_original = archivo.name
        st.session_state.fecha_carga = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.session_state.fecha_carga_original = st.session_state.fecha_carga
        st.session_state.pagina_actual = 1
        st.session_state.mostrar_carga = False
        st.session_state.modelo_entrenado = False
        st.session_state.resultado_entrenamiento = None
        st.session_state.pop("variable_perfil_resultados", None)
        st.session_state.columnas_seleccionadas = []
        st.session_state.dataframe_filtrado = None
        st.session_state.indices_filas_filtradas = None
        st.session_state.firma_filtro_activo = None
        st.session_state.filtro_calidad = None
        st.session_state.dataframe_entrenamiento = None
        st.session_state.firma_entrenamiento = None
        st.session_state.dataset_likert = None
        st.session_state.mapeo_likert = {}
        st.session_state.firma_likert = None
        st.session_state.columnas_likert = []
        st.session_state.resultado_limpieza = None
        st.session_state.dataset_limpio = None
        st.session_state.rango_fechas_filtro = None
        _limpiar_resultado_sintetico()
        st.session_state["sel_columna_filtro"] = _SIN_FILTRO
        st.session_state["sel_valor_filtro"] = _TODOS
        st.toast("Dataset convertido correctamente a cinco rasgos Big Five")
        st.rerun()


def _renderizar_tabla_paginada(df: pd.DataFrame) -> None:
    """Renderiza la tabla nativa con paginación compacta."""
    col_info, col_paginas = st.columns([3, 1], vertical_alignment="bottom")
    with col_paginas:
        opciones_filas = [100, 500, 1000, 2000]
        if st.session_state.get("filas_pagina") not in opciones_filas:
            st.session_state["filas_pagina"] = opciones_filas[0]
        filas_por_pagina = st.selectbox(
            "Filas por página",
            options=opciones_filas,
            index=0,
            key="filas_pagina",
        )

    total_registros = len(df)
    total_paginas = max(1, -(-total_registros // filas_por_pagina))
    pagina_actual = min(st.session_state.pagina_actual, total_paginas)
    st.session_state.pagina_actual = pagina_actual

    inicio = (pagina_actual - 1) * filas_por_pagina
    fin = min(inicio + filas_por_pagina, total_registros)
    df_pagina = df.iloc[inicio:fin].reset_index(drop=True)
    df_pagina.index = df_pagina.index + inicio + 1
    df_pagina.index.name = "#"

    with col_info:
        st.caption(f"Mostrando {inicio + 1} a {fin} de {total_registros:,} registros")

    configuracion_columnas = {
        columna: st.column_config.TextColumn(_etiqueta_columna(columna))
        for columna in df_pagina.columns
        if _etiqueta_columna(columna) != str(columna)
    }
    st.dataframe(
        df_pagina,
        width="stretch",
        height=min(430, (len(df_pagina) + 1) * 38 + 4),
        column_config=configuracion_columnas,
    )
    _renderizar_paginacion(pagina_actual, total_paginas)


def _renderizar_paginacion(pagina_actual: int, total_paginas: int) -> None:
    """Renderiza los controles de paginación con estados accesibles."""
    with st.container(horizontal=True, horizontal_alignment="center", gap="small"):
        if st.button(
            "Primera",
            key="pag_primera",
            disabled=pagina_actual == 1,
            help="Primera página",
            type="secondary",
            width="content",
            icon=":material/first_page:",
        ):
            st.session_state.pagina_actual = 1
            st.rerun()
        if st.button(
            ":material/chevron_left:",
            key="pag_anterior",
            disabled=pagina_actual == 1,
            help="Página anterior",
            type="secondary",
            width="content",
        ):
            st.session_state.pagina_actual = max(1, pagina_actual - 1)
            st.rerun()

        # Renderizar todas las páginas como botones
        for pagina_num in range(1, total_paginas + 1):
            es_pagina_actual = (pagina_num == pagina_actual)
            if st.button(
                str(pagina_num),
                key=f"pag_num_{pagina_num}",
                type="primary" if es_pagina_actual else "secondary",
                width="content",
            ):
                st.session_state.pagina_actual = pagina_num
                st.rerun()

        if st.button(
            ":material/chevron_right:",
            key="pag_siguiente",
            disabled=pagina_actual == total_paginas,
            help="Página siguiente",
            type="secondary",
            width="content",
        ):
            st.session_state.pagina_actual = min(total_paginas, pagina_actual + 1)
            st.rerun()
        if st.button(
            "Última",
            key="pag_ultima",
            disabled=pagina_actual == total_paginas,
            help="Última página",
            type="secondary",
            width="content",
            icon=":material/last_page:",
        ):
            st.session_state.pagina_actual = total_paginas
            st.rerun()


def _nombre_descarga_sinteticos(nombre_archivo: object, mime: str) -> str:
    """Construye un nombre de descarga claro sin depender del formato de entrada."""
    nombre = Path(str(nombre_archivo or "dataset")).stem or "dataset"
    extension = ".csv" if mime == "text/csv" else ".xlsx"
    return f"{nombre}_sinteticos_y_original{extension}"


def _renderizar_generador_sinteticos(datos_originales: pd.DataFrame) -> None:
    """Muestra la generación normal de perfiles Big Five y su activación."""
    with st.container(border=True, key="synthetic-data-panel"):
        st.subheader("Generar perfiles Big Five sintéticos")
        st.caption(
            "Para cada rasgo activo se calcula la media y desviación estándar de los "
            "perfiles originales y se generan valores normales entre 1 y 5."
        )

        columna_cantidad, columna_accion = st.columns(
            [2, 1], vertical_alignment="bottom"
        )
        with columna_cantidad:
            cantidad = st.number_input(
                "Nuevos registros a generar",
                min_value=1,
                max_value=10_000,
                value=100,
                step=1,
                key="cantidad_registros_sinteticos",
                help="El resultado final contendrá esta cantidad más los registros originales.",
            )
        with columna_accion:
            generar = st.button(
                "Generar registros",
                key="btn_generar_datos_sinteticos",
                type="primary",
                width="stretch",
                icon=":material/auto_awesome:",
            )

        if generar:
            try:
                with st.spinner("Generando perfiles sintéticos…"):
                    resultado = generar_dataset_sintetico(
                        datos_originales,
                        cantidad=int(cantidad),
                    )
            except (ErrorDatosSinteticos, TypeError, ValueError) as error:
                st.error(
                    "No se pudieron generar los registros: " + str(error),
                    icon=":material/error:",
                )
            else:
                st.session_state.datos_sinteticos_generados = (
                    resultado.datos_sinteticos
                )
                st.session_state.datos_combinados_sinteticos = (
                    resultado.datos_combinados
                )
                st.toast("Datos sintéticos generados correctamente")
                st.rerun()

        datos_sinteticos = st.session_state.get("datos_sinteticos_generados")
        datos_combinados = st.session_state.get("datos_combinados_sinteticos")
        if not isinstance(datos_sinteticos, pd.DataFrame) or not isinstance(
            datos_combinados, pd.DataFrame
        ):
            return

        total_originales = len(datos_originales)
        total_sinteticos = len(datos_sinteticos)
        metrica_originales, metrica_sinteticos, metrica_final = st.columns(3)
        metrica_originales.metric("Originales", f"{total_originales:,}")
        metrica_sinteticos.metric("Sintéticos", f"{total_sinteticos:,}")
        metrica_final.metric("Archivo final", f"{len(datos_combinados):,}")

        st.caption(
            "Vista previa: cada fila contiene solo los rasgos activos. Las primeras "
            "filas son sintéticas y las restantes corresponden a perfiles originales."
        )
        st.dataframe(datos_combinados.head(8), width="stretch", hide_index=True)

        with st.expander("Ver parámetros estimados desde los originales"):
            originales_generados = datos_combinados.iloc[total_sinteticos:]
            parametros = pd.DataFrame(
                {
                    "Media": originales_generados.mean(),
                    "Desviación estándar": originales_generados.std(ddof=1),
                }
            ).round(3)
            parametros.index.name = "Rasgo"
            st.dataframe(parametros, width="stretch")

        (
            columna_descarga,
            columna_activar,
            columna_restaurar,
            columna_borrado,
        ) = st.columns(4)
        datos_descarga, _, mime_descarga = _preparar_descarga(datos_combinados)
        with columna_descarga:
            st.download_button(
                "Exportar sintéticos + original",
                data=datos_descarga,
                file_name=_nombre_descarga_sinteticos(
                    st.session_state.get("nombre_archivo_original")
                    or st.session_state.get("nombre_archivo"),
                    mime_descarga,
                ),
                mime=mime_descarga,
                key="btn_exportar_datos_sinteticos",
                width="stretch",
                icon=":material/download:",
            )
        with columna_activar:
            datos_activos = st.session_state.get("dataset_sintetico_activo", False)
            if st.button(
                "Set de datos activo"
                if datos_activos
                else "Usar como set de datos",
                key="btn_usar_datos_sinteticos",
                type="primary" if not datos_activos else "secondary",
                disabled=datos_activos,
                width="stretch",
                help=(
                    "Usa el resultado combinado para la tabla, los filtros y "
                    "la estadística descriptiva."
                ),
                icon=":material/dataset:",
            ):
                _activar_datos_combinados()
                st.toast("Set de datos combinado activado")
                st.rerun()
        with columna_restaurar:
            datos_activos = st.session_state.get("dataset_sintetico_activo", False)
            if st.button(
                "Restaurar original",
                key="btn_restaurar_datos_originales",
                type="secondary",
                disabled=not datos_activos,
                width="stretch",
                help=(
                    "Vuelve a usar el dataset original en la tabla, los filtros "
                    "y la estadística descriptiva."
                ),
                icon=":material/restore:",
            ):
                _restaurar_datos_originales()
                st.toast("Dataset original restaurado")
                st.rerun()
        with columna_borrado:
            if st.button(
                "Borrar resultado sintético",
                key="btn_borrar_datos_sinteticos",
                type="secondary",
                width="stretch",
                icon=":material/delete:",
            ):
                _limpiar_resultado_sintetico()
                st.toast("Resultado sintético eliminado")
                st.rerun()


def renderizar_vista_datos() -> None:
    """Renderiza la interfaz de datos (carga, filtros, exportación y tabla)."""
    _inicializar_estado()
    _renderizar_carga()

    df_cargado: pd.DataFrame | None = st.session_state.dataframe_cargado
    if df_cargado is None:
        return


    todas_columnas = list(df_cargado.columns)
    columnas_categoricas = _detectar_columnas_categoricas(df_cargado)
    columnas_temporales = _detectar_columnas_temporales(df_cargado)
    columnas_filtrables = list(
        dict.fromkeys(columnas_temporales + columnas_categoricas)
    )

    # Sincronizar selección con el estado de sesión; descartar columnas que ya no existen
    seleccion_previa = [
        c for c in st.session_state.columnas_seleccionadas if c in todas_columnas
    ]
    # El análisis requiere al menos dos rasgos Big Five. Si una sesión antigua
    # conserva una selección menor, se recupera una selección segura antes de
    # que se renderice el widget.
    if not _tiene_minimo_rasgos_big_five(seleccion_previa):
        seleccion_previa = todas_columnas
        st.session_state.columnas_seleccionadas = todas_columnas

    seleccion_widget = st.session_state.get("ms_columnas_filtro")
    seleccion_widget_valida = isinstance(seleccion_widget, list) and _tiene_minimo_rasgos_big_five(
        [columna for columna in seleccion_widget if columna in todas_columnas]
    )
    if not seleccion_widget_valida:
        st.session_state["ms_columnas_filtro"] = seleccion_previa
    # Calcular df_filtrado ANTES del contenedor usando la selección persistida,
    # para que el botón de exportar ya tenga el dataset correcto al renderizarse.
    columnas_persistidas = st.session_state.get(
        "ms_columnas_filtro", st.session_state.columnas_seleccionadas
    )
    columnas_persistidas = [
        columna for columna in columnas_persistidas if columna in todas_columnas
    ]
    columnas_persistidas_validas = _tiene_minimo_rasgos_big_five(
        columnas_persistidas
    )
    columnas_para_exportar = (
        columnas_persistidas if columnas_persistidas_validas else seleccion_previa
    )
    columna_filtro_exportacion = st.session_state.get(
        "sel_columna_filtro", _SIN_FILTRO
    )
    if columna_filtro_exportacion in columnas_temporales:
        _asegurar_rango_fechas(df_cargado[columna_filtro_exportacion])
        valor_filtro_exportacion = st.session_state.get(
            "rango_fechas_filtro", _TODOS
        )
    else:
        valor_filtro_exportacion = st.session_state.get(
            "sel_valor_filtro", _TODOS
        )
    df_para_exportar = _obtener_subconjunto(
        df_cargado,
        columnas_para_exportar,
        columna_filtro_exportacion,
        valor_filtro_exportacion,
    )

    with st.container(border=True, key="dataset-panel"):
        col_titulo, col_exportar, col_borrar = st.columns(
            [4.5, 0.75, 0.75],
            vertical_alignment="bottom",
        )
        with col_titulo:
            st.subheader("Conjunto de datos")
            st.caption(
                "Cada registro representa una persona mediante sus cinco rasgos "
                "Big Five. Las columnas de contexto se conservan únicamente para "
                "filtrar filas y nunca se incorporan a K-Means."
            )

        with col_exportar:
            datos_descarga, nombre_descarga, mime_descarga = _preparar_descarga(df_para_exportar)
            columnas_exportadas = len(columnas_para_exportar)
            total_columnas = len(todas_columnas)
            etiqueta_ayuda = (
                f"Exporta las {columnas_exportadas} columnas visibles de {total_columnas} totales"
                if columnas_exportadas < total_columnas
                else "Exporta el dataset completo en Excel o CSV"
            )
            st.download_button(
                label="Exportar",
                data=datos_descarga,
                file_name=nombre_descarga,
                mime=mime_descarga,
                key="btn_exportar",
                width="stretch",
                help=etiqueta_ayuda,
                icon=":material/download:",
            )

        with col_borrar:
            if st.button(
                "Borrar dataset",
                key="btn_borrar_dataset",
                type="secondary",
                width="stretch",
                help=(
                    "Elimina de memoria el original, el combinado y cualquier "
                    "resultado sintético; después vuelve a mostrar la carga."
                ),
                icon=":material/delete:",
            ):
                _limpiar_estado_dataset()
                st.toast("Dataset eliminado")
                st.rerun()

        # ── Selector de columnas ──────────────────────────────────────────
        columnas_elegidas = st.multiselect(
            "Columnas visibles y activas (mínimo 2 rasgos Big Five)",
            options=todas_columnas,
            format_func=_etiqueta_columna,
            placeholder="Selecciona al menos 2 columnas…",
            key="ms_columnas_filtro",
            on_change=_validar_seleccion_columnas,
            args=(todas_columnas,),
            help=(
                "Debes conservar al menos dos rasgos Big Five. Las columnas elegidas "
                "se aplican a la tabla, exportación, estadísticas y entrenamiento; "
                "los filtros de filas también se respetan en todas esas vistas."
            ),
        )

        col_filtro, col_valor = st.columns(2)
        with col_filtro:
            if columnas_filtrables:
                columna_filtro = st.selectbox(
                    "Filtrar filas por columna",
                    options=[_SIN_FILTRO] + columnas_filtrables,
                    format_func=_etiqueta_columna,
                    key="sel_columna_filtro",
                    persist_state="session",
                    help="Selecciona una columna para revisar solo los registros que coincidan.",
                )
            else:
                columna_filtro = _SIN_FILTRO
                st.caption("Sin columnas categóricas para filtrar filas")

        with col_valor:
            if columna_filtro == _SIN_FILTRO:
                valor_filtro = _TODOS
                st.caption("Selecciona una columna para filtrar filas")
            elif columna_filtro in columnas_temporales:
                fecha_minima, fecha_maxima = _asegurar_rango_fechas(
                    df_cargado[columna_filtro]
                )
                valor_filtro = st.date_input(
                    "Rango de fechas",
                    min_value=fecha_minima,
                    max_value=fecha_maxima,
                    key="rango_fechas_filtro",
                    format="DD/MM/YYYY",
                    persist_state="session",
                    help="Selecciona la fecha inicial y final. Se incluyen ambos días.",
                )
            else:
                valores_unicos = [_TODOS] + sorted(
                    {
                        str(valor)
                        for valor in df_cargado[columna_filtro].dropna().unique()
                    },
                    key=str,
                )
                if st.session_state.get("sel_valor_filtro") not in valores_unicos:
                    st.session_state["sel_valor_filtro"] = _TODOS
                valor_filtro = st.selectbox(
                    "Valor",
                    options=valores_unicos,
                    key="sel_valor_filtro",
                    persist_state="session",
                )

        columnas_validas = _tiene_minimo_rasgos_big_five(columnas_elegidas)
        if st.session_state.pop("aviso_minimo_rasgos", False):
            st.warning(
                "Selecciona al menos dos rasgos Big Five. Se conservó la última "
                "selección válida para que el análisis no use una sola variable.",
                icon=":material/info:",
            )

        # Solo una selección con al menos dos rasgos Big Five se vuelve activa.
        if columnas_validas:
            st.session_state.columnas_seleccionadas = columnas_elegidas
        else:
            columnas_elegidas = seleccion_previa

        columnas_a_mostrar = columnas_elegidas
        df_filtrado = _obtener_subconjunto(
            df_cargado,
            columnas_a_mostrar,
            columna_filtro,
            valor_filtro,
        )
        firma_filtro = _crear_firma_filtro(
            columnas_a_mostrar,
            columna_filtro,
            valor_filtro,
        )

        # La firma del filtro se guarda siempre, aunque el dataset no haya
        # requerido limpieza. Así todas las vistas comparten la misma selección.
        firma_anterior = st.session_state.get("firma_filtro_activo")
        if firma_anterior is not None and firma_anterior != firma_filtro:
            _invalidar_resultados_filtro()
        st.session_state.firma_filtro_activo = firma_filtro
        if (
            st.session_state.get("filtro_calidad") == firma_filtro
            and st.session_state.get("resultado_limpieza") is not None
            and st.session_state.get("dataset_limpio") is not None
        ):
            df_filtrado = st.session_state.dataset_limpio.copy()
        st.session_state.dataframe_filtrado = df_filtrado.copy()
        st.session_state.indices_filas_filtradas = df_filtrado.index.tolist()

        nombre_archivo = escape(str(st.session_state.nombre_archivo or "Dataset sin nombre"))
        fecha_carga = escape(str(st.session_state.fecha_carga or "Sin fecha"))
        st.html(
            '<div class="cl-dataset-meta">'
            '<div><div class="cl-meta-label">Dataset activo</div>'
            f'<div class="cl-meta-name">{nombre_archivo}</div></div>'
            '<div class="cl-meta-date"><div class="cl-meta-label">Actualizado</div>'
            f'<div>{fecha_carga}</div></div></div>'
        )

        notas_filtro = []
        if columnas_validas and len(columnas_elegidas) < len(todas_columnas):
            excluidas = len(todas_columnas) - len(columnas_elegidas)
            notas_filtro.append(
                f"Columnas visibles: <strong>{len(columnas_elegidas)}</strong> de "
                f"<strong>{len(todas_columnas)}</strong> · "
                f"{excluidas} columna(s) ocultada(s)"
            )
        if columna_filtro != _SIN_FILTRO and valor_filtro != _TODOS:
            etiqueta_filtro = _etiqueta_columna(columna_filtro)
            if columna_filtro in columnas_temporales:
                fechas_validas = _convertir_fechas(
                    df_cargado[columna_filtro]
                ).dropna()
                inicio, fin = _limites_rango_fechas(
                    valor_filtro,
                    fechas_validas.min().date(),
                    fechas_validas.max().date(),
                )
                valor_visible = (
                    inicio.strftime("%d/%m/%Y")
                    if inicio == fin
                    else f"{inicio:%d/%m/%Y} al {fin:%d/%m/%Y}"
                )
            else:
                valor_visible = str(valor_filtro)
            notas_filtro.append(
                f"Filas: <strong>{len(df_filtrado):,}</strong> de "
                f"<strong>{len(df_cargado):,}</strong> · "
                f"<strong>{escape(etiqueta_filtro)}</strong>: "
                f"<strong>{escape(valor_visible)}</strong>"
            )
        if notas_filtro:
            st.html(
                '<div class="cl-filter-note">'
                + " · ".join(notas_filtro)
                + "</div>"
            )

        _renderizar_tabla_paginada(df_filtrado)

    st.divider()
    _renderizar_generador_sinteticos(df_filtrado)
