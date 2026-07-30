"""Vista para cargar, filtrar y consultar conjuntos de datos."""

import io
from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st


_SIN_FILTRO = "Sin filtro"
_TODOS = "Todos"




def _inicializar_estado() -> None:
    """Garantiza que la vista pueda arrancar sin un dataset cargado."""
    valores_iniciales = {
        "dataframe_cargado": None,
        "dataset_original": None,
        "dataset_limpio": None,
        "resultado_limpieza": None,
        "nombre_archivo": None,
        "fecha_carga": None,
        "pagina_actual": 1,
        "mostrar_carga": False,
        "modelo_entrenado": False,
        "modelos_guardados": [],
        "columnas_seleccionadas": [],
        "dataframe_filtrado": None,
        "filtro_calidad": None,
        "dataframe_entrenamiento": None,
        "firma_entrenamiento": None,
        "dataset_likert": None,
        "mapeo_likert": {},
        "firma_likert": None,
        "columnas_likert": [],
    }
    for clave, valor in valores_iniciales.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


def _limpiar_estado_dataset() -> None:
    """Elimina el dataset activo del estado y vuelve a habilitar la opción de carga."""
    st.session_state.dataframe_cargado = None
    st.session_state.dataset_original = None
    st.session_state.dataset_limpio = None
    st.session_state.resultado_limpieza = None
    st.session_state.nombre_archivo = None
    st.session_state.fecha_carga = None
    st.session_state.pagina_actual = 1
    st.session_state.mostrar_carga = True
    st.session_state.columnas_seleccionadas = []
    st.session_state.dataframe_filtrado = None
    st.session_state.filtro_calidad = None
    st.session_state.dataframe_entrenamiento = None
    st.session_state.firma_entrenamiento = None
    st.session_state.dataset_likert = None
    st.session_state.mapeo_likert = {}
    st.session_state.firma_likert = None
    st.session_state.columnas_likert = []


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


def _icono_svg(nombre: str) -> str:
    """Devuelve un glifo sobrio para las métricas, sin emojis ni texto técnico."""
    glifos = {
        "people": "••",
        "columns": "▦",
        "status": "✓",
        "save": "▣",
    }
    return f'<span class="cl-icon-glyph cl-icon-glyph--{nombre}" aria-hidden="true">{glifos[nombre]}</span>'


def renderizar_encabezado() -> None:
    """Renderiza el encabezado global del dashboard."""
    _inicializar_estado()
    st.markdown(
        '<div class="cl-hero">'
        '<h1 class="cl-hero-title">Dashboard de procesamiento</h1>'
        '<p class="cl-hero-subtitle">Administra los datos, el entrenamiento y los resultados desde un solo lugar.</p>'
        "</div>",
        unsafe_allow_html=True,
    )



def renderizar_metricas() -> None:
    """Renderiza las métricas globales del dashboard, incluso en estado vacío."""
    _inicializar_estado()
    df = st.session_state.dataframe_cargado
    num_registros = len(df) if df is not None else 0
    num_variables = len(df.columns) if df is not None else 0
    modelo_entrenado = bool(st.session_state.modelo_entrenado)
    modelos_guardados = len(st.session_state.modelos_guardados)

    estado_modelo = "Entrenado" if modelo_entrenado else "No entrenado"
    estado_badge = "Listo" if modelo_entrenado else "Pendiente"
    clase_badge = "cl-badge cl-badge--green" if modelo_entrenado else "cl-badge cl-badge--amber"

    tarjetas = (
        ("blue", "people", "Registros cargados", f"{num_registros:,}", "Disponible", "green"),
        ("violet", "columns", "Variables detectadas", f"{num_variables:,}", "Columnas identificadas", "muted"),
        ("green", "status", "Estado del modelo", estado_modelo, estado_badge, "status"),
        ("amber", "save", "Modelos guardados", f"{modelos_guardados:,}", "Disponibles para consulta", "muted"),
    )

    html_tarjetas = [
        '<div class="cl-metric-grid" role="list" aria-label="Resumen del proyecto">'
    ]
    for color, icono, etiqueta, valor, detalle, tipo_detalle in tarjetas:
        if tipo_detalle == "green":
            detalle_html = '<span class="cl-badge cl-badge--green">Disponible</span>'
        elif tipo_detalle == "status":
            detalle_html = f'<span class="{clase_badge}">{escape(detalle)}</span>'
        else:
            detalle_html = f'<span class="cl-metric-detail">{escape(detalle)}</span>'

        html_tarjetas.append(
            f'<article class="cl-metric-card cl-metric-card--{color}" role="listitem">'
            f'<div class="cl-metric-icon">{_icono_svg(icono)}</div>'
            '<div class="cl-metric-copy">'
            f'<div class="cl-metric-label">{escape(etiqueta)}</div>'
            f'<div class="cl-metric-value">{escape(valor)}</div>'
            f"{detalle_html}"
            "</div></article>"
        )
    html_tarjetas.append("</div>")
    st.html("".join(html_tarjetas))


def _renderizar_carga() -> None:
    """Renderiza el uploader únicamente cuando hace falta."""
    df_cargado = st.session_state.dataframe_cargado
    mostrar = st.session_state.mostrar_carga
    if not mostrar:
        return

    with st.expander(
        "Cargar archivo de datos",
        expanded=st.session_state.mostrar_carga or df_cargado is None,
    ):
        st.caption("Selecciona un archivo CSV o Excel (.xlsx / .xls) para incorporarlo al análisis.")
        archivo = st.file_uploader(
            "Selecciona un archivo",
            type=["csv", "xlsx", "xls"],
            key="file_uploader",
        )

        if archivo is None:
            if df_cargado is not None:
                if st.button("Cancelar", key="btn_cancelar_carga", type="secondary"):
                    st.session_state.mostrar_carga = False
                    st.rerun()
            return


        with st.spinner("Cargando y validando archivo…"):
            df_nuevo, error = _cargar_archivo(archivo)

        if error:
            st.error(error)
            return

        st.session_state.dataframe_cargado = df_nuevo
        st.session_state.dataset_original = df_nuevo
        # RF-08: al cargar un nuevo archivo se invalida la limpieza anterior
        st.session_state.dataset_limpio = None
        st.session_state.resultado_limpieza = None
        st.session_state.nombre_archivo = archivo.name
        st.session_state.fecha_carga = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.session_state.pagina_actual = 1
        st.session_state.mostrar_carga = False
        st.session_state.columnas_seleccionadas = []
        st.session_state.dataframe_filtrado = None
        st.session_state.filtro_calidad = None
        st.session_state.dataframe_entrenamiento = None
        st.session_state.firma_entrenamiento = None
        st.session_state.dataset_likert = None
        st.session_state.mapeo_likert = {}
        st.session_state.firma_likert = None
        st.session_state.columnas_likert = []
        st.session_state.resultado_limpieza = None
        st.session_state.dataset_limpio = None
        st.session_state["sel_columna_filtro"] = _SIN_FILTRO
        st.session_state["sel_valor_filtro"] = _TODOS
        st.toast("Dataset cargado correctamente")
        st.rerun()


def _renderizar_tabla_paginada(df: pd.DataFrame) -> None:
    """Renderiza la tabla nativa con paginación compacta."""
    col_info, col_paginas = st.columns([3, 1], vertical_alignment="bottom")
    with col_paginas:
        filas_por_pagina = st.selectbox(
            "Filas por página",
            options=[10, 25, 50, 100],
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

    st.dataframe(
        df_pagina,
        width="stretch",
        height=min(430, (len(df_pagina) + 1) * 38 + 4),
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
        ):
            st.session_state.pagina_actual = total_paginas
            st.rerun()



def renderizar_vista_datos() -> None:
    """Renderiza la interfaz de datos (carga, filtros, exportación y tabla)."""
    _inicializar_estado()
    _renderizar_carga()

    df_cargado: pd.DataFrame | None = st.session_state.dataframe_cargado
    if df_cargado is None:
        if not st.session_state.mostrar_carga:
            with st.container(border=True, key="empty-data-state"):
                st.subheader("Aún no hay datos cargados")
                st.caption("Haz clic en el botón a continuación para seleccionar un archivo CSV o Excel.")
                if st.button(
                    "Cargar conjunto de datos",
                    key="btn_abrir_carga",
                    type="primary",
                ):
                    st.session_state.mostrar_carga = True
                    st.rerun()
        return


    todas_columnas = list(df_cargado.columns)
    columnas_categoricas = _detectar_columnas_categoricas(df_cargado)

    # Sincronizar selección con el estado de sesión; descartar columnas que ya no existen
    seleccion_previa = [
        c for c in st.session_state.columnas_seleccionadas if c in todas_columnas
    ]
    # Si no hay selección previa, usar todas las columnas como estado inicial
    if not seleccion_previa:
        seleccion_previa = todas_columnas
        st.session_state.columnas_seleccionadas = todas_columnas

    # Calcular df_filtrado ANTES del contenedor usando la selección persistida,
    # para que el botón de exportar ya tenga el dataset correcto al renderizarse.
    columnas_persistidas = st.session_state.get(
        "ms_columnas_filtro", st.session_state.columnas_seleccionadas
    )
    columnas_persistidas = [
        columna for columna in columnas_persistidas if columna in todas_columnas
    ]
    columnas_persistidas_validas = len(columnas_persistidas) >= 2
    columnas_para_exportar = (
        columnas_persistidas if columnas_persistidas_validas else todas_columnas
    )
    columna_filtro_exportacion = st.session_state.get(
        "sel_columna_filtro", _SIN_FILTRO
    )
    valor_filtro_exportacion = st.session_state.get("sel_valor_filtro", _TODOS)
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
            st.caption("Consulta y administra la información importada.")

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
            )

        with col_borrar:
            if st.button(
                "Borrar dataset",
                key="btn_borrar_dataset",
                type="secondary",
                width="stretch",
                help="Elimina el conjunto de datos activo y vuelve a mostrar el apartado de carga",
            ):
                _limpiar_estado_dataset()
                st.toast("Dataset eliminado")
                st.rerun()

        # ── Selector de columnas ──────────────────────────────────────────
        columnas_elegidas = st.multiselect(
            "Filtrar columnas a mostrar",
            options=todas_columnas,
            default=seleccion_previa,
            placeholder="Selecciona al menos 2 columnas…",
            key="ms_columnas_filtro",
            help="Elige las columnas que deseas visualizar. Se requieren mínimo 2 columnas para realizar clustering.",
        )

        col_filtro, col_valor = st.columns(2)
        with col_filtro:
            if columnas_categoricas:
                columna_filtro = st.selectbox(
                    "Filtrar filas por columna",
                    options=[_SIN_FILTRO] + columnas_categoricas,
                    key="sel_columna_filtro",
                    help="Selecciona una columna categórica, por ejemplo Comuna, para revisar solo sus registros.",
                )
            else:
                columna_filtro = _SIN_FILTRO
                st.caption("Sin columnas categóricas para filtrar filas")

        with col_valor:
            if columna_filtro == _SIN_FILTRO:
                valor_filtro = _TODOS
                st.caption("Selecciona una columna para filtrar filas")
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
                )

        # Validar mínimo 2 columnas para clustering
        columnas_validas = len(columnas_elegidas) >= 2
        if columnas_elegidas and not columnas_validas:
            st.warning(
                "⚠ Se requieren **mínimo 2 columnas** para que el algoritmo de "
                "clustering (ML no supervisado) pueda generar al menos 2 grupos. "
                "Selecciona al menos una columna adicional.",
                icon=None,
            )

        # Persistir selección en session_state para el siguiente rerun
        if columnas_elegidas:
            st.session_state.columnas_seleccionadas = columnas_elegidas
        else:
            # Si deseleccionaron todo, restauramos todas las columnas
            st.session_state.columnas_seleccionadas = todas_columnas

        columnas_a_mostrar = columnas_elegidas if columnas_validas else todas_columnas
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

        # Un resultado de limpieza solo es válido para el subconjunto que lo
        # originó. Si cambia cualquier filtro, se descarta para evitar mostrar
        # métricas o un estado de calidad pertenecientes a otra selección.
        if (
            st.session_state.get("filtro_calidad") is not None
            and st.session_state.get("filtro_calidad") != firma_filtro
        ):
            st.session_state.dataset_limpio = None
            st.session_state.resultado_limpieza = None
            st.session_state.filtro_calidad = None
            st.session_state.dataframe_entrenamiento = None
            st.session_state.firma_entrenamiento = None
            st.session_state.dataset_likert = None
            st.session_state.mapeo_likert = {}
            st.session_state.firma_likert = None
            st.session_state.columnas_likert = []
        if (
            st.session_state.get("filtro_calidad") == firma_filtro
            and st.session_state.get("resultado_limpieza") is not None
            and st.session_state.get("dataset_limpio") is not None
        ):
            df_filtrado = st.session_state.dataset_limpio.copy()
        st.session_state.dataframe_filtrado = df_filtrado.copy()

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
            notas_filtro.append(
                f"Filas: <strong>{len(df_filtrado):,}</strong> de "
                f"<strong>{len(df_cargado):,}</strong> · "
                f"<strong>{escape(str(columna_filtro))}</strong> = "
                f"<strong>{escape(str(valor_filtro))}</strong>"
            )
        if notas_filtro:
            st.html(
                '<div class="cl-filter-note">'
                + " · ".join(notas_filtro)
                + "</div>"
            )

        _renderizar_tabla_paginada(df_filtrado)
