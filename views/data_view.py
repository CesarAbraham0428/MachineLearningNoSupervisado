"""Vista para cargar, filtrar y consultar conjuntos de datos."""

import io
from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st


_SIN_FILTRO = "Sin filtro"


def _inicializar_estado() -> None:
    """Garantiza que la vista pueda arrancar sin un dataset cargado."""
    valores_iniciales = {
        "dataframe_cargado": None,
        "nombre_archivo": None,
        "fecha_carga": None,
        "pagina_actual": 1,
        "mostrar_carga": False,
        "modelo_entrenado": False,
        "modelos_guardados": [],
    }
    for clave, valor in valores_iniciales.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


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


def _detectar_columnas_categoricas(df: pd.DataFrame) -> list[str]:
    """Devuelve columnas de texto con cardinalidad razonable para filtrar."""
    candidatas = []
    for col in df.columns:
        valores_unicos = df[col].nunique()
        es_numerica = pd.api.types.is_numeric_dtype(df[col])
        if not es_numerica and 2 <= valores_unicos <= 60:
            candidatas.append(col)
    return candidatas


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
    """Renderiza el encabezado global y la acción primaria de carga."""
    _inicializar_estado()
    col_titulo, col_accion = st.columns([4.5, 1], vertical_alignment="top")

    with col_titulo:
        st.markdown(
            '<div class="cl-hero">'
            '<h1 class="cl-hero-title">Dashboard de procesamiento</h1>'
            '<p class="cl-hero-subtitle">Administra los datos, el entrenamiento y los resultados desde un solo lugar.</p>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col_accion:
        if st.button(
            "Cargar conjunto de datos",
            key="btn_abrir_carga",
            type="primary",
            width="stretch",
        ):
            st.session_state.mostrar_carga = True


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
            return

        with st.spinner("Cargando y validando archivo…"):
            df_nuevo, error = _cargar_archivo(archivo)

        if error:
            st.error(error)
            return

        st.session_state.dataframe_cargado = df_nuevo
        st.session_state.nombre_archivo = archivo.name
        st.session_state.fecha_carga = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.session_state.pagina_actual = 1
        st.session_state.mostrar_carga = False
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
            "«",
            key="pag_primera",
            disabled=pagina_actual == 1,
            help="Primera página",
            type="secondary",
            width="content",
        ):
            st.session_state.pagina_actual = 1
            st.rerun()
        if st.button(
            "‹",
            key="pag_anterior",
            disabled=pagina_actual == 1,
            help="Página anterior",
            type="secondary",
            width="content",
        ):
            st.session_state.pagina_actual = max(1, pagina_actual - 1)
            st.rerun()

        st.html(f'<span class="cl-page-current" aria-current="page">{pagina_actual}</span>')

        if st.button(
            "›",
            key="pag_siguiente",
            disabled=pagina_actual == total_paginas,
            help="Página siguiente",
            type="secondary",
            width="content",
        ):
            st.session_state.pagina_actual = min(total_paginas, pagina_actual + 1)
            st.rerun()
        if st.button(
            "»",
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
        with st.container(border=True, key="empty-data-state"):
            st.html('<div class="cl-empty-icon">↑</div>')
            st.subheader("Aún no hay datos cargados")
            st.caption("Usa el botón superior para cargar un CSV o Excel y comenzar el análisis.")
        return

    columnas_categoricas = _detectar_columnas_categoricas(df_cargado)
    columna_filtro = _SIN_FILTRO
    valor_filtro = "Todos"

    with st.container(border=True, key="dataset-panel"):
        col_titulo, col_columna, col_valor, col_exportar = st.columns(
            [2.4, 1.35, 1.35, 0.85],
            vertical_alignment="bottom",
        )
        with col_titulo:
            st.subheader("Conjunto de datos")
            st.caption("Consulta y administra la información importada.")

        with col_columna:
            if columnas_categoricas:
                columna_filtro = st.selectbox(
                    "Filtrar por columna",
                    options=[_SIN_FILTRO] + columnas_categoricas,
                    key="sel_columna_filtro",
                )
            else:
                st.caption("Sin columnas categóricas")

        with col_valor:
            if columna_filtro != _SIN_FILTRO:
                valores_unicos = ["Todos"] + sorted(
                    df_cargado[columna_filtro].dropna().unique().tolist(),
                    key=str,
                )
                valor_filtro = st.selectbox(
                    "Categoría",
                    options=valores_unicos,
                    key="sel_valor_filtro",
                )
            else:
                st.caption("Selecciona una columna")

        with col_exportar:
            datos_descarga, nombre_descarga, mime_descarga = _preparar_descarga(df_cargado)
            st.download_button(
                label="Exportar",
                data=datos_descarga,
                file_name=nombre_descarga,
                mime=mime_descarga,
                key="btn_exportar",
                width="stretch",
                help="Descarga el dataset actual en Excel o CSV según los motores disponibles",
            )

        df_filtrado = (
            df_cargado[df_cargado[columna_filtro].astype(str) == str(valor_filtro)]
            if columna_filtro != _SIN_FILTRO and valor_filtro != "Todos"
            else df_cargado
        )

        nombre_archivo = escape(str(st.session_state.nombre_archivo or "Dataset sin nombre"))
        fecha_carga = escape(str(st.session_state.fecha_carga or "Sin fecha"))
        st.html(
            '<div class="cl-dataset-meta">'
            '<div><div class="cl-meta-label">Dataset activo</div>'
            f'<div class="cl-meta-name">{nombre_archivo}</div></div>'
            '<div class="cl-meta-date"><div class="cl-meta-label">Actualizado</div>'
            f'<div>{fecha_carga}</div></div></div>'
        )

        if columna_filtro != _SIN_FILTRO and valor_filtro != "Todos":
            st.html(
                '<div class="cl-filter-note">'
                f"Filtro activo · <strong>{escape(str(columna_filtro))}</strong> = "
                f"<strong>{escape(str(valor_filtro))}</strong> · "
                f"{len(df_filtrado):,} de {len(df_cargado):,} registros</div>"
            )

        _renderizar_tabla_paginada(df_filtrado)
