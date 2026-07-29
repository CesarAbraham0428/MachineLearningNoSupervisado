"""Vista para cargar, filtrar y consultar conjuntos de datos."""

import io
from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st

from services.dataset_service import (
    DiagnosticoCalidad,
    ResultadoLimpieza,
    diagnosticar_calidad,
    limpiar_dataset,
)


_SIN_FILTRO = "Sin filtro"


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


def _renderizar_fila_calidad(etiqueta: str, valor: str, clase_color: str = "") -> str:
    """Genera una fila HTML para la tabla del resumen de calidad."""
    clase_val = f'cl-quality-value {clase_color}'.strip()
    return (
        f'<div class="cl-quality-row">'
        f'<span class="cl-quality-label">{escape(etiqueta)}</span>'
        f'<span class="{clase_val}">{escape(str(valor))}</span>'
        f'</div>'
    )


def _renderizar_validacion_limpieza(df: pd.DataFrame) -> None:
    """Renderiza el panel de calidad del dataset y el botón de limpieza (RF-05 a RF-07).

    Muestra automáticamente:
    - Resumen estructural del dataset (filas, columnas, tipos).
    - Hallazgos de calidad (nulos, duplicados, outliers por IQR).
    - Indicador visual del estado del dataset.
    - Botón "LIMPIAR DATASET" si hay problemas (RF-06).
    - Reporte de la limpieza ejecutada (RF-07).

    Args:
        df: DataFrame original cargado (nunca se modifica).
    """
    diagnostico: DiagnosticoCalidad = diagnosticar_calidad(df)
    resultado: ResultadoLimpieza | None = st.session_state.resultado_limpieza

    with st.container(border=True, key="quality-panel"):
        # ── Encabezado ──────────────────────────────────────────────────────
        col_titulo, col_estado = st.columns([3, 1], vertical_alignment="center")
        with col_titulo:
            st.subheader("Calidad del dataset")
            st.caption("Diagnóstico automático antes del entrenamiento.")
        with col_estado:
            if diagnostico.requiere_limpieza and resultado is None:
                st.html(
                    '<div class="cl-quality-badge cl-quality-badge--warn">'
                    '⚠ Requiere limpieza</div>'
                )
            else:
                st.html(
                    '<div class="cl-quality-badge cl-quality-badge--ok">'
                    '✓ Dataset listo</div>'
                )

        # ── Resumen estructural ─────────────────────────────────────────────
        filas_html = [
            _renderizar_fila_calidad("Filas", f"{diagnostico.num_filas:,}"),
            _renderizar_fila_calidad("Columnas", str(diagnostico.num_columnas)),
            _renderizar_fila_calidad("Columnas numéricas", str(diagnostico.num_columnas_numericas)),
            _renderizar_fila_calidad("Columnas categóricas", str(diagnostico.num_columnas_categoricas)),
            _renderizar_fila_calidad(
                "Valores nulos",
                str(diagnostico.total_nulos),
                "cl-quality-value--warn" if diagnostico.total_nulos > 0 else "",
            ),
            _renderizar_fila_calidad(
                "Registros duplicados",
                str(diagnostico.num_duplicados),
                "cl-quality-value--warn" if diagnostico.num_duplicados > 0 else "",
            ),
            _renderizar_fila_calidad(
                "Outliers detectados (IQR)",
                str(diagnostico.total_outliers),
                "cl-quality-value--warn" if diagnostico.total_outliers > 0 else "",
            ),
        ]
        st.html(
            '<div class="cl-quality-grid">'
            + "".join(filas_html)
            + "</div>"
        )

        # ── Detalle de outliers por columna ─────────────────────────────────
        if diagnostico.outliers_por_columna:
            with st.expander("Ver outliers por columna", expanded=False):
                for col, cantidad in diagnostico.outliers_por_columna.items():
                    st.caption(f"**{col}**: {cantidad} valor(es) fuera del rango IQR")

        # ── Detalle de nulos por columna ─────────────────────────────────────
        if diagnostico.total_nulos > 0:
            with st.expander("Ver nulos por columna", expanded=False):
                for col, cantidad in diagnostico.nulos_por_columna.items():
                    st.caption(f"**{col}**: {cantidad} valor(es) nulo(s)")

        # ── Botón de limpieza (RF-06) ────────────────────────────────────────
        if diagnostico.requiere_limpieza and resultado is None:
            st.divider()
            col_btn, col_info = st.columns([1, 3], vertical_alignment="center")
            with col_btn:
                if st.button(
                    "LIMPIAR DATASET",
                    key="btn_limpiar_dataset",
                    type="primary",
                    width="stretch",
                ):
                    with st.spinner("Limpiando dataset…"):
                        df_original = st.session_state.dataset_original
                        res = limpiar_dataset(df_original)
                    st.session_state.dataset_limpio = res.dataset_limpio
                    st.session_state.resultado_limpieza = res
                    st.toast("Dataset limpiado correctamente")
                    st.rerun()
            with col_info:
                st.caption(
                    "La limpieza es automática: elimina duplicados, imputa nulos "
                    "(mediana/moda) y aplica winsorización IQR. "
                    "El dataset original no se modifica."
                )

        # ── Reporte post-limpieza (RF-07) ────────────────────────────────────
        if resultado is not None:
            st.html(
                '<div class="cl-cleanup-report">'
                '<div class="cl-cleanup-title">Limpieza completada correctamente</div>'
                '<div class="cl-cleanup-grid">'
                f'<div class="cl-cleanup-item"><span class="cl-cleanup-num">{resultado.duplicados_eliminados}</span><span class="cl-cleanup-desc">Duplicados eliminados</span></div>'
                f'<div class="cl-cleanup-item"><span class="cl-cleanup-num">{resultado.nulos_corregidos}</span><span class="cl-cleanup-desc">Nulos corregidos</span></div>'
                f'<div class="cl-cleanup-item"><span class="cl-cleanup-num">{resultado.outliers_tratados}</span><span class="cl-cleanup-desc">Outliers tratados</span></div>'
                f'<div class="cl-cleanup-item"><span class="cl-cleanup-num">{resultado.filas_finales:,}</span><span class="cl-cleanup-desc">Filas finales</span></div>'
                f'<div class="cl-cleanup-item"><span class="cl-cleanup-num">{resultado.columnas_finales}</span><span class="cl-cleanup-desc">Columnas finales</span></div>'
                '</div>'
                '<div class="cl-cleanup-status">✓ Dataset listo para entrenamiento</div>'
                '</div>'
            )


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


    columnas_categoricas = _detectar_columnas_categoricas(df_cargado)
    columna_filtro = _SIN_FILTRO
    valor_filtro = "Todos"

    with st.container(border=True, key="dataset-panel"):
        col_titulo, col_columna, col_valor, col_exportar, col_borrar = st.columns(
            [2.0, 1.25, 1.25, 0.75, 0.75],
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

    # RF-05 / RF-06 / RF-07: panel de calidad debajo del panel de datos
    _renderizar_validacion_limpieza(df_cargado)


