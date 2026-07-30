"""
Vista para la configuración y entrenamiento de modelos.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from services.dataset_service import (
    DiagnosticoCalidad,
    ETIQUETAS_LIKERT,
    ErrorDatos,
    ResultadoLimpieza,
    aplicar_mapeo_likert,
    detectar_columnas_likert,
    diagnosticar_calidad,
    limpiar_dataset,
    obtener_valor_likert,
)


def _obtener_datos_activos() -> pd.DataFrame | None:
    """Obtiene el dataset activo (subconjunto filtrado, dataset limpio o dataset cargado)."""
    dataframe_filtrado = st.session_state.get("dataframe_filtrado")
    if dataframe_filtrado is not None:
        return dataframe_filtrado

    dataset_limpio = st.session_state.get("dataset_limpio")
    if dataset_limpio is not None:
        return dataset_limpio

    return st.session_state.get("dataframe_cargado")


def _renderizar_fila_calidad(etiqueta: str, valor: str, clase_color: str = "") -> str:
    """Genera una fila HTML para la tabla del resumen de calidad."""
    clase_val = f"cl-quality-value {clase_color}".strip()
    return (
        f'<div class="cl-quality-row">'
        f'<span class="cl-quality-label">{escape(etiqueta)}</span>'
        f'<span class="{clase_val}">{escape(str(valor))}</span>'
        f"</div>"
    )


def renderizar_validacion_limpieza(
    df: pd.DataFrame,
    firma_filtro: tuple[tuple[str, ...], str, str] | None = None,
) -> None:
    """Renderiza el panel de calidad del dataset y el botón de limpieza (RF-05 a RF-07).

    Muestra automáticamente:
    - Resumen estructural del dataset (filas, columnas, tipos).
    - Hallazgos de calidad (nulos, duplicados, outliers por IQR).
    - Indicador visual del estado del dataset.
    - Botón "LIMPIAR DATASET" si hay problemas (RF-06).
    - Reporte de la limpieza ejecutada (RF-07).

    Args:
        df: subconjunto activo que se muestra en la tabla (nunca se modifica).
        firma_filtro: identificador de las filas y columnas que originaron el
            diagnóstico actual.
    """
    resultado_guardado: ResultadoLimpieza | None = st.session_state.get("resultado_limpieza")
    filtro_guardado = st.session_state.get("filtro_calidad")
    if firma_filtro is None:
        firma_filtro = filtro_guardado

    resultado = (
        resultado_guardado
        if resultado_guardado is not None
        and (filtro_guardado is None or filtro_guardado == firma_filtro)
        else None
    )
    df_diagnostico = resultado.dataset_limpio if resultado is not None else df
    diagnostico: DiagnosticoCalidad = diagnosticar_calidad(df_diagnostico)

    with st.container(border=True, key="quality-panel-entrenamiento"):
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
        ]
        st.html(
            '<div class="cl-quality-grid">'
            + "".join(filas_html)
            + "</div>"
        )

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
                    key="btn_limpiar_dataset_entrenamiento",
                    type="primary",
                    width="stretch",
                ):
                    with st.spinner("Limpiando dataset…"):
                        res = limpiar_dataset(df)
                    st.session_state.dataset_limpio = res.dataset_limpio
                    st.session_state.resultado_limpieza = res
                    st.session_state.filtro_calidad = firma_filtro
                    st.toast("Dataset limpiado correctamente")
                    st.rerun()
            with col_info:
                st.caption(
                    "La limpieza es automática: elimina duplicados e imputa nulos "
                    "(mediana/moda). El dataset original no se modifica."
                )

        # ── Reporte post-limpieza (RF-07) ────────────────────────────────────
        if resultado is not None:
            st.html(
                '<div class="cl-cleanup-report">'
                '<div class="cl-cleanup-title">Limpieza completada correctamente</div>'
                '<div class="cl-cleanup-grid">'
                f'<div class="cl-cleanup-item"><span class="cl-cleanup-num">{resultado.duplicados_eliminados}</span><span class="cl-cleanup-desc">Duplicados eliminados</span></div>'
                f'<div class="cl-cleanup-item"><span class="cl-cleanup-num">{resultado.nulos_corregidos}</span><span class="cl-cleanup-desc">Nulos corregidos</span></div>'
                f'<div class="cl-cleanup-item"><span class="cl-cleanup-num">{resultado.filas_finales:,}</span><span class="cl-cleanup-desc">Filas finales</span></div>'
                f'<div class="cl-cleanup-item"><span class="cl-cleanup-num">{resultado.columnas_finales}</span><span class="cl-cleanup-desc">Columnas finales</span></div>'
                '</div>'
                '<div class="cl-cleanup-status">✓ Dataset listo para entrenamiento</div>'
                '</div>'
            )


def _firma_datos_likert(
    df: pd.DataFrame,
    firma_filtro: tuple[tuple[str, ...], str, str] | None,
) -> tuple[object, tuple[str, ...], int, int]:
    """Genera una firma para invalidar una conversión cuando cambia el dataset."""
    huella = int(pd.util.hash_pandas_object(df, index=True).sum())
    return firma_filtro, tuple(map(str, df.columns)), len(df), huella


def _limpiar_estado_preparacion_entrenamiento() -> None:
    """Elimina los resultados derivados del dataset activo actual."""
    st.session_state.dataset_likert = None
    st.session_state.mapeo_likert = {}
    st.session_state.firma_likert = None
    st.session_state.columnas_likert = []
    st.session_state.dataframe_entrenamiento = None
    st.session_state.firma_entrenamiento = None


def _guardar_dataframe_entrenamiento(
    df: pd.DataFrame,
    firma_datos: tuple[object, tuple[str, ...], int, int],
) -> None:
    """Guarda una copia del dataset final que consumirá el entrenamiento."""
    st.session_state.dataframe_entrenamiento = df.copy()
    st.session_state.firma_entrenamiento = firma_datos


def _categorias_detectadas(
    columnas_likert: dict[object, tuple[str, ...]],
) -> list[str]:
    """Devuelve las respuestas únicas conservando el orden de aparición."""
    categorias: list[str] = []
    for valores in columnas_likert.values():
        for valor in valores:
            if valor not in categorias:
                categorias.append(valor)
    return categorias


def _renderizar_tabla_sugerencias_likert(
    categorias: list[str],
) -> None:
    """Muestra cómo se interpreta cada respuesta encontrada en el dataset."""
    filas = []
    for categoria in categorias:
        valor = obtener_valor_likert(categoria)
        filas.append(
            {
                "Respuesta detectada": categoria,
                "Categoría interpretada": ETIQUETAS_LIKERT[valor],
                "Valor sugerido": valor,
            }
        )
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")


def _renderizar_configuracion_likert(
    df: pd.DataFrame,
    firma_filtro: tuple[tuple[str, ...], str, str] | None,
) -> None:
    """Detecta respuestas Likert y permite convertirlas a valores numéricos."""
    firma_datos = _firma_datos_likert(df, firma_filtro)
    firma_guardada = st.session_state.get("firma_likert")
    dataset_likert = st.session_state.get("dataset_likert")

    diagnostico = diagnosticar_calidad(df)
    if diagnostico.requiere_limpieza:
        with st.container(border=True, key="likert-panel-entrenamiento-bloqueado"):
            st.subheader("Escala Likert")
            st.warning(
                "Limpia primero los valores nulos, duplicados u otros problemas "
                "de calidad para generar `dataframe_entrenamiento`."
            )
        return

    if dataset_likert is not None and firma_guardada == firma_datos:
        columnas = list(st.session_state.get("columnas_likert", []))
        if st.session_state.get("dataframe_entrenamiento") is None:
            _guardar_dataframe_entrenamiento(dataset_likert, firma_datos)
        with st.container(border=True, key="likert-panel-entrenamiento-aplicado"):
            st.subheader("Escala Likert")
            st.success(
                f"Asignación aplicada en {len(columnas)} columna(s). "
                "Los valores ya están listos para el entrenamiento."
            )
            st.caption(
                "Puedes cambiar la asignación si necesitas utilizar otra codificación."
            )
            with st.expander("Ver dataset convertido", expanded=False):
                st.dataframe(dataset_likert, hide_index=True, width="stretch")
            if st.button(
                "Cambiar asignación",
                key="btn_cambiar_mapeo_likert",
                type="secondary",
            ):
                _limpiar_estado_preparacion_entrenamiento()
                st.rerun()
        return

    columnas_likert = detectar_columnas_likert(df)
    if not columnas_likert:
        _guardar_dataframe_entrenamiento(df, firma_datos)
        with st.container(border=True, key="likert-panel-entrenamiento-vacio"):
            st.subheader("Escala Likert")
            st.info(
                "No se detectaron columnas con respuestas Likert textuales. "
                "Las columnas numéricas del 1 al 5 ya están codificadas."
            )
        return

    categorias = _categorias_detectadas(columnas_likert)
    modo_automatico = "Automático (1 a 5)"
    modo_manual = "Manual por categoría"

    with st.container(border=True, key="likert-panel-entrenamiento"):
        st.subheader("Asignación de valores Likert")
        st.caption(
            f"Se detectaron {len(columnas_likert)} columna(s) y "
            f"{len(categorias)} categoría(s) de respuesta."
        )
        _renderizar_tabla_sugerencias_likert(categorias)

        modo = st.radio(
            "¿Cómo deseas asignar los valores?",
            options=[modo_automatico, modo_manual],
            key="radio_modo_likert",
            horizontal=True,
            help=(
                "Automático usa la escala estándar: 1 totalmente en desacuerdo, "
                "2 en desacuerdo, 3 neutral, 4 de acuerdo y 5 totalmente de acuerdo."
            ),
        )

        mapeo: dict[str, int] = {}
        with st.form(key="form_mapeo_likert"):
            if modo == modo_automatico:
                mapeo = {
                    categoria: obtener_valor_likert(categoria)
                    for categoria in categorias
                }
                st.caption(
                    "Se utilizará la codificación estándar 1–5 según la categoría "
                    "detectada, incluyendo sus variaciones."
                )
            else:
                st.caption(
                    "Elige un valor distinto del 1 al 5 para cada categoría. "
                    "Esto permite adaptar respuestas con etiquetas personalizadas."
                )
                opciones = list(range(1, 6))
                for indice, categoria in enumerate(categorias):
                    sugerido = obtener_valor_likert(categoria) or 3
                    mapeo[categoria] = st.selectbox(
                        f"{categoria} → valor",
                        options=opciones,
                        index=sugerido - 1,
                        key=f"sel_mapeo_likert_{indice}_{categoria}",
                    )

            aplicar = st.form_submit_button(
                "Aplicar asignación",
                type="primary",
                width="stretch",
            )

        if not aplicar:
            return

        if len(set(mapeo.values())) != len(mapeo):
            st.error(
                "Cada categoría debe tener un valor distinto para conservar el "
                "orden de la escala Likert."
            )
            return

        try:
            dataset_convertido = aplicar_mapeo_likert(
                df,
                list(columnas_likert),
                mapeo,
            )
        except ErrorDatos as error:
            st.error(f"No se pudo aplicar la escala Likert: {error}")
            return

        st.session_state.dataset_likert = dataset_convertido
        st.session_state.mapeo_likert = mapeo
        st.session_state.firma_likert = firma_datos
        st.session_state.columnas_likert = list(columnas_likert)
        _guardar_dataframe_entrenamiento(dataset_convertido, firma_datos)
        st.toast("Escala Likert asignada correctamente")
        st.rerun()


def renderizar_vista_entrenamiento() -> None:
    """Renderiza la interfaz del entrenamiento del modelo."""
    # Garantizar que las claves de sesión existan aunque el usuario llegue
    # directamente a esta pestaña sin pasar antes por la de Datos.
    claves_requeridas = {
        "dataframe_cargado": None,
        "dataset_limpio": None,
        "dataframe_filtrado": None,
        "resultado_limpieza": None,
        "filtro_calidad": None,
        "dataset_likert": None,
        "mapeo_likert": {},
        "firma_likert": None,
        "columnas_likert": [],
        "dataframe_entrenamiento": None,
        "firma_entrenamiento": None,
    }
    for clave, valor_defecto in claves_requeridas.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor_defecto

    st.header("Entrenamiento del modelo")

    df_activo = _obtener_datos_activos()
    if df_activo is None:
        st.info(
            "Carga un conjunto de datos en la pestaña '▦  Datos cargados' para verificar "
            "su calidad y proceder con el entrenamiento."
        )
        return

    firma_filtro = st.session_state.get("filtro_calidad")
    renderizar_validacion_limpieza(df_activo, firma_filtro=firma_filtro)

    firma_datos = _firma_datos_likert(df_activo, firma_filtro)
    if (
        st.session_state.get("firma_likert") is not None
        and st.session_state.get("firma_likert") != firma_datos
    ) or (
        st.session_state.get("firma_entrenamiento") is not None
        and st.session_state.get("firma_entrenamiento") != firma_datos
    ):
        _limpiar_estado_preparacion_entrenamiento()

    _renderizar_configuracion_likert(df_activo, firma_filtro=firma_filtro)
