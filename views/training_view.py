"""
Vista para la configuración y entrenamiento de modelos.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from services.dataset_service import (
    DiagnosticoCalidad,
    ResultadoLimpieza,
    diagnosticar_calidad,
    limpiar_dataset,
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
