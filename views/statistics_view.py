"""Vista descriptiva de los cinco rasgos Big Five."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from services.dataset_service import ErrorDatos
from services.report_service import ServicioReportes
from services.statistics_service import ResumenEstadistico, ServicioEstadisticas


def _obtener_datos_activos() -> pd.DataFrame | None:
    """Obtiene las cinco dimensiones y conserva los filtros aplicados a filas."""
    filtrado = st.session_state.get("dataframe_filtrado")
    limpio = st.session_state.get("dataset_limpio")
    base = limpio if isinstance(limpio, pd.DataFrame) else st.session_state.get("dataframe_cargado")
    if not isinstance(base, pd.DataFrame):
        return None
    if isinstance(filtrado, pd.DataFrame):
        indices = base.index.intersection(filtrado.index)
        return base.loc[indices].copy()
    return base.copy()


def _grafica_promedios(resumen: ResumenEstadistico):
    datos = resumen.promedio_dimensiones.rename_axis("Rasgo").reset_index(name="Promedio")
    return px.bar(
        datos,
        x="Rasgo",
        y="Promedio",
        text="Promedio",
        color="Rasgo",
        color_discrete_sequence=["#43d78a", "#2388ff", "#b876ff", "#f39a32", "#f05d7a"],
        title="Promedio por rasgo Big Five",
        labels={"Promedio": "Promedio (1 a 5)"},
    ).update_traces(texttemplate="%{text:.2f}", textposition="outside").update_layout(
        showlegend=False,
        yaxis=dict(range=[0, 5.5], dtick=1),
        margin=dict(t=55, l=20, r=20, b=20),
    )


def _grafica_distribuciones(resumen: ResumenEstadistico):
    datos = resumen.dimensiones_por_registro.melt(
        var_name="Rasgo", value_name="Puntuación"
    )
    return px.box(
        datos,
        x="Rasgo",
        y="Puntuación",
        color="Rasgo",
        points="outliers",
        title="Dispersión de perfiles por rasgo",
        color_discrete_sequence=["#43d78a", "#2388ff", "#b876ff", "#f39a32", "#f05d7a"],
    ).update_layout(
        showlegend=False,
        yaxis=dict(range=[0.8, 5.2], dtick=1),
        margin=dict(t=55, l=20, r=20, b=20),
    )


def _grafica_histograma(resumen: ResumenEstadistico, rasgo: str):
    datos = resumen.dimensiones_por_registro[[rasgo]].rename(columns={rasgo: "Valor"})
    return px.histogram(
        datos,
        x="Valor",
        nbins=20,
        text_auto=True,
        color_discrete_sequence=["#b876ff"],
        title=f"Distribución de {rasgo}",
        labels={"Valor": "Puntuación promedio (1 a 5)", "count": "Perfiles"},
    ).update_layout(
        bargap=0.08,
        xaxis=dict(range=[1, 5], dtick=0.5),
        yaxis_rangemode="tozero",
        margin=dict(t=55, l=20, r=20, b=20),
    )


def _grafica_correlaciones(resumen: ResumenEstadistico):
    figura = px.imshow(
        resumen.correlaciones,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="Correlación entre rasgos",
        labels={"color": "Correlación"},
    )
    return figura.update_layout(margin=dict(t=55, l=20, r=20, b=20))


def _renderizar_tarjetas(resumen: ResumenEstadistico) -> None:
    valores = (
        (":material/table_rows: Perfiles analizados", resumen.cantidad_registros),
        (":material/psychology: Rasgos analizados", resumen.cantidad_variables),
        (":material/data_array: Valores evaluados", resumen.cantidad_registros * resumen.cantidad_variables),
        (":material/error_outline: Datos faltantes", int(resumen.faltantes_por_rasgo.sum())),
    )
    for columna, (etiqueta, valor) in zip(st.columns(4), valores):
        columna.metric(etiqueta, f"{valor:,}")


def renderizar_vista_estadisticas() -> None:
    """Muestra estadísticas descriptivas de la matriz activa de cinco rasgos."""
    st.header("Estadística descriptiva")
    datos = _obtener_datos_activos()
    if datos is None:
        st.info("Carga un conjunto de datos compatible para generar las estadísticas.")
        return

    try:
        resumen = ServicioEstadisticas().calcular_resumen(datos)
    except (ErrorDatos, TypeError) as error:
        st.error(f"No es posible calcular las estadísticas: {error}")
        return

    descripcion, accion_reporte = st.columns([3, 1], vertical_alignment="bottom")
    with descripcion:
        st.caption("Resumen de los cinco rasgos que utilizará K-Means.")
    with accion_reporte:
        reporte_pdf = ServicioReportes().generar_reporte_estadistico(
            resumen,
            nombre_dataset=str(st.session_state.get("nombre_archivo") or "dataset.csv"),
            fecha_generacion=datetime.now(),
        )
        st.download_button(
            "Descargar PDF",
            data=reporte_pdf,
            file_name="reporte_estadistico_inicial.pdf",
            mime="application/pdf",
            type="primary",
            key="descargar_reporte_estadistico",
            width="stretch",
            icon=":material/picture_as_pdf:",
        )

    _renderizar_tarjetas(resumen)
    st.subheader("Medidas estadísticas por rasgo")
    st.dataframe(resumen.estadisticas_por_rasgo, width="stretch")

    izquierda, derecha = st.columns(2)
    with izquierda:
        st.plotly_chart(_grafica_promedios(resumen), width="stretch")
    with derecha:
        st.plotly_chart(_grafica_distribuciones(resumen), width="stretch")

    st.subheader("Distribución por rasgo")
    rasgo = st.selectbox(
        "Selecciona un rasgo",
        options=list(resumen.dimensiones_por_registro.columns),
        key="rasgo_histograma",
    )
    st.plotly_chart(_grafica_histograma(resumen, rasgo), width="stretch")

    with st.expander("Ver correlación entre rasgos", expanded=False):
        st.plotly_chart(_grafica_correlaciones(resumen), width="stretch")
