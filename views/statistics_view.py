"""Vista única del análisis estadístico descriptivo (RF-05)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from services.dataset_service import ErrorDatos
from services.report_service import ServicioReportes
from services.statistics_service import ResumenEstadistico, ServicioEstadisticas


_ETIQUETAS_LIKERT = {
    1: "Totalmente en desacuerdo",
    2: "En desacuerdo",
    3: "Neutral",
    4: "De acuerdo",
    5: "Totalmente de acuerdo",
}


def _obtener_datos_activos() -> pd.DataFrame | None:
    """Usa el subconjunto activo; si no existe, recurre al dataset limpio o cargado."""
    dataframe_filtrado = st.session_state.get("dataframe_filtrado")
    if dataframe_filtrado is not None:
        return dataframe_filtrado

    dataset_limpio = st.session_state.get("dataset_limpio")
    if dataset_limpio is not None:
        return dataset_limpio
    return st.session_state.get("dataframe_cargado")


def _grafica_frecuencias(resumen: ResumenEstadistico):
    datos = resumen.frecuencia_respuestas
    return px.bar(
        datos,
        x="Respuesta",
        y="Frecuencia",
        text="Frecuencia",
        category_orders={"Respuesta": list(_ETIQUETAS_LIKERT.values())},
        color_discrete_sequence=["#2388ff"],
        title="Distribución general de respuestas",
        labels={"Respuesta": "Escala Likert", "Frecuencia": "Número de respuestas"},
    ).update_traces(textposition="outside").update_layout(
        showlegend=False, yaxis_rangemode="tozero", margin=dict(t=55, l=20, r=20, b=20)
    )


def _grafica_dimensiones(resumen: ResumenEstadistico):
    datos = resumen.promedio_dimensiones.rename_axis("Dimensión").reset_index(name="Promedio")
    return px.bar(
        datos,
        x="Dimensión",
        y="Promedio",
        text="Promedio",
        color="Dimensión",
        color_discrete_sequence=["#43d78a", "#2388ff", "#b876ff", "#f39a32", "#f05d7a"],
        title="Promedio por dimensión Big Five",
        labels={"Promedio": "Promedio (1 a 5)"},
    ).update_traces(texttemplate="%{text:.2f}", textposition="outside").update_layout(
        showlegend=False,
        yaxis=dict(range=[0, 5.5], dtick=1),
        margin=dict(t=55, l=20, r=20, b=20),
    )


def _grafica_histograma(resumen: ResumenEstadistico, pregunta: str):
    datos = resumen.respuestas_numericas[[pregunta]].rename(columns={pregunta: "Valor"})
    return px.histogram(
        datos,
        x="Valor",
        nbins=5,
        text_auto=True,
        color_discrete_sequence=["#b876ff"],
        title="Distribución de la pregunta seleccionada",
        labels={"Valor": "Respuesta Likert (1 a 5)", "count": "Frecuencia"},
    ).update_layout(
        bargap=0.12,
        xaxis=dict(tickmode="array", tickvals=[1, 2, 3, 4, 5]),
        yaxis_rangemode="tozero",
        margin=dict(t=55, l=20, r=20, b=20),
    )


def renderizar_vista_estadisticas() -> None:
    """Muestra todos los datos estadísticos y gráficas en una sola sección."""
    st.header("Estadística descriptiva")
    datos = _obtener_datos_activos()
    if datos is None:
        st.info("Carga un conjunto de datos compatible para generar las estadísticas.")
        return

    try:
        resumen = ServicioEstadisticas().calcular_resumen(datos)
    except ErrorDatos as error:
        st.error(f"No es posible calcular las estadísticas: {error}")
        return

    descripcion, accion_reporte = st.columns([3, 1], vertical_alignment="bottom")
    with descripcion:
        st.caption("Resumen del conjunto de datos activo antes del entrenamiento.")
    with accion_reporte:
        nombre_archivo = st.session_state.get("nombre_archivo") or "dataset.csv"
        reporte_pdf = ServicioReportes().generar_reporte_estadistico(
            resumen,
            nombre_dataset=str(nombre_archivo),
            fecha_generacion=datetime.now(),
        )
        st.download_button(
            "📄 Descargar PDF",
            data=reporte_pdf,
            file_name="reporte_estadistico_inicial.pdf",
            mime="application/pdf",
            type="primary",
            key="descargar_reporte_estadistico",
            width="stretch",
        )

    faltantes = int(resumen.faltantes_por_pregunta.sum())
    metricas = st.columns(4)
    metricas[0].metric("Registros analizados", resumen.cantidad_registros)
    metricas[1].metric("Variables analizadas", resumen.cantidad_variables)
    metricas[2].metric("Respuestas evaluadas", resumen.cantidad_registros * resumen.cantidad_variables)
    metricas[3].metric("Datos faltantes", faltantes)

    st.subheader("Medidas estadísticas por pregunta")
    st.dataframe(resumen.estadisticas_por_pregunta, width="stretch")

    izquierda, derecha = st.columns(2)
    with izquierda:
        st.plotly_chart(_grafica_frecuencias(resumen), width="stretch")
    with derecha:
        st.plotly_chart(_grafica_dimensiones(resumen), width="stretch")

    st.subheader("Distribución por pregunta")
    pregunta = st.selectbox(
        "Selecciona una pregunta para revisar su distribución",
        options=list(resumen.respuestas_numericas.columns),
        format_func=lambda nombre: str(nombre),
        key="pregunta_histograma",
    )
    st.plotly_chart(_grafica_histograma(resumen, pregunta), width="stretch")

    with st.expander("Frecuencia de respuestas y datos faltantes"):
        frecuencia, faltantes_tabla = st.columns(2)
        with frecuencia:
            st.dataframe(resumen.frecuencia_respuestas, hide_index=True, width="stretch")
        with faltantes_tabla:
            tabla_faltantes = resumen.faltantes_por_pregunta.rename("Faltantes").to_frame()
            tabla_faltantes.index.name = "Pregunta"
            st.dataframe(tabla_faltantes, width="stretch")
