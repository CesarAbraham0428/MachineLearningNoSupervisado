"""Vista descriptiva de los cinco rasgos Big Five."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.dataset_service import ErrorDatos
from services.report_service import ServicioReportes
from services.statistics_service import ResumenEstadistico, ServicioEstadisticas


def _obtener_datos_activos() -> pd.DataFrame | None:
    """Obtiene el subconjunto filtrado y limpio vigente en Datos cargados."""
    firma_filtro = st.session_state.get("firma_filtro_activo")
    limpio = st.session_state.get("dataset_limpio")
    if (
        isinstance(limpio, pd.DataFrame)
        and st.session_state.get("filtro_calidad") == firma_filtro
    ):
        return limpio.copy()

    filtrado = st.session_state.get("dataframe_filtrado")
    if isinstance(filtrado, pd.DataFrame):
        return filtrado.copy()

    indices_filtrados = st.session_state.get("indices_filas_filtradas")
    base = limpio if isinstance(limpio, pd.DataFrame) else st.session_state.get("dataframe_cargado")
    if not isinstance(base, pd.DataFrame):
        return None
    if indices_filtrados is not None:
        indices = base.index.intersection(pd.Index(indices_filtrados))
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


def _grafica_histograma(tabla_frecuencia: pd.DataFrame, rasgo: str):
    datos = tabla_frecuencia.copy()
    limites = (
        datos["Intervalo"]
        .str.extract(
            r"^\[\s*([-+]?\d+(?:\.\d+)?)\s*-\s*([-+]?\d+(?:\.\d+)?)",
            expand=True,
        )
        .astype(float)
    )
    datos["_limite_inferior"] = limites[0]
    datos["_limite_superior"] = limites[1]
    datos["_centro_intervalo"] = (
        datos["_limite_inferior"] + datos["_limite_superior"]
    ) / 2
    datos["_ancho_intervalo"] = (
        datos["_limite_superior"] - datos["_limite_inferior"]
    )

    titulo_eje_x = f"Puntaje del rasgo {rasgo.lower()}"
    limites_eje_x = [
        *datos["_limite_inferior"].tolist(),
        float(datos["_limite_superior"].iloc[-1]),
    ]
    etiquetas_eje_x = [f"{limite:.2f}" for limite in limites_eje_x]

    figura = go.Figure(
        data=[
            go.Bar(
                x=datos["_centro_intervalo"],
                y=datos["f"],
                width=datos["_ancho_intervalo"],
                text=datos["f"],
                texttemplate="%{text}",
                textposition="outside",
                marker=dict(
                    color="#b876ff",
                    line=dict(color="#8a52c7", width=1),
                ),
                customdata=datos[
                    ["Intervalo", "Marca de Clase", "Fr", "%"]
                ].to_numpy(),
                hovertemplate=(
                    "Intervalo: %{customdata[0]}<br>"
                    "Marca de Clase: %{customdata[1]:.2f}<br>"
                    "Fr: %{customdata[2]:.4f}<br>"
                    "%: %{customdata[3]:.2f}<br>"
                    "frecuencia: %{y}<extra></extra>"
                ),
                showlegend=False,
            ),
            go.Scatter(
                x=datos["_centro_intervalo"],
                y=datos["f"],
                mode="lines+markers",
                name="Polígono de frecuencia",
                line=dict(color="#4c286e", width=3),
                marker=dict(
                    size=8,
                    color="#4c286e",
                    line=dict(color="#ffffff", width=1.5),
                ),
                hovertemplate=(
                    "Centro del intervalo: %{x:.2f}<br>"
                    "frecuencia: %{y}<extra>Polígono de frecuencia</extra>"
                ),
                showlegend=False,
            ),
        ]
    )
    return figura.update_layout(
        title=f"Distribución de {rasgo}",
        bargap=0,
        xaxis=dict(
            title=titulo_eje_x,
            tickmode="array",
            tickvals=limites_eje_x,
            ticktext=etiquetas_eje_x,
            range=[limites_eje_x[0], limites_eje_x[-1]],
        ),
        yaxis=dict(title="frecuencia", rangemode="tozero"),
        margin=dict(t=55, l=20, r=20, b=20),
    )


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
        st.caption("Resumen de los rasgos activos elegidos en Datos cargados.")
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
    st.dataframe(
        resumen.estadisticas_por_rasgo,
        width="stretch",
        column_config={
            "Varianza": st.column_config.NumberColumn(format="%.2f"),
            "Coeficiente de variación (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
        },
    )

    st.plotly_chart(_grafica_promedios(resumen), width="stretch")

    st.subheader("Distribución por rasgo")
    rasgo = st.selectbox(
        "Selecciona un rasgo",
        options=list(resumen.dimensiones_por_registro.columns),
        key="rasgo_histograma",
    )
    valores_rasgo = resumen.dimensiones_por_registro[rasgo]
    parametros = ServicioEstadisticas.calcular_parametros_intervalos(valores_rasgo)
    tabla_frecuencia = ServicioEstadisticas.calcular_frecuencia_intervalos(
        valores_rasgo,
        parametros=parametros,
    )
    st.markdown(f"#### Tabla de frecuencia por intervalos — {rasgo}")
    st.markdown(
        f"**R** = máximo - mínimo = {parametros.maximo:.2f} - {parametros.minimo:.2f} "
        f"= {parametros.rango:.2f}  \n"
        f"**K** = 1 + 1.3322 × log(N) = 1 + 1.3322 × log({parametros.cantidad_datos}) "
        f"= {parametros.k_formula:.2f} ≈ {parametros.k}  \n"
        f"**A** = R / K = {parametros.rango:.2f} / {parametros.k} "
        f"= {parametros.amplitud:.2f}"
    )
    rango, intervalos, amplitud = st.columns(3)
    rango.metric("Rango (R)", f"{parametros.rango:.2f}")
    intervalos.metric("K (intervalo)", f"{parametros.k}")
    amplitud.metric("A (amplitud)", f"{parametros.amplitud:.2f}")
    st.caption(
        "f = frecuencia absoluta · Fr = frecuencia relativa (f/N) · % = Fr × 100 · "
        "F = frecuencia absoluta acumulada."
    )
    tabla_mostrada = tabla_frecuencia.rename(columns={"Intervalo": rasgo})
    st.dataframe(
        tabla_mostrada,
        width="stretch",
        hide_index=True,
        column_config={
            rasgo: st.column_config.TextColumn(width="medium"),
            "Marca de Clase": st.column_config.NumberColumn(format="%.2f"),
            "f": st.column_config.NumberColumn(format="%d"),
            "Fr": st.column_config.NumberColumn(format="%.4f"),
            "%": st.column_config.NumberColumn(format="%.2f%%"),
            "F": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.plotly_chart(_grafica_histograma(tabla_frecuencia, rasgo), width="stretch")
