"""Vista independiente para analizar los resultados del entrenamiento."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.model_service import ErrorModelo, ServicioModelo
from services.report_service import ServicioReportes
from services.results_service import ErrorResultados, ServicioResultados
from services.training_service import ResultadoEntrenamiento


_COLORES_GRUPOS = [
    "#2388ff",
    "#43d78a",
    "#b06df5",
    "#f59e2f",
    "#ef5576",
    "#22b8cf",
    "#f4c95d",
    "#8b9a77",
]


def _nombre_base_dataset(nombre_archivo: object) -> str:
    """Convierte el archivo de origen en una etiqueta amigable."""
    nombre = Path(str(nombre_archivo or "Dataset")).stem.strip()
    nombre = nombre.replace("_", " ").replace("-", " ")
    return " ".join(nombre.split()) or "Dataset"


@st.dialog(
    "Guardar modelo",
    width="small",
    icon=":material/save:",
    on_dismiss="rerun",
)
def _dialogo_guardar_modelo(resultado: ResultadoEntrenamiento) -> None:
    """Solicita solo los datos comprensibles que identifican al modelo."""
    nombre_archivo = str(
        st.session_state.get("nombre_archivo") or "Dataset sin nombre"
    )
    nombre_base = _nombre_base_dataset(nombre_archivo)

    st.caption(
        "Asigna un nombre para encontrar este entrenamiento fácilmente después."
    )
    with st.container(border=True):
        st.markdown(
            f"**Resumen:** {len(resultado.asignaciones)} registros · "
            f"{len(resultado.columnas)} variables · "
            f"{resultado.k_usado} grupos · K-Means"
        )
        st.caption("La fecha de creación se registrará automáticamente.")

    with st.form("form_guardar_modelo"):
        nombre = st.text_input(
            "Nombre del modelo",
            value=f"{nombre_base} - {resultado.k_usado} grupos",
            max_chars=100,
            help="Este nombre aparecerá en la pestaña Modelos guardados.",
        )
        categoria = st.text_input(
            "Categoría o propósito",
            value=nombre_base,
            max_chars=100,
            help=(
                "Describe para qué se utilizará, por ejemplo: Personalidad, "
                "Clientes o Encuesta académica."
            ),
        )
        guardar = st.form_submit_button(
            "Guardar modelo",
            type="primary",
            icon=":material/save:",
            key="btn_guardar_modelo_form",
            width="stretch",
        )

    if not guardar:
        return

    try:
        modelo_guardado = ServicioModelo().guardar_modelo(
            resultado,
            nombre=nombre,
            categoria=categoria,
            dataset_origen=nombre_archivo,
            mapeo_likert=st.session_state.get("mapeo_likert", {}),
            columnas_likert=st.session_state.get("columnas_likert", []),
        )
    except ErrorModelo as error:
        st.error(str(error), icon=":material/error:")
        return

    st.session_state["confirmacion_modelo_guardado"] = (
        f'El modelo "{modelo_guardado.nombre}" se guardó correctamente.'
    )
    st.rerun()


def _grafica_distribucion(resumen: pd.DataFrame):
    """Muestra cuántos registros contiene cada grupo."""
    figura = px.bar(
        resumen,
        x="Grupo",
        y="Registros",
        color="Grupo",
        text="Registros",
        color_discrete_sequence=_COLORES_GRUPOS,
        labels={"Registros": "Cantidad de registros"},
    )
    figura.update_traces(textposition="outside")
    figura.update_layout(
        showlegend=False,
        yaxis=dict(rangemode="tozero"),
        margin=dict(t=15, l=20, r=20, b=20),
    )
    return figura


def _grafica_pca(proyeccion):
    """Representa registros y centros de grupo en dos dimensiones."""
    varianza_1, varianza_2 = proyeccion.varianza_explicada
    figura = px.scatter(
        proyeccion.puntos,
        x="Componente 1",
        y="Componente 2",
        color="Grupo",
        hover_name="Registro",
        color_discrete_sequence=_COLORES_GRUPOS,
        labels={
            "Componente 1": f"Componente 1 ({varianza_1:.1%})",
            "Componente 2": f"Componente 2 ({varianza_2:.1%})",
        },
    )
    figura.update_traces(marker=dict(size=9, opacity=0.78))
    figura.add_trace(
        go.Scatter(
            x=proyeccion.centros["Componente 1"],
            y=proyeccion.centros["Componente 2"],
            mode="markers",
            name="Centro de cada grupo",
            text=proyeccion.centros["Grupo"],
            hovertemplate="%{text}<br>Centro del grupo<extra></extra>",
            marker=dict(
                symbol="x",
                size=16,
                color="#ffffff",
                line=dict(width=2, color="#0f172a"),
            ),
        )
    )
    figura.update_layout(margin=dict(t=15, l=20, r=20, b=20))
    return figura


def _grafica_perfil(centros: pd.DataFrame, variable: str):
    """Compara el valor representativo de una variable entre grupos."""
    perfil = centros.loc[:, ["Grupo", variable]].rename(
        columns={variable: "Valor representativo"}
    )
    minimo = float(perfil["Valor representativo"].min())
    maximo = float(perfil["Valor representativo"].max())
    margen = max(0.5, (maximo - minimo) * 0.15)
    limite_inferior = min(0.0, minimo - margen)
    limite_superior = max(1.0, maximo + margen)

    figura = px.bar(
        perfil,
        x="Grupo",
        y="Valor representativo",
        color="Grupo",
        text="Valor representativo",
        color_discrete_sequence=_COLORES_GRUPOS,
        labels={"Valor representativo": "Valor promedio del centro"},
    )
    figura.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    figura.update_layout(
        showlegend=False,
        yaxis=dict(range=[limite_inferior, limite_superior]),
        margin=dict(t=15, l=20, r=20, b=20),
    )
    return figura


def _mostrar_lectura_silhouette(resultado: ResultadoEntrenamiento) -> None:
    """Explica la métrica principal sin exigir conocimientos de clustering."""
    titulo, explicacion = ServicioResultados.interpretar_silhouette(
        resultado.silhouette
    )
    mensaje = (
        f"**{titulo} ({resultado.silhouette:.3f}).** {explicacion} "
        "Esta lectura describe el conjunto analizado, no a una persona individual."
    )
    if resultado.silhouette < 0.25:
        st.warning(mensaje)
    else:
        st.info(mensaje)


def _mostrar_contexto_resultado(resultado: ResultadoEntrenamiento) -> None:
    """Aclara cómo se eligieron los grupos y qué significan sus nombres."""
    candidatos = [evaluacion.k for evaluacion in resultado.evaluaciones]
    if candidatos:
        rango_evaluado = (
            str(candidatos[0])
            if len(candidatos) == 1
            else f"{candidatos[0]} a {candidatos[-1]}"
        )
        seleccion = (
            f"Se evaluaron opciones de {rango_evaluado} grupos y se "
            f"seleccionaron {resultado.k_usado} por obtener el mayor Silhouette."
        )
    else:
        seleccion = f"El modelo se entrenó con {resultado.k_usado} grupos."

    with st.container(border=True):
        st.markdown("**Cómo interpretar este resultado**")
        st.markdown(
            f"- **Selección del número de grupos:** {seleccion}\n"
            "- **Nombre de los grupos:** Grupo 1, Grupo 2, etc. son solamente "
            "identificadores; no representan una calificación ni un orden de "
            "mejor a peor."
        )


def _renderizar_tarjetas_resultado(resultado: ResultadoEntrenamiento) -> None:
    """Muestra las métricas del entrenamiento como tarjetas diferenciadas."""
    tarjetas = (
        (
            "grupos",
            ":material/groups: Grupos encontrados",
            resultado.k_usado,
            "Cantidad de grupos generados por K-Means.",
        ),
        (
            "registros",
            ":material/table_rows: Registros agrupados",
            len(resultado.asignaciones),
            "Registros incluidos en la asignación del modelo.",
        ),
        (
            "variables",
            ":material/tune: Variables utilizadas",
            len(resultado.columnas),
            "Variables empleadas para entrenar el modelo.",
        ),
        (
            "calidad",
            ":material/insights: Calidad de separación",
            f"{resultado.silhouette:.3f}",
            "Silhouette: mientras más cerca de 1, más claros son los grupos.",
        ),
        (
            "distancia",
            ":material/straighten: Distancia interna",
            f"{resultado.inercia:,.1f}",
            "Inercia del modelo: un valor menor indica grupos más compactos.",
        ),
    )

    metricas = st.columns(len(tarjetas), gap="medium")
    for columna, (clave, etiqueta, valor, ayuda) in zip(metricas, tarjetas):
        with columna:
            with st.container(border=True, key=f"result-metric-card-{clave}"):
                st.metric(etiqueta, valor, help=ayuda)


def renderizar_vista_resultados() -> None:
    """Renderiza métricas, gráficas y tablas del último modelo entrenado."""
    st.title("Resultados del entrenamiento")
    st.caption(
        "Analiza los grupos generados por K-Means en una sección independiente "
        "de las estadísticas iniciales."
    )

    resultado = st.session_state.get("resultado_entrenamiento")
    if not isinstance(resultado, ResultadoEntrenamiento):
        with st.container(border=True):
            st.subheader("Aún no hay resultados disponibles")
            st.info(
                "Prepara el dataset y entrena K-Means desde la pestaña "
                "**Entrenamiento**. Al finalizar, los resultados aparecerán aquí."
            )
        return

    with st.container(horizontal=True, horizontal_alignment="right"):
        nombre_archivo = st.session_state.get("nombre_archivo") or "dataset.csv"
        reporte_pdf = ServicioReportes().generar_reporte_entrenamiento(
            resultado,
            nombre_dataset=str(nombre_archivo),
            fecha_generacion=datetime.now(),
        )
        st.download_button(
            "Descargar PDF",
            data=reporte_pdf,
            file_name="reporte_resultados_entrenamiento.pdf",
            mime="application/pdf",
            key="descargar_reporte_entrenamiento",
            icon=":material/picture_as_pdf:",
        )
        if st.button(
            "Guardar modelo",
            type="primary",
            icon=":material/save:",
            key="btn_abrir_guardar_modelo",
        ):
            _dialogo_guardar_modelo(resultado)

    confirmacion = st.session_state.pop("confirmacion_modelo_guardado", None)
    if confirmacion:
        st.success(confirmacion, icon=":material/check_circle:")

    servicio = ServicioResultados()
    try:
        resumen = servicio.crear_resumen_grupos(resultado)
        proyeccion = servicio.crear_proyeccion_pca(resultado)
        centros = servicio.crear_tabla_centros(resultado)
        interpretaciones = servicio.crear_interpretaciones_grupos(resultado)
        perfiles = servicio.crear_resumen_perfiles_grupos(resultado)
        datos_originales = st.session_state.get("dataframe_cargado")
        asignaciones = servicio.crear_tabla_asignaciones(
            resultado,
            datos_originales=(
                datos_originales
                if isinstance(datos_originales, pd.DataFrame)
                else None
            ),
        )
    except ErrorResultados as error:
        st.error(str(error))
        return

    _renderizar_tarjetas_resultado(resultado)

    _mostrar_lectura_silhouette(resultado)
    _mostrar_contexto_resultado(resultado)

    columna_distribucion, columna_pca = st.columns([1, 1.65])
    with columna_distribucion:
        with st.container(border=True):
            st.subheader("Tamaño de los grupos")
            st.caption("Cantidad de registros asignados a cada grupo.")
            st.plotly_chart(_grafica_distribucion(resumen), width="stretch")

    with columna_pca:
        with st.container(border=True):
            st.subheader("Mapa de similitud de los registros")
            st.caption(
                "Cada punto representa un registro; los puntos cercanos tienen "
                "respuestas parecidas. La X marca el centro de cada grupo."
            )
            st.plotly_chart(_grafica_pca(proyeccion), width="stretch")
            varianza_total = sum(proyeccion.varianza_explicada)
            st.caption(
                f"Esta vista 2D resume {varianza_total:.1%} de la información "
                "utilizada por el modelo. PCA solo se usa para visualizar."
            )

    st.subheader("Perfil de los grupos")
    st.caption(
        "Compara el valor representativo de una variable en el centro de cada grupo."
    )
    variable = st.selectbox(
        "Variable para comparar",
        options=list(resultado.columnas),
        key="variable_perfil_resultados",
        persist_state="session",
    )
    st.plotly_chart(_grafica_perfil(centros, variable), width="stretch")

    st.subheader("Perfil e interpretación de cada grupo")
    st.caption(
        "El perfil predominante resume los dos o tres rasgos que más distinguen a "
        "cada grupo. El detalle muestra los valores que respaldan esa lectura."
    )
    perfiles_por_grupo = perfiles.set_index("Grupo")["Perfil"]
    for grupo, lecturas_grupo in interpretaciones.groupby("Grupo", sort=False):
        with st.container(border=True):
            st.markdown(f"**{grupo} - Perfil predominante:** {perfiles_por_grupo[grupo]}")
            with st.expander("Ver cómo se obtuvo este perfil"):
                for _, fila in lecturas_grupo.iterrows():
                    st.markdown(
                        f"**{fila['Rasgo']}:** {fila['Comparación']} "
                        f"({fila['Valor del grupo']:.2f} frente a "
                        f"{fila['Promedio general']:.2f}). "
                        f"{fila['Interpretación']}"
                    )

    st.subheader("Resumen de los grupos")
    st.dataframe(
        resumen,
        hide_index=True,
        width="stretch",
        column_config={
            "Porcentaje": st.column_config.NumberColumn(
                "Porcentaje del total",
                format="percent",
                help="Proporción de registros que pertenece al grupo.",
            )
        },
    )

    with st.expander("Ver asignación de cada registro"):
        st.caption(
            "Cada fila conserva su identificador y los valores mostrados en "
            "Datos cargados, junto con el grupo asignado por K-Means."
        )
        st.dataframe(
            asignaciones,
            hide_index=True,
            width="stretch",
            column_config={
                "Identificador": st.column_config.TextColumn(
                    pinned=True,
                    help="Número de la fila correspondiente en los datos cargados.",
                ),
                "Grupo asignado": st.column_config.TextColumn(
                    pinned=True,
                    help="Grupo encontrado por el algoritmo K-Means.",
                ),
            },
        )

    with st.expander("Ver centros completos de los grupos"):
        st.caption(
            "Cada centro resume el valor representativo del grupo en todas las "
            "variables utilizadas para entrenar."
        )
        st.dataframe(centros.round(3), hide_index=True, width="stretch")