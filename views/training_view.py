"""Vista para analizar K y entrenar modelos K-Means."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services.training_service import (
    ErrorEntrenamiento,
    EvaluacionK,
    ServicioEntrenamiento,
)


def _obtener_datos_preparados() -> pd.DataFrame | None:
    """Obtiene la copia numérica creada durante la preparación del dataset."""
    datos = st.session_state.get("dataframe_entrenamiento")
    if not isinstance(datos, pd.DataFrame):
        return None
    return datos.copy()


def _firma_datos(datos: pd.DataFrame, columnas: list[str]) -> tuple[tuple[str, ...], int]:
    """Identifica los datos seleccionados para evitar reutilizar un K obsoleto."""
    seleccion = datos.loc[:, columnas]
    huella = int(pd.util.hash_pandas_object(seleccion, index=True).sum())
    return tuple(columnas), huella


def _tabla_evaluaciones(evaluaciones: tuple[EvaluacionK, ...]) -> pd.DataFrame:
    """Da formato legible a las métricas evaluadas para cada K."""
    return pd.DataFrame(
        [
            {
                "K": evaluacion.k,
                "Inercia": round(evaluacion.inercia, 3),
                "Silhouette": round(evaluacion.silhouette, 3),
            }
            for evaluacion in evaluaciones
        ]
    )


def _grafica_codo(tabla: pd.DataFrame):
    """Construye la gráfica del método del codo a partir de la inercia."""
    return px.line(
        tabla,
        x="K",
        y="Inercia",
        markers=True,
        title="Método del codo",
        labels={"K": "Número de clústeres", "Inercia": "Inercia"},
        color_discrete_sequence=["#2388ff"],
    ).update_layout(margin=dict(t=55, l=20, r=20, b=20))


def _grafica_silhouette(tabla: pd.DataFrame):
    """Muestra la métrica usada por el sistema para recomendar K."""
    return px.bar(
        tabla,
        x="K",
        y="Silhouette",
        text="Silhouette",
        title="Separación de clústeres",
        labels={"K": "Número de clústeres", "Silhouette": "Silhouette"},
        color_discrete_sequence=["#43d78a"],
    ).update_traces(texttemplate="%{text:.3f}", textposition="outside").update_layout(
        yaxis=dict(range=[-1, 1]),
        margin=dict(t=55, l=20, r=20, b=20),
    )


def renderizar_vista_entrenamiento() -> None:
    """Guía la selección de variables, recomendación de K y entrenamiento."""
    st.header("Entrenamiento K-Means")
    st.caption(
        "Utiliza la copia numérica confirmada durante la preparación del dataset."
    )

    datos_preparados = _obtener_datos_preparados()
    if datos_preparados is None:
        st.info(
            "Aún no hay un dataset numérico preparado. Confirma la conversión "
            "de las respuestas Likert para continuar con el entrenamiento."
        )
        return

    columnas_numericas = datos_preparados.select_dtypes(include="number").columns.tolist()
    columnas_no_numericas = [
        str(columna)
        for columna in datos_preparados.columns
        if columna not in columnas_numericas
    ]
    if columnas_no_numericas:
        st.warning(
            "Estas columnas no numéricas no pueden utilizarse en K-Means: "
            + ", ".join(columnas_no_numericas)
            + "."
        )
    if len(columnas_numericas) < 2:
        st.error("Se requieren al menos dos columnas numéricas para entrenar.")
        return

    resumen_izquierda, resumen_centro, resumen_derecha = st.columns(3)
    resumen_izquierda.metric("Registros disponibles", len(datos_preparados))
    resumen_centro.metric("Variables numéricas", len(columnas_numericas))
    resumen_derecha.metric("Origen", "Datos preparados")

    columnas_seleccionadas = st.multiselect(
        "Variables que formarán los clústeres",
        options=columnas_numericas,
        default=columnas_numericas,
        key="columnas_entrenamiento",
        help="Selecciona al menos dos variables. Fechas e identificadores deben excluirse.",
    )
    if len(columnas_seleccionadas) < 2:
        st.warning("Selecciona al menos dos variables para continuar.")
        return

    datos_modelo = datos_preparados.loc[:, columnas_seleccionadas]
    firma = _firma_datos(datos_modelo, columnas_seleccionadas)
    servicio = ServicioEntrenamiento()

    if st.button("Analizar K recomendado", type="secondary", width="content"):
        try:
            evaluaciones = servicio.evaluar_k(datos_modelo)
            st.session_state["evaluaciones_k"] = evaluaciones
            st.session_state["firma_evaluaciones_k"] = firma
        except ErrorEntrenamiento as error:
            st.error(str(error))

    evaluaciones = st.session_state.get("evaluaciones_k")
    if st.session_state.get("firma_evaluaciones_k") != firma:
        evaluaciones = None

    if not evaluaciones:
        st.caption("Analiza los valores candidatos para obtener una recomendación de K.")
        return

    tabla = _tabla_evaluaciones(evaluaciones)
    k_recomendado = servicio.recomendar_k(evaluaciones)
    st.success(
        f"K recomendado: {k_recomendado}. Se eligió por obtener el mayor Silhouette."
    )

    columna_codo, columna_silhouette = st.columns(2)
    with columna_codo:
        st.plotly_chart(_grafica_codo(tabla), width="stretch")
    with columna_silhouette:
        st.plotly_chart(_grafica_silhouette(tabla), width="stretch")
    st.dataframe(tabla, hide_index=True, width="stretch")

    if st.button(
        f"Entrenar K-Means con K = {k_recomendado}",
        type="primary",
        width="content",
    ):
        progreso = st.progress(0, text="Revisando los datos preparados...")
        try:
            with st.status("Preparando entrenamiento", expanded=True) as estado:
                servicio.validar_datos(datos_modelo)
                estado.write("✓ Revisando calidad y variables numéricas")
                progreso.progress(20, text="Usando la conversión Likert confirmada...")

                estado.write("✓ Conversión Likert confirmada")
                progreso.progress(40, text="Estandarizando variables...")

                estado.write("✓ Estandarizando variables con StandardScaler")
                progreso.progress(65, text="Calculando K recomendado...")

                estado.write("✓ K recomendado mediante Silhouette")
                progreso.progress(80, text="Entrenando K-Means...")

                resultado = servicio.entrenar_modelo(
                    datos_modelo, k=k_recomendado
                )
                estado.write("✓ Entrenando K-Means")
                progreso.progress(100, text="Modelo listo")
                estado.update(label="Modelo listo", state="complete", expanded=False)

            st.session_state["resultado_entrenamiento"] = resultado
            st.session_state["modelo_entrenado"] = True
            st.success(
                f"Modelo entrenado con {resultado.k_usado} clústeres y "
                f"Silhouette de {resultado.silhouette:.3f}."
            )
            tamanos = resultado.tamanos_clusters.rename("Registros").to_frame()
            tamanos.index.name = "Clúster"
            st.subheader("Registros por clúster")
            st.dataframe(tamanos, width="stretch")
        except ErrorEntrenamiento as error:
            progreso.empty()
            st.error(str(error))
