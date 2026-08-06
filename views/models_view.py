"""Vista del catálogo persistente de modelos entrenados."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from services.dataset_service import (
    ErrorDatos,
    aplicar_mapeo_likert,
    diagnosticar_calidad,
    limpiar_dataset,
)
from services.model_service import ErrorModelo, ModeloGuardado, ServicioModelo
from services.training_service import ErrorEntrenamiento, ServicioEntrenamiento

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


def _obtener_dataset_activo() -> pd.DataFrame | None:
    """Usa el mismo criterio de prioridad que el resto de la app.

    Subconjunto filtrado -> dataset limpio -> dataset tal como se cargó.
    """
    dataframe_filtrado = st.session_state.get("dataframe_filtrado")
    if dataframe_filtrado is not None:
        return dataframe_filtrado
    dataset_limpio = st.session_state.get("dataset_limpio")
    if dataset_limpio is not None:
        return dataset_limpio
    return st.session_state.get("dataframe_cargado")


@st.cache_resource(show_spinner=False)
def _cargar_artefacto_cacheado(
    modelo_id: int, fecha_modificacion: datetime | None = None
) -> dict:
    """Evita releer el archivo joblib del modelo en cada rerun de Streamlit."""
    return ServicioModelo().cargar_modelo(modelo_id)


def _crear_tabla_modelos(modelos, datos_activos: pd.DataFrame | None) -> pd.DataFrame:
    """Prepara metadatos comprensibles, incluyendo compatibilidad con lo activo."""
    filas = []
    for modelo in modelos:
        if datos_activos is None:
            compatible = "Sin dataset activo"
        else:
            try:
                artefacto = _cargar_artefacto_cacheado(
                    modelo.id, modelo.fecha_modificacion
                )
                _, error = _preparar_datos_para_prediccion(datos_activos, artefacto)
                compatible = "Sí" if not error else "No"
            except ErrorModelo:
                compatible = "No"

        filas.append(
            {
                "Modelo": modelo.nombre,
                "Fecha de creación": modelo.fecha_creacion,
                "Fecha de modificación": modelo.fecha_modificacion,
                "Dataset de origen": modelo.dataset_origen,
                "Registros": modelo.cantidad_registros,
                "Variables": modelo.cantidad_variables,
                "Grupos": modelo.cantidad_grupos,
                "Algoritmo": modelo.algoritmo,
                "Calidad de separación": modelo.silhouette,
                "Compatible con el dataset activo": compatible,
            }
        )
    return pd.DataFrame(filas)


def _cargar_archivo_prediccion(archivo_subido) -> tuple[pd.DataFrame | None, str]:
    """Valida y carga un archivo CSV o Excel para aplicar un modelo guardado."""
    nombre = archivo_subido.name.lower()
    try:
        if nombre.endswith(".csv"):
            datos = pd.read_csv(archivo_subido)
        elif nombre.endswith((".xlsx", ".xls")):
            datos = pd.read_excel(archivo_subido, engine="openpyxl")
        else:
            return None, "Formato no compatible. Usa CSV o Excel (.xlsx / .xls)."

        if datos.empty:
            return None, "El archivo está vacío o no contiene datos válidos."
        return datos, ""
    except Exception as error:  # pragma: no cover - depende del archivo elegido
        return None, f"Error al leer el archivo: {error}"


def _tabla_centros_modelo(artefacto: dict) -> pd.DataFrame:
    """Reconstruye los centros de cada grupo en la escala original."""
    columnas = list(artefacto["columnas"])
    centros = artefacto["escalador"].inverse_transform(
        artefacto["modelo"].cluster_centers_
    )
    tabla = pd.DataFrame(centros, columns=columnas)
    tabla.insert(0, "Grupo", [f"Grupo {numero}" for numero in range(1, len(tabla) + 1)])
    return tabla.round(3)


def _preparar_datos_para_prediccion(
    datos: pd.DataFrame, artefacto: dict
) -> tuple[pd.DataFrame | None, str]:
    """Deja el nuevo dataset en el mismo formato que consumió el entrenamiento."""
    columnas_requeridas = list(artefacto.get("columnas", ()))
    faltantes = [columna for columna in columnas_requeridas if columna not in datos.columns]
    if faltantes:
        return (
            None,
            "Al nuevo dataset le faltan columnas que el modelo necesita: "
            + ", ".join(faltantes)
            + ".",
        )

    datos_trabajo = datos.copy()
    mapeo_likert = dict(artefacto.get("mapeo_likert", {}))
    for columna in artefacto.get("columnas_likert", ()):
        if columna not in datos_trabajo.columns or pd.api.types.is_numeric_dtype(
            datos_trabajo[columna]
        ):
            continue
        try:
            datos_trabajo = aplicar_mapeo_likert(datos_trabajo, [columna], mapeo_likert)
        except ErrorDatos as error:
            return None, f"No fue posible convertir la columna '{columna}': {error}"

    seleccion = datos_trabajo.loc[:, columnas_requeridas]
    no_numericas = [
        columna
        for columna in columnas_requeridas
        if not pd.api.types.is_numeric_dtype(seleccion[columna])
    ]
    if no_numericas:
        return (
            None,
            "Estas columnas deben contener valores numéricos o respuestas Likert "
            "reconocibles: " + ", ".join(no_numericas) + ".",
        )

    if seleccion.isna().any().any():
        con_nulos = seleccion.columns[seleccion.isna().any()].tolist()
        return (
            None,
            "El dataset tiene valores faltantes en: " + ", ".join(con_nulos) + ". "
            "Complétalos antes de continuar.",
        )

    return seleccion.astype(float), ""


def _predecir_clusters(datos_listos: pd.DataFrame, artefacto: dict) -> pd.Series:
    """Aplica el escalador y el modelo guardados a datos ya preparados."""
    estandarizados = pd.DataFrame(
        artefacto["escalador"].transform(datos_listos),
        columns=datos_listos.columns,
        index=datos_listos.index,
    )
    etiquetas = artefacto["modelo"].predict(estandarizados)
    return pd.Series(
        [f"Grupo {etiqueta + 1}" for etiqueta in etiquetas],
        index=datos_listos.index,
        name="Grupo asignado",
    )


def _grafica_distribucion_prediccion(asignaciones: pd.Series):
    """Muestra cuántos registros nuevos cayeron en cada grupo."""
    resumen = (
        asignaciones.value_counts()
        .rename_axis("Grupo")
        .reset_index(name="Registros")
        .sort_values("Grupo")
    )
    figura = px.bar(
        resumen,
        x="Grupo",
        y="Registros",
        color="Grupo",
        text="Registros",
        color_discrete_sequence=_COLORES_GRUPOS,
    )
    figura.update_traces(textposition="outside")
    figura.update_layout(
        showlegend=False,
        yaxis=dict(rangemode="tozero"),
        margin=dict(t=15, l=20, r=20, b=20),
    )
    return figura


def _continuar_entrenamiento_en_dataset_activo(
    modelo: ModeloGuardado, artefacto: dict, datos_listos: pd.DataFrame
) -> None:
    """Continúa desde los centros guardados sobre el dataset activo."""
    try:
        resultado_nuevo = ServicioEntrenamiento().continuar_entrenamiento(
            datos_listos,
            tuple(artefacto["columnas"]),
            artefacto["modelo"],
            artefacto["escalador"],
        )
    except ErrorEntrenamiento as error:
        st.error(
            f"No fue posible continuar entrenando con este dataset: {error}",
            icon=":material/error:",
        )
        return

    try:
        ServicioModelo().actualizar_modelo_reentrenado(modelo.id, resultado_nuevo)
        _cargar_artefacto_cacheado.clear()
    except ErrorModelo as error:
        st.error(
            f"El modelo se reentrenó, pero no fue posible guardar los cambios: {error}",
            icon=":material/error:",
        )
        return

    st.session_state["resultado_entrenamiento"] = resultado_nuevo
    st.session_state["modelo_entrenado"] = True
    st.session_state["mapeo_likert"] = dict(artefacto.get("mapeo_likert", {}))
    st.session_state["columnas_likert"] = list(artefacto.get("columnas_likert", ()))
    st.session_state["tab_activa"] = 3
    st.session_state["confirmacion_modelo_guardado"] = None

    st.session_state["_pending_navigation"] = ":material/query_stats: Resultados"
    st.session_state["_pending_toast"] = (
        (
            f'Se reentrenó "{modelo.nombre}" con {len(datos_listos)} registro(s) '
            "del dataset activo."
        ),
        ":material/model_training:",
    )
    st.rerun()


def _renderizar_continuar_entrenando(modelo: ModeloGuardado, artefacto: dict) -> None:
    """Reutiliza variables, K y mapeo Likert guardados sobre el dataset activo."""
    st.subheader("Continuar entrenando con el dataset activo")
    st.caption(
        "Reentrena este modelo con la misma configuración (variables, K y "
        "escala Likert) usando el dataset que tienes cargado actualmente, sin "
        "repetir la selección de variables ni el análisis de K."
    )

    datos_activos = _obtener_dataset_activo()
    if datos_activos is None:
        st.info(
            "No hay un dataset activo. Carga uno en la pestaña **Datos cargados** "
            "para poder reutilizar este modelo.",
            icon=":material/info:",
        )
        return

    datos_listos, error = _preparar_datos_para_prediccion(datos_activos, artefacto)
    if error:
        st.warning(
            f"El dataset activo no es compatible con este modelo: {error}",
            icon=":material/report:",
        )
        return

    st.success(
        f"El dataset activo es compatible ({len(datos_listos)} registro(s) "
        f"listos, {len(datos_listos.columns)} variable(s)).",
        icon=":material/check_circle:",
    )
    if st.button(
        f"Continuar entrenando (K = {modelo.cantidad_grupos})",
        type="primary",
        icon=":material/model_training:",
        key=f"btn_continuar_entrenando_{modelo.id}",
    ):
        _continuar_entrenamiento_en_dataset_activo(modelo, artefacto, datos_listos)


def _renderizar_aplicar_a_nuevo_dataset(modelo: ModeloGuardado, artefacto: dict) -> None:
    """Permite subir un dataset nuevo y predecir el grupo de cada registro."""
    st.subheader("Aplicar a un nuevo dataset")
    st.caption(
        "Sube un archivo con las mismas variables utilizadas al entrenar "
        f"(“{modelo.nombre}”) para asignar cada registro a un grupo existente."
    )

    archivo = st.file_uploader(
        "Selecciona un archivo CSV o Excel",
        type=["csv", "xlsx", "xls"],
        key=f"file_uploader_prediccion_{modelo.id}",
    )
    if archivo is None:
        return

    datos, error_carga = _cargar_archivo_prediccion(archivo)
    if error_carga:
        st.error(error_carga, icon=":material/error:")
        return

    diagnostico = diagnosticar_calidad(datos)
    if diagnostico.requiere_limpieza:
        resultado_limpieza = limpiar_dataset(datos)
        datos = resultado_limpieza.dataset_limpio
        st.info(
            "Se limpiaron automáticamente los datos antes de predecir: "
            f"{resultado_limpieza.duplicados_eliminados} duplicado(s) eliminado(s), "
            f"{resultado_limpieza.nulos_corregidos} valor(es) faltante(s) corregido(s).",
            icon=":material/cleaning_services:",
        )

    datos_listos, error_preparacion = _preparar_datos_para_prediccion(datos, artefacto)
    if error_preparacion:
        st.error(error_preparacion, icon=":material/error:")
        return

    asignaciones = _predecir_clusters(datos_listos, artefacto)
    resultado = datos.copy()
    resultado.insert(0, "Grupo asignado", asignaciones)

    st.success(
        f"Se asignó un grupo a {len(resultado)} registro(s) nuevo(s).",
        icon=":material/check_circle:",
    )

    columna_grafica, columna_tabla = st.columns([1, 1.6])
    with columna_grafica:
        st.plotly_chart(
            _grafica_distribucion_prediccion(asignaciones),
            width="stretch",
            key=f"grafica_prediccion_{modelo.id}",
        )
    with columna_tabla:
        st.dataframe(resultado, hide_index=True, width="stretch")

    st.download_button(
        "Descargar resultados con el grupo asignado",
        data=resultado.to_csv(index=False).encode("utf-8-sig"),
        file_name="prediccion_clusters.csv",
        mime="text/csv",
        key=f"descargar_prediccion_{modelo.id}",
        icon=":material/download:",
    )


def _renderizar_detalle_modelo(modelo: ModeloGuardado) -> None:
    """Muestra el detalle de un modelo guardado y permite reutilizarlo."""
    with st.container(border=True, key=f"detalle-modelo-{modelo.id}"):
        st.subheader(f":material/model_training: {modelo.nombre}")
        st.caption(f"Dataset de origen: {modelo.dataset_origen}")

        tarjetas = (
            ("Grupos", modelo.cantidad_grupos),
            ("Registros entrenados", modelo.cantidad_registros),
            ("Variables", modelo.cantidad_variables),
            ("Calidad de separación", f"{modelo.silhouette:.3f}"),
            ("Algoritmo", modelo.algoritmo),
        )
        columnas_metricas = st.columns(len(tarjetas), gap="medium")
        for columna, (etiqueta, valor) in zip(columnas_metricas, tarjetas):
            with columna:
                st.metric(etiqueta, valor)

        try:
            artefacto = _cargar_artefacto_cacheado(
                modelo.id, modelo.fecha_modificacion
            )
        except ErrorModelo as error:
            st.error(str(error), icon=":material/error:")
            return

        st.markdown("**Variables utilizadas al entrenar**")
        st.caption(", ".join(artefacto.get("columnas", ())) or "Sin información.")

        with st.expander("Ver centros de cada grupo", expanded=False):
            st.dataframe(
                _tabla_centros_modelo(artefacto), hide_index=True, width="stretch"
            )

        st.divider()
        _renderizar_continuar_entrenando(modelo, artefacto)
        st.divider()
        _renderizar_aplicar_a_nuevo_dataset(modelo, artefacto)


def renderizar_vista_modelos() -> None:
    """Muestra los modelos guardados y permite seleccionar uno para reutilizarlo."""
    st.title(":material/inventory_2: Modelos guardados")
    st.caption(
        "Consulta los entrenamientos almacenados, revisa su detalle y aplícalos "
        "a nuevos conjuntos de datos compatibles."
    )

    try:
        modelos = ServicioModelo().listar_modelos()
    except ErrorModelo as error:
        st.error(str(error), icon=":material/error:")
        return

    with st.container(key="models-summary"):
        st.metric(
            label=":material/inventory_2: Modelos guardados",
            value=f"{len(modelos):,}",
            border=True,
            width=360,
            help="Cantidad de entrenamientos almacenados en el catálogo.",
        )

    if not modelos:
        with st.container(border=True):
            st.subheader("Aún no hay modelos guardados")
            st.info(
                "Entrena un modelo y usa **Guardar modelo** desde la pestaña "
                "**Resultados**. Aquí aparecerá automáticamente.",
                icon=":material/info:",
            )
        return

    datos_activos = _obtener_dataset_activo()
    if datos_activos is None:
        st.caption(
            "Carga un dataset en la pestaña **Datos cargados** para que el "
            "sistema identifique qué modelos son compatibles con él."
        )
    st.caption("Selecciona una fila para ver el detalle del modelo y reutilizarlo.")
    tabla = _crear_tabla_modelos(modelos, datos_activos)
    evento = st.dataframe(
        tabla,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="tabla_modelos_guardados",
        column_config={
            "Modelo": st.column_config.TextColumn(
                "Modelo",
                pinned=True,
                help="Nombre asignado al guardar el entrenamiento.",
            ),
            "Fecha de creación": st.column_config.DatetimeColumn(
                "Fecha de creación",
                format="DD/MM/YYYY, hh:mm a",
            ),
            "Fecha de modificación": st.column_config.DatetimeColumn(
                "Fecha de modificación",
                format="DD/MM/YYYY, hh:mm a",
                help="Se actualiza automáticamente al reentrenar el modelo.",
            ),
            "Calidad de separación": st.column_config.NumberColumn(
                "Calidad de separación",
                format="%.3f",
                help=(
                    "Valor Silhouette registrado al entrenar. Mientras más cerca "
                    "de 1, más claramente separados estaban los grupos."
                ),
            ),
            "Compatible con el dataset activo": st.column_config.TextColumn(
                "Compatible con el dataset activo",
                help=(
                    "Indica si el dataset activo (pestaña Datos cargados) tiene "
                    "todas las variables que este modelo necesita."
                ),
            ),
        },
    )
    st.caption(
        "El modelo, el escalador y la preparación de variables se conservan "
        "internamente para poder utilizar este entrenamiento posteriormente."
    )

    filas_seleccionadas = evento.selection.rows if evento and evento.selection else []
    if not filas_seleccionadas:
        st.info(
            "Ningún modelo seleccionado. Haz clic sobre una fila de la tabla.",
            icon=":material/touch_app:",
        )
        return

    modelo_seleccionado = modelos[filas_seleccionadas[0]]
    _renderizar_detalle_modelo(modelo_seleccionado)

