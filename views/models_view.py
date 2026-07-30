"""Vista del catálogo persistente de modelos entrenados."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.model_service import ErrorModelo, ServicioModelo


def _crear_tabla_modelos(modelos) -> pd.DataFrame:
    """Prepara únicamente metadatos comprensibles para el usuario."""
    return pd.DataFrame(
        [
            {
                "Modelo": modelo.nombre,
                "Categoría": modelo.categoria,
                "Fecha de creación": modelo.fecha_creacion,
                "Dataset de origen": modelo.dataset_origen,
                "Registros": modelo.cantidad_registros,
                "Variables": modelo.cantidad_variables,
                "Grupos": modelo.cantidad_grupos,
                "Algoritmo": modelo.algoritmo,
                "Calidad de separación": modelo.silhouette,
            }
            for modelo in modelos
        ]
    )


def renderizar_vista_modelos() -> None:
    """Muestra los modelos guardados sin exponer archivos internos."""
    st.title("Modelos guardados")
    st.caption(
        "Consulta los entrenamientos almacenados para utilizarlos nuevamente "
        "en futuras ejecuciones."
    )

    try:
        modelos = ServicioModelo().listar_modelos()
    except ErrorModelo as error:
        st.error(str(error), icon=":material/error:")
        return

    if not modelos:
        with st.container(border=True):
            st.subheader("Aún no hay modelos guardados")
            st.info(
                "Entrena un modelo y usa **Guardar modelo** desde la pestaña "
                "**Resultados**. Aquí aparecerá automáticamente.",
                icon=":material/info:",
            )
        return

    st.metric(
        "Modelos disponibles",
        len(modelos),
        border=True,
        help="Cantidad de entrenamientos almacenados en el catálogo.",
    )

    tabla = _crear_tabla_modelos(modelos)
    st.dataframe(
        tabla,
        hide_index=True,
        width="stretch",
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
            "Calidad de separación": st.column_config.NumberColumn(
                "Calidad de separación",
                format="%.3f",
                help=(
                    "Valor Silhouette registrado al entrenar. Mientras más cerca "
                    "de 1, más claramente separados estaban los grupos."
                ),
            ),
        },
    )
    st.caption(
        "El modelo, el escalador y la preparación de variables se conservan "
        "internamente para poder utilizar este entrenamiento posteriormente."
    )
