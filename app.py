"""Punto de entrada principal de ClusterLab."""

import pathlib

import streamlit as st

from views.data_view import (
    renderizar_encabezado,
    renderizar_metricas,
    renderizar_vista_datos,
)


_SECCIONES = (
    "▦  Datos cargados",
    "▥  Estadística descriptiva",
    "◈  Entrenamiento",
    "⌁  Resultados",
    "▣  Modelos guardados",
)


def _cargar_estilos() -> None:
    """Carga los estilos estructurales específicos de la aplicación."""
    ruta_css = pathlib.Path(__file__).parent / "assets" / "styles.css"
    st.html(f"<style>{ruta_css.read_text(encoding='utf-8')}</style>")


def _renderizar_navegacion() -> int:
    """Renderiza una única navegación segmentada y devuelve la sección activa."""
    indice_actual = st.session_state.get("tab_activa", 0)
    indice_actual = max(0, min(indice_actual, len(_SECCIONES) - 1))

    seleccion = st.segmented_control(
        "Secciones del flujo",
        options=list(_SECCIONES),
        default=_SECCIONES[indice_actual],
        key="main_navigation",
        label_visibility="collapsed",
    )
    seleccion = seleccion or _SECCIONES[indice_actual]
    indice_nuevo = _SECCIONES.index(seleccion)

    if indice_nuevo != indice_actual:
        st.session_state.tab_activa = indice_nuevo

    return indice_nuevo


def _renderizar_estado_vacio(titulo: str, descripcion: str, icono: str) -> None:
    """Muestra un estado de sección consistente, sin placeholders genéricos."""
    with st.container(border=True, key=f"empty-state-{icono}"):
        simbolos = {
            "bar_chart": "▥",
            "model_training": "◈",
            "query_stats": "⌁",
            "inventory_2": "▣",
        }
        st.html(f'<div class="cl-empty-icon">{simbolos.get(icono, "·")}</div>')
        st.subheader(titulo)
        st.caption(descripcion)


def main() -> None:
    """Arranca la interfaz principal de ClusterLab."""
    st.set_page_config(
        page_title="ClusterLab",
        page_icon=":material/science:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _cargar_estilos()

    # La primera vista debe explicar el producto y mostrar su estado antes de
    # pedirle al usuario que navegue por el flujo de análisis.
    renderizar_encabezado()
    renderizar_metricas()
    tab_activa = _renderizar_navegacion()

    if tab_activa == 0:
        renderizar_vista_datos()
    elif tab_activa == 1:
        _renderizar_estado_vacio(
            "Estadística descriptiva",
            "Aquí podrás revisar distribución, valores nulos, duplicados y medidas descriptivas del dataset.",
            "bar_chart",
        )
    elif tab_activa == 2:
        _renderizar_estado_vacio(
            "Entrenamiento K-Means",
            "Selecciona variables, configura el número de clústeres y prepara el modelo.",
            "model_training",
        )
    elif tab_activa == 3:
        _renderizar_estado_vacio(
            "Resultados del entrenamiento",
            "Cuando exista un modelo entrenado, aquí aparecerán sus métricas y visualizaciones.",
            "query_stats",
        )
    else:
        _renderizar_estado_vacio(
            "Modelos guardados",
            "Consulta y reutiliza modelos entrenados desde un único lugar.",
            "inventory_2",
        )


if __name__ == "__main__":
    main()
