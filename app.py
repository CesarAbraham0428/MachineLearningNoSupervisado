"""Punto de entrada principal de ClusterLab."""

import pathlib

import streamlit as st

from views.data_view import (
    renderizar_encabezado,
    renderizar_metricas,
    renderizar_vista_datos,
)
from views.statistics_view import renderizar_vista_estadisticas
from views.training_view import renderizar_vista_entrenamiento
from views.results_view import renderizar_vista_resultados
from views.models_view import renderizar_vista_modelos


_SECCIONES = (
    ":material/table_chart: Datos cargados",
    ":material/bar_chart: Estadística descriptiva",
    ":material/model_training: Entrenamiento",
    ":material/query_stats: Resultados",
    ":material/inventory_2: Modelos guardados",
)


_DIR_ESTILOS = pathlib.Path(__file__).parent / "assets" / "styles"


def _cargar_estilos() -> None:
    """Carga y concatena todos los archivos CSS de la carpeta de estilos.

    Esto mantiene la organización de los estilos en archivos pequeños y dedicados
    por pestaña/componente, pero asegura que toda la interfaz conserve sus estilos
    sin importar qué pestaña esté activa.
    """
    archivos_css = [
        "base.css",
        "componentes.css",
        "datos.css",
        "estadisticas.css",
        "entrenamiento.css",
        "resultados.css",
        "modelos.css",
    ]
    
    css_completo = []
    for archivo in archivos_css:
        ruta = _DIR_ESTILOS / archivo
        if ruta.exists():
            css_completo.append(ruta.read_text(encoding="utf-8"))
            
    st.html(f"<style>{''.join(css_completo)}</style>")


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
        width="stretch",
    )
    seleccion = seleccion or _SECCIONES[indice_actual]
    indice_nuevo = _SECCIONES.index(seleccion)

    if indice_nuevo != indice_actual:
        st.session_state.tab_activa = indice_nuevo

    return indice_nuevo


def _renderizar_estado_vacio(titulo: str, descripcion: str, icono: str) -> None:
    """Muestra un estado de sección consistente, sin placeholders genéricos."""
    with st.container(border=True, key=f"empty-state-{icono}"):
        st.markdown(f":material/{icono}:", text_alignment="center")
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
        renderizar_vista_estadisticas()
    elif tab_activa == 2:
        renderizar_vista_entrenamiento()
    elif tab_activa == 3:
        renderizar_vista_resultados()
    else:
        renderizar_vista_modelos()


if __name__ == "__main__":
    main()
