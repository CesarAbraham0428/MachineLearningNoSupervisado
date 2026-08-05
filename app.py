"""Punto de entrada principal de ClusterLab."""

import pathlib

import streamlit as st

from views.data_view import renderizar_vista_datos
from views.statistics_view import renderizar_vista_estadisticas
from views.training_view import renderizar_vista_entrenamiento
from views.results_view import renderizar_vista_resultados
from views.models_view import renderizar_vista_modelos
from services.model_service import ErrorModelo, ServicioModelo


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
        "profesional.css",
    ]

    css_completo = []
    for archivo in archivos_css:
        ruta = _DIR_ESTILOS / archivo
        if ruta.exists():
            css_completo.append(ruta.read_text(encoding="utf-8"))

    st.html(f"<style>{''.join(css_completo)}</style>")


def _aplicar_navegacion_pendiente() -> None:
    """Traslada una navegación solicitada por otra vista antes de crear el widget.

    Streamlit no permite modificar `st.session_state[key]` de un widget después de
    que ese widget ya fue instanciado en el mismo run. Por eso, en vez de tocar
    "main_navigation" directamente desde otra vista, esa vista guarda el valor en
    "_pending_navigation" y llama a `st.rerun()`. Aquí, en el siguiente run y ANTES
    de crear el widget, copiamos ese valor pendiente a "main_navigation", lo cual
    sí está permitido.
    """
    if "_pending_navigation" in st.session_state:
        st.session_state["main_navigation"] = st.session_state.pop("_pending_navigation")


def _mostrar_toast_pendiente() -> None:
    """Muestra un toast solicitado por otra vista justo antes de un rerun."""
    if "_pending_toast" in st.session_state:
        mensaje, icono = st.session_state.pop("_pending_toast")
        st.toast(mensaje, icon=icono)


def _renderizar_navegacion() -> int:
    """Renderiza una única navegación segmentada y devuelve la sección activa."""
    try:
        modelos_guardados = ServicioModelo().contar_modelos()
    except ErrorModelo:
        modelos_guardados = 0

    secciones = _SECCIONES[:-1] + (
        f":material/inventory_2: Modelos guardados ({modelos_guardados:,})",
    )
    indice_actual = st.session_state.get("tab_activa", 0)
    indice_actual = max(0, min(indice_actual, len(secciones) - 1))

    # Si el contador de modelos cambia, la etiqueta anterior deja de ser una
    # opción válida. Se actualiza antes de crear el widget.
    if (
        "main_navigation" in st.session_state
        and st.session_state["main_navigation"] not in secciones
    ):
        st.session_state["main_navigation"] = secciones[indice_actual]

    parametros_navegacion = {
        "label": "Secciones del flujo",
        "options": list(secciones),
        "key": "main_navigation",
        "label_visibility": "collapsed",
        "width": "stretch",
    }
    # Streamlit advierte si una misma clave recibe a la vez `default` y un
    # valor en session_state. El valor inicial solo se define la primera vez.
    if "main_navigation" not in st.session_state:
        parametros_navegacion["default"] = secciones[indice_actual]

    seleccion = st.segmented_control(**parametros_navegacion)
    seleccion = seleccion or secciones[indice_actual]
    indice_nuevo = secciones.index(seleccion)

    if indice_nuevo != indice_actual:
        st.session_state.tab_activa = indice_nuevo

    return indice_nuevo


def _renderizar_encabezado() -> None:
    """Muestra una cabecera sobria que identifica la aplicacion."""
    st.markdown(
        """
        <header class="cl-app-header">
            <div>
                <p class="cl-app-kicker">ANALISIS Y AGRUPAMIENTO DE DATOS</p>
                <h1>ClusterLab</h1>
                <p class="cl-app-subtitle">
                    Espacio de trabajo para explorar datos y entrenar modelos no supervisados.
                </p>
            </div>
            <div class="cl-app-status">
                <span class="cl-app-status-dot"></span>
                Flujo de analisis
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


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

    # IMPORTANTE: esto debe ejecutarse ANTES de crear el widget "main_navigation".
    _aplicar_navegacion_pendiente()

    _cargar_estilos()

    _renderizar_encabezado()
    tab_activa = _renderizar_navegacion()

    # Puede mostrarse en cualquier punto del run; lo hacemos aquí por orden.
    _mostrar_toast_pendiente()

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