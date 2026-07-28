"""
Punto de entrada principal de la aplicación ClusterLab.
"""

import streamlit as st


def _cargar_estilos():
    """Inyecta el CSS global desde assets/styles.css."""
    import pathlib
    ruta_css = pathlib.Path(__file__).parent / "assets" / "styles.css"
    with open(ruta_css, encoding="utf-8") as archivo:
        css = archivo.read()
    st.html(f"<style>{css}</style>")


def _renderizar_tabs():
    """Navegación superior por pestañas."""
    tabs_info = [
        ("🗃️", "Datos cargados"),
        ("📊", "Estadística descriptiva"),
        ("🧠", "Entrenamiento"),
        ("📈", "Resultados"),
        ("💾", "Modelos guardados"),
    ]

    if "tab_activa" not in st.session_state:
        st.session_state.tab_activa = 0

    tab_activa = st.session_state.tab_activa

    items_html = ""
    for i, (icono, nombre) in enumerate(tabs_info):
        clase = "tab-item active" if i == tab_activa else "tab-item"
        items_html += f"<div class='{clase}'>{icono} {nombre}</div>"

    st.html(f"<div class='tab-nav'>{items_html}</div>")

    cols = st.columns(len(tabs_info))
    for i, (_, nombre) in enumerate(tabs_info):
        with cols[i]:
            if st.button(nombre, key=f"tab_btn_{i}", use_container_width=True):
                st.session_state.tab_activa = i
                st.rerun()

    return tab_activa


def main():
    """Función principal de la aplicación."""

    # Configuración de la página
    st.set_page_config(
        page_title="ClusterLab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Cargar estilos CSS
    _cargar_estilos()

    # Renderizar tabs de navegación y obtener la activa
    tab_activa = _renderizar_tabs()

    # ── Ruteo de vistas ──────────────────────────
    if tab_activa == 0:
        from views.data_view import renderizar_vista_datos
        renderizar_vista_datos()

    elif tab_activa == 1:
        st.markdown(
            "<div style='text-align:center; padding:80px; color:#64748B;'>"
            "<div style='font-size:3rem;'>📊</div>"
            "<p style='font-size:1rem; margin-top:12px;'>Estadística descriptiva — Próximamente</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    elif tab_activa == 2:
        st.markdown(
            "<div style='text-align:center; padding:80px; color:#64748B;'>"
            "<div style='font-size:3rem;'>🧠</div>"
            "<p style='font-size:1rem; margin-top:12px;'>Entrenamiento K-Means — Próximamente</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    elif tab_activa == 3:
        st.markdown(
            "<div style='text-align:center; padding:80px; color:#64748B;'>"
            "<div style='font-size:3rem;'>📈</div>"
            "<p style='font-size:1rem; margin-top:12px;'>Resultados del entrenamiento — Próximamente</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    elif tab_activa == 4:
        st.markdown(
            "<div style='text-align:center; padding:80px; color:#64748B;'>"
            "<div style='font-size:3rem;'>💾</div>"
            "<p style='font-size:1rem; margin-top:12px;'>Modelos guardados — Próximamente</p>"
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
