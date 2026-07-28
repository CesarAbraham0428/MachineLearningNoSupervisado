"""
Vista para la gestión y carga de datos.
RF-01: Cargar dataset (CSV / Excel)
RF-02: Visualizar el dataset en tabla paginada
RF-03: Filtrar por categoría
RF-04: Exportar datos filtrados a Excel
"""

import io
import streamlit as st
import pandas as pd


# ─────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────

def _cargar_archivo(archivo_subido) -> tuple[pd.DataFrame | None, str]:
    """Valida y carga el archivo en un DataFrame.

    Returns:
        (dataframe, mensaje_error)  — mensaje_error vacío si todo va bien.
    """
    nombre = archivo_subido.name.lower()
    try:
        if nombre.endswith(".csv"):
            df = pd.read_csv(archivo_subido)
        elif nombre.endswith((".xlsx", ".xls")):
            df = pd.read_excel(archivo_subido, engine="openpyxl")
        else:
            return None, "Formato no compatible. Usa CSV o Excel (.xlsx / .xls)."

        if df.empty:
            return None, "El archivo está vacío o no contiene datos válidos."

        return df, ""
    except Exception as error:
        return None, f"Error al leer el archivo: {error}"


def _detectar_columnas_categoricas(df: pd.DataFrame) -> list[str]:
    """Devuelve columnas que pueden usarse como filtro de categoría."""
    candidatas = []
    for col in df.columns:
        valores_unicos = df[col].nunique()
        # Categóricas de texto con cardinalidad razonable
        if df[col].dtype == object and 2 <= valores_unicos <= 60:
            candidatas.append(col)
    return candidatas


def _exportar_excel(df: pd.DataFrame) -> bytes:
    """Serializa el DataFrame en un buffer Excel y devuelve los bytes."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as escritor:
        df.to_excel(escritor, index=False, sheet_name="Datos")
    return buffer.getvalue()


def _renderizar_metricas(df: pd.DataFrame):
    """Renderiza las 4 tarjetas métricas superiores del dashboard."""
    num_registros = len(df)
    num_variables = len(df.columns)

    html_metricas = f"""
    <div style="display:grid; grid-template-columns: repeat(4,1fr); gap:16px; margin-bottom:28px;">

      <!-- Registros cargados -->
      <div class="metric-card">
        <div class="metric-icon" style="background:rgba(37,99,235,0.15); color:#2563EB;">👥</div>
        <div class="metric-info">
          <div class="metric-label">Registros cargados</div>
          <div class="metric-value">{num_registros}</div>
          <span class="badge-disponible">Disponible</span>
        </div>
      </div>

      <!-- Variables detectadas -->
      <div class="metric-card">
        <div class="metric-icon" style="background:rgba(168,85,247,0.15); color:#A855F7;">⊞</div>
        <div class="metric-info">
          <div class="metric-label">Variables detectadas</div>
          <div class="metric-value">{num_variables}</div>
          <div class="metric-sub">Columnas identificadas</div>
        </div>
      </div>

      <!-- Estado del modelo -->
      <div class="metric-card">
        <div class="metric-icon" style="background:rgba(34,197,94,0.15); color:#22C55E;">✓</div>
        <div class="metric-info">
          <div class="metric-label">Estado del modelo</div>
          <div class="metric-value" style="font-size:1.25rem;">No entrenado</div>
          <span class="badge-pendiente">Pendiente</span>
        </div>
      </div>

      <!-- Modelos guardados -->
      <div class="metric-card">
        <div class="metric-icon" style="background:rgba(245,158,11,0.15); color:#F59E0B;">💾</div>
        <div class="metric-info">
          <div class="metric-label">Modelos guardados</div>
          <div class="metric-value">0</div>
          <div class="metric-sub">Disponibles para consulta</div>
        </div>
      </div>

    </div>
    """
    st.html(html_metricas)


def _renderizar_tabla_paginada(df: pd.DataFrame):
    """Renderiza la tabla de datos con paginación manual."""
    col_info, col_pag = st.columns([3, 1])

    with col_pag:
        filas_por_pagina = st.selectbox(
            "Filas por página",
            options=[10, 25, 50, 100],
            index=0,
            key="filas_pagina",
        )

    total_registros = len(df)
    total_paginas = max(1, -(-total_registros // filas_por_pagina))  # ceil division

    # Estado de página en session_state
    if "pagina_actual" not in st.session_state:
        st.session_state.pagina_actual = 1

    pagina_actual = st.session_state.pagina_actual
    pagina_actual = min(pagina_actual, total_paginas)

    inicio = (pagina_actual - 1) * filas_por_pagina
    fin = min(inicio + filas_por_pagina, total_registros)
    df_pagina = df.iloc[inicio:fin].reset_index(drop=True)
    df_pagina.index = df_pagina.index + inicio + 1  # índice 1-based

    with col_info:
        st.markdown(
            f"<p style='color:#94A3B8; font-size:0.8rem; margin-top:36px;'>"
            f"Mostrando {inicio+1} a {fin} de {total_registros} registros</p>",
            unsafe_allow_html=True,
        )

    # Tabla con estilo nativo de Streamlit
    st.dataframe(
        df_pagina,
        use_container_width=True,
        height=min(400, (len(df_pagina) + 1) * 38 + 4),
    )

    # Controles de paginación
    _renderizar_paginacion(pagina_actual, total_paginas)


def _renderizar_paginacion(pagina_actual: int, total_paginas: int):
    """Botones de paginación."""
    cols = st.columns([2, 1, 1, 1, 1, 1, 1, 2])

    # Primera página
    with cols[1]:
        if st.button("«", key="pag_primera", help="Primera página"):
            st.session_state.pagina_actual = 1
            st.rerun()

    # Anterior
    with cols[2]:
        if st.button("‹", key="pag_anterior", help="Página anterior"):
            st.session_state.pagina_actual = max(1, pagina_actual - 1)
            st.rerun()

    # Número de página actual
    with cols[3]:
        st.markdown(
            f"<div style='text-align:center; background:#2563EB; border-radius:6px;"
            f" padding:6px 0; font-size:0.8rem; font-weight:700; color:#fff;'>"
            f"{pagina_actual}</div>",
            unsafe_allow_html=True,
        )

    # Siguiente
    with cols[4]:
        if st.button("›", key="pag_siguiente", help="Siguiente página"):
            st.session_state.pagina_actual = min(total_paginas, pagina_actual + 1)
            st.rerun()

    # Última página
    with cols[5]:
        if st.button("»", key="pag_ultima", help="Última página"):
            st.session_state.pagina_actual = total_paginas
            st.rerun()


# ─────────────────────────────────────────────────
# Vista principal (pública)
# ─────────────────────────────────────────────────

def renderizar_vista_datos():
    """Renderiza la interfaz para la visualización de datos (RF-01 a RF-04)."""

    # ── Header de página ──────────────────────────
    col_titulo, col_boton = st.columns([3, 1])
    with col_titulo:
        st.markdown(
            "<h1 class='page-title'>Dashboard de procesamiento</h1>"
            "<p class='page-subtitle'>Administra los datos, el entrenamiento y los resultados desde un solo lugar.</p>",
            unsafe_allow_html=True,
        )

    # ── Estado del dataset en session_state ───────
    if "dataframe_cargado" not in st.session_state:
        st.session_state.dataframe_cargado = None
    if "nombre_archivo" not in st.session_state:
        st.session_state.nombre_archivo = None
    if "fecha_carga" not in st.session_state:
        st.session_state.fecha_carga = None

    df_cargado: pd.DataFrame | None = st.session_state.dataframe_cargado

    # ── Botón "Cargar conjunto de datos" ──────────
    with col_boton:
        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        abrir_carga = st.button(
            "⬆ Cargar conjunto de datos",
            key="btn_abrir_carga",
            use_container_width=True,
        )

    # ── Métricas superiores ───────────────────────
    if df_cargado is not None:
        _renderizar_metricas(df_cargado)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Sección: Carga de archivo ─────────────────
    if abrir_carga or df_cargado is None:
        with st.expander(
            "⬆ Cargar archivo de datos",
            expanded=(df_cargado is None),
        ):
            st.markdown(
                "<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:16px;'>"
                "Selecciona un archivo <strong style='color:#F8FAFC'>CSV</strong> o "
                "<strong style='color:#F8FAFC'>Excel (.xlsx / .xls)</strong> para cargarlo.</p>",
                unsafe_allow_html=True,
            )

            archivo = st.file_uploader(
                "Selecciona un archivo",
                type=["csv", "xlsx", "xls"],
                key="file_uploader",
                label_visibility="collapsed",
            )

            if archivo is not None:
                with st.spinner("Cargando y validando archivo…"):
                    df_nuevo, error = _cargar_archivo(archivo)

                if error:
                    st.error(f"❌ {error}")
                else:
                    from datetime import datetime

                    st.session_state.dataframe_cargado = df_nuevo
                    st.session_state.nombre_archivo = archivo.name
                    st.session_state.fecha_carga = datetime.now().strftime("%d/%m/%Y %H:%M")
                    st.session_state.pagina_actual = 1
                    st.session_state.filtro_columna = None
                    st.session_state.filtro_valor = "Todos"
                    st.success(
                        f"✅ Archivo **{archivo.name}** cargado correctamente — "
                        f"{len(df_nuevo):,} registros, {len(df_nuevo.columns)} columnas."
                    )
                    st.rerun()

    # ── Sección: Conjunto de datos ─────────────────
    if df_cargado is not None:

        # Sub-encabezado de sección
        col_sec_titulo, col_sec_controles = st.columns([2, 2])
        with col_sec_titulo:
            st.markdown(
                "<div class='section-title'>Conjunto de datos</div>"
                "<div class='section-subtitle'>Consulta y administra la información importada.</div>",
                unsafe_allow_html=True,
            )

        # ── Controles: Filtro + Exportar ──────────
        columnas_categoricas = _detectar_columnas_categoricas(df_cargado)

        with col_sec_controles:
            col_filtro_col, col_filtro_val, col_exportar = st.columns([2, 2, 1.2])

            with col_filtro_col:
                if columnas_categoricas:
                    columna_filtro = st.selectbox(
                        "Filtrar por columna",
                        options=["— Sin filtro —"] + columnas_categoricas,
                        key="sel_columna_filtro",
                    )
                else:
                    columna_filtro = "— Sin filtro —"
                    st.markdown(
                        "<p style='color:#64748B; font-size:0.75rem; margin-top:30px;'>"
                        "Sin columnas categóricas disponibles</p>",
                        unsafe_allow_html=True,
                    )

            # Valores de la columna seleccionada
            with col_filtro_val:
                if columna_filtro != "— Sin filtro —":
                    valores_unicos = ["Todos"] + sorted(
                        df_cargado[columna_filtro].dropna().unique().tolist(),
                        key=str,
                    )
                    valor_filtro = st.selectbox(
                        f"Categoría · {columna_filtro}",
                        options=valores_unicos,
                        key="sel_valor_filtro",
                    )
                else:
                    valor_filtro = "Todos"
                    st.markdown(
                        "<p style='color:#64748B; font-size:0.75rem; margin-top:30px;'>"
                        "Selecciona una columna primero</p>",
                        unsafe_allow_html=True,
                    )

        # ── Aplicar filtro ─────────────────────────
        if columna_filtro != "— Sin filtro —" and valor_filtro != "Todos":
            df_filtrado = df_cargado[df_cargado[columna_filtro].astype(str) == str(valor_filtro)]
        else:
            df_filtrado = df_cargado

        # ── Botón de exportar ──────────────────────
        with col_exportar:
            st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
            datos_excel = _exportar_excel(df_filtrado)
            nombre_descarga = (
                f"datos_filtrados_{columna_filtro}_{valor_filtro}.xlsx"
                if valor_filtro != "Todos"
                else "datos_exportados.xlsx"
            )
            st.download_button(
                label="⬇ Exportar",
                data=datos_excel,
                file_name=nombre_descarga,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="btn_exportar",
                use_container_width=True,
                help="Descarga los datos (filtrados o completos) en formato Excel",
            )

        # ── Info del dataset activo ────────────────
        html_info_bar = f"""
        <div class="dataset-info-bar">
          <div>
            <div class="dataset-label">Dataset activo</div>
            <div class="dataset-name">{st.session_state.nombre_archivo}</div>
          </div>
          <div style="text-align:right;">
            <div class="dataset-date-label">Actualizado</div>
            <div class="dataset-date">{st.session_state.fecha_carga}</div>
          </div>
        </div>
        """
        st.html(html_info_bar)

        # Indicador de filtro activo
        if columna_filtro != "— Sin filtro —" and valor_filtro != "Todos":
            st.info(
                f"🔍 Filtro activo: **{columna_filtro}** = **{valor_filtro}** "
                f"— {len(df_filtrado):,} de {len(df_cargado):,} registros mostrados."
            )

        # ── Tabla paginada ─────────────────────────
        _renderizar_tabla_paginada(df_filtrado)

    else:
        # Estado vacío: no hay dataset cargado
        st.markdown(
            """
            <div style="
                border: 2px dashed #2A3953;
                border-radius: 12px;
                padding: 64px 24px;
                text-align: center;
                background: #111827;
                margin-top: 24px;
            ">
              <div style="font-size:3rem; margin-bottom:16px;">📂</div>
              <div style="font-size:1.1rem; font-weight:600; color:#F8FAFC; margin-bottom:8px;">
                Aún no hay datos cargados
              </div>
              <div style="font-size:0.875rem; color:#64748B;">
                Haz clic en <strong style="color:#2563EB;">Cargar conjunto de datos</strong> para comenzar.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
