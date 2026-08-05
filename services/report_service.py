"""Generación del reporte estadístico inicial en PDF (RF-06)."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from math import ceil
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import ScatterPlot
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.graphics.widgets.markers import makeMarker

import pandas as pd

from services.results_service import ProyeccionPCA, ServicioResultados
from services.statistics_service import ResumenEstadistico, ServicioEstadisticas
from services.training_service import ResultadoEntrenamiento


class ServicioReportes:
    """Crea documentos PDF a partir de las estadísticas de RF-05."""

    _AZUL = colors.HexColor("#C85D3F")
    _VERDE = colors.HexColor("#718A66")
    _VIOLETA = colors.HexColor("#C7973D")
    _AMBAR = colors.HexColor("#A96743")
    _ROSA = colors.HexColor("#9A554E")
    _TINTA = colors.HexColor("#2B2622")
    _GRIS = colors.HexColor("#6E5C50")
    _FONDO_TABLA = colors.HexColor("#F4E8D9")

    _PALETA_GRUPOS = [
        colors.HexColor("#2388FF"),
        colors.HexColor("#43D78A"),
        colors.HexColor("#B06DF5"),
        colors.HexColor("#F59E2F"),
        colors.HexColor("#EF5576"),
        colors.HexColor("#22B8CF"),
        colors.HexColor("#F4C95D"),
        colors.HexColor("#8B9A77"),
    ]

    @staticmethod
    def _estilos() -> dict[str, ParagraphStyle]:
        base = getSampleStyleSheet()
        return {
            "titulo": ParagraphStyle(
                "TituloClusterLab",
                parent=base["Title"],
                fontName="Helvetica-Bold",
                fontSize=22,
                leading=27,
                alignment=0,
                textColor=ServicioReportes._TINTA,
                spaceAfter=5,
            ),
            "subtitulo": ParagraphStyle(
                "SubtituloClusterLab",
                parent=base["Normal"],
                fontSize=8.5,
                leading=12,
                textColor=ServicioReportes._GRIS,
                spaceAfter=14,
            ),
            "seccion": ParagraphStyle(
                "SeccionClusterLab",
                parent=base["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=13,
                leading=17,
                textColor=ServicioReportes._TINTA,
                spaceBefore=8,
                spaceAfter=8,
            ),
            "cuerpo": ParagraphStyle(
                "CuerpoClusterLab",
                parent=base["BodyText"],
                fontSize=9.5,
                leading=14,
                textColor=colors.HexColor("#26394D"),
            ),
            "tabla": ParagraphStyle(
                "TablaClusterLab",
                parent=base["BodyText"],
                fontSize=6.5,
                leading=8,
                spaceBefore=0,
                spaceAfter=0,
                textColor=colors.HexColor("#1D304A"),
            ),
            "tabla_encabezado": ParagraphStyle(
                "TablaEncabezadoClusterLab",
                parent=base["BodyText"],
                fontName="Helvetica-Bold",
                fontSize=6.5,
                leading=8,
                spaceBefore=0,
                spaceAfter=0,
                alignment=TA_CENTER,
                textColor=colors.white,
            ),
            "metadato_clave": ParagraphStyle(
                "MetadatoClaveClusterLab",
                parent=base["BodyText"],
                fontName="Helvetica-Bold",
                fontSize=9,
                leading=11,
                textColor=ServicioReportes._TINTA,
                spaceBefore=0,
                spaceAfter=0,
            ),
            "metadato_valor": ParagraphStyle(
                "MetadatoValorClusterLab",
                parent=base["BodyText"],
                fontName="Helvetica",
                fontSize=9,
                leading=11,
                textColor=colors.HexColor("#3C332C"),
                spaceBefore=0,
                spaceAfter=0,
            ),
        }

    @staticmethod
    def _grafica_barras(
        titulo: str,
        categorias: list[str],
        valores: list[float],
        color: colors.Color,
        etiqueta_eje_y: str,
        maximo_eje: float | None = None,
        ancho: float = 680,
        alto: float = 238,
        mostrar_valores: bool = False,
        etiqueta_eje_x: str | None = None,
    ) -> Drawing:
        """Crea una gráfica vectorial nítida sin archivos temporales."""
        dibujo = Drawing(ancho, alto)
        dibujo.add(
            String(
                0,
                alto - 14,
                titulo,
                fontName="Helvetica-Bold",
                fontSize=9.5,
                fillColor=ServicioReportes._TINTA,
            )
        )

        grafica = VerticalBarChart()
        grafica.x = 45
        grafica.y = 32
        grafica.width = ancho - 65
        grafica.height = alto - 78
        grafica.data = [valores]
        grafica.categoryAxis.categoryNames = categorias
        grafica.categoryAxis.labels.fontName = "Helvetica"
        grafica.categoryAxis.labels.fontSize = 5.8 if ancho < 500 else 6.5
        grafica.categoryAxis.labels.boxAnchor = "n"
        grafica.valueAxis.labels.fontName = "Helvetica"
        grafica.valueAxis.labels.fontSize = 7
        grafica.valueAxis.valueMin = 0
        limite_superior = maximo_eje or max(1, max(valores) * 1.18)
        grafica.valueAxis.valueMax = limite_superior
        grafica.valueAxis.valueStep = max(1, round(grafica.valueAxis.valueMax / 5))
        grafica.bars[0].fillColor = color
        grafica.bars[0].strokeColor = color
        grafica.bars[0].strokeWidth = 0
        dibujo.add(grafica)
        if etiqueta_eje_y:
            etiqueta_vertical = Label()
            etiqueta_vertical.x = grafica.x - 33
            etiqueta_vertical.y = grafica.y + (grafica.height / 2)
            etiqueta_vertical.setText(etiqueta_eje_y)
            etiqueta_vertical.fontName = "Helvetica"
            etiqueta_vertical.fontSize = 7
            etiqueta_vertical.fillColor = ServicioReportes._GRIS
            etiqueta_vertical.angle = 90
            etiqueta_vertical.boxAnchor = "c"
            etiqueta_vertical.textAnchor = "middle"
            dibujo.add(etiqueta_vertical)
        if etiqueta_eje_x:
            dibujo.add(
                String(
                    grafica.x + (grafica.width / 2),
                    3,
                    etiqueta_eje_x,
                    fontName="Helvetica",
                    fontSize=7,
                    fillColor=ServicioReportes._GRIS,
                    textAnchor="middle",
                )
            )
        if mostrar_valores:
            ancho_categoria = grafica.width / len(valores)
            for indice, valor in enumerate(valores):
                x = grafica.x + ancho_categoria * (indice + 0.5)
                y = grafica.y + (grafica.height * valor / limite_superior) + 5
                dibujo.add(
                    String(
                        x,
                        y,
                        f"{valor:.2f}",
                        textAnchor="middle",
                        fontName="Helvetica-Bold",
                        fontSize=7.5,
                        fillColor=ServicioReportes._TINTA,
                    )
                )
        return dibujo

    @classmethod
    def _grafica_pastel(
        cls,
        titulo: str,
        categorias: list[str],
        valores: list[float],
        ancho: float = 680,
        alto: float = 200,
    ) -> Drawing:
        """Representa proporciones Likert con una gráfica de pastel legible."""
        dibujo = Drawing(ancho, alto)
        dibujo.add(
            String(
                0,
                alto - 14,
                titulo,
                fontName="Helvetica-Bold",
                fontSize=10,
                fillColor=cls._TINTA,
            )
        )
        pastel = Pie()
        pastel.x = 255
        pastel.y = 20
        pastel.width = 155
        pastel.height = 155
        total = sum(valores)
        pastel.data = valores
        pastel.labels = [
            f"{categoria} ({valor / total * 100:.1f}%)"
            for categoria, valor in zip(categorias, valores, strict=True)
        ]
        pastel.sideLabels = True
        pastel.slices.strokeColor = colors.HexColor("#FFFDF9")
        pastel.slices.strokeWidth = 1
        colores = [
            colors.HexColor("#8F4D3E"),
            colors.HexColor("#C4774E"),
            colors.HexColor("#D5A33A"),
            colors.HexColor("#718A66"),
            colors.HexColor("#A96743"),
        ]
        for indice, color in enumerate(colores):
            pastel.slices[indice].fillColor = color
            pastel.slices[indice].labelRadius = 1.18
            pastel.slices[indice].fontName = "Helvetica"
            pastel.slices[indice].fontSize = 7
        dibujo.add(pastel)
        return dibujo

    @classmethod
    def _tarjetas(cls, indicadores: list[tuple[str, str, colors.Color]]) -> Table:
        """Construye una cuadrícula compacta de tarjetas de indicadores clave."""
        celdas = []
        for etiqueta, valor, color in indicadores:
            celdas.append(
                Table(
                    [[etiqueta], [valor]],
                    colWidths=[1.65 * inch],
                    rowHeights=[0.22 * inch, 0.34 * inch],
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FBF5EC")),
                            ("LINEABOVE", (0, 0), (-1, 0), 3, color),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 6.5),
                            ("FONTSIZE", (0, 1), (-1, 1), 15),
                            ("TEXTCOLOR", (0, 0), (-1, 0), cls._GRIS),
                            ("TEXTCOLOR", (0, 1), (-1, 1), cls._TINTA),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    ),
                )
            )
        return Table([celdas], colWidths=[1.72 * inch] * len(indicadores), hAlign="LEFT")

    @classmethod
    def _tarjetas_resumen(cls, resumen: ResumenEstadistico) -> Table:
        """Presenta los indicadores clave del resumen estadístico."""
        indicadores = [
            ("REGISTROS", str(resumen.cantidad_registros), cls._AZUL),
            ("VARIABLES", str(resumen.cantidad_variables), cls._VIOLETA),
            ("RESPUESTAS", str(resumen.cantidad_registros * resumen.cantidad_variables), cls._VERDE),
            ("FALTANTES", str(int(resumen.faltantes_por_rasgo.sum())), cls._AMBAR),
        ]
        return cls._tarjetas(indicadores)

    @classmethod
    def _tarjetas_entrenamiento(cls, resultado: ResultadoEntrenamiento) -> Table:
        """Presenta los indicadores clave del entrenamiento K-Means."""
        indicadores = [
            ("GRUPOS", str(resultado.k_usado), cls._AZUL),
            ("REGISTROS", str(len(resultado.asignaciones)), cls._VIOLETA),
            ("VARIABLES", str(len(resultado.columnas)), cls._VERDE),
            ("SILHOUETTE", f"{resultado.silhouette:.3f}", cls._AMBAR),
            ("INERCIA", f"{resultado.inercia:,.1f}", cls._ROSA),
        ]
        return cls._tarjetas(indicadores)

    @staticmethod
    def _interpretacion(resumen: ResumenEstadistico) -> str:
        """Produce una interpretación breve, descriptiva y no clínica."""
        dimension_mayor = resumen.promedio_dimensiones.idxmax()
        valor_mayor = resumen.promedio_dimensiones.max()
        dimension_menor = resumen.promedio_dimensiones.idxmin()
        valor_menor = resumen.promedio_dimensiones.min()
        faltantes = int(resumen.faltantes_por_rasgo.sum())

        definiciones = {
            "Extraversión": "la sociabilidad, la energía en interacciones y la iniciativa para relacionarse con otras personas",
            "Estabilidad emocional": "la calma, la regulación emocional y la recuperación ante situaciones de presión",
            "Apertura a la experiencia": "la curiosidad, el interés por aprender y la disposición hacia ideas o experiencias nuevas",
            "Responsabilidad": "la planificación, el orden y el cumplimiento de tareas o compromisos",
            "Amabilidad": "la cooperación, la empatía y la consideración hacia otras personas",
        }
        texto = (
            f"El análisis reúne {resumen.cantidad_registros} registros y "
            f"{resumen.cantidad_variables} rasgos Big Five. "
            f"El promedio más alto fue <b>{escape(str(dimension_mayor))}</b> "
            f"({valor_mayor:.2f}/5); en este cuestionario, esta dimensión representa "
            f"{definiciones[dimension_mayor]}. "
            f"El promedio más bajo fue <b>{escape(str(dimension_menor))}</b> "
            f"({valor_menor:.2f}/5), relacionado con {definiciones[dimension_menor]}. "
            "Estos resultados describen el comportamiento promedio del conjunto de datos; "
            "no constituyen un diagnóstico individual. "
        )
        if faltantes:
            texto += f"Se identificaron <b>{faltantes}</b> respuestas faltantes que deben revisarse antes del entrenamiento."
        else:
            texto += "No se identificaron respuestas faltantes en las variables analizadas."
        return texto

    @staticmethod
    def _tabla_resumen(
        resumen: ResumenEstadistico, estilos: dict[str, ParagraphStyle]
    ) -> Table:
        encabezados = [
            "Rasgo",
            "Media",
            "Mediana",
            "Moda",
            "Varianza",
            "Desv. est.",
            "CV (%)",
            "Mín.",
            "Máx.",
        ]
        filas = [
            [Paragraph(encabezado, estilos["tabla_encabezado"]) for encabezado in encabezados]
        ]
        for pregunta, valores in resumen.estadisticas_por_rasgo.iterrows():
            filas.append(
                [
                    Paragraph(escape(str(pregunta)), estilos["tabla"]),
                    f"{valores['Media']:.2f}",
                    f"{valores['Mediana']:.2f}",
                    f"{valores['Moda']:.2f}",
                    f"{valores['Varianza']:.2f}",
                    f"{valores['Desviación estándar']:.2f}",
                    f"{valores['Coeficiente de variación (%)']:.2f}",
                    f"{valores['Mínimo']:.2f}",
                    f"{valores['Máximo']:.2f}",
                ]
            )

        tabla = Table(
            filas,
            colWidths=[
                2.65 * inch,
                0.58 * inch,
                0.65 * inch,
                0.52 * inch,
                0.68 * inch,
                0.72 * inch,
                0.62 * inch,
                0.5 * inch,
                0.5 * inch,
            ],
            repeatRows=1,
            hAlign="LEFT",
        )
        tabla.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), ServicioReportes._TINTA),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8C3A5")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ServicioReportes._FONDO_TABLA]),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        return tabla

    @staticmethod
    def _tabla_frecuencias(
        tabla_frecuencia: pd.DataFrame,
        estilos: dict[str, ParagraphStyle],
    ) -> Table:
        """Convierte una distribución agrupada en una tabla lista para el PDF."""
        encabezados = ["Intervalo", "Marca de clase", "f", "Fr", "%", "F"]
        filas = [
            [Paragraph(encabezado, estilos["tabla_encabezado"]) for encabezado in encabezados]
        ]
        for _, fila in tabla_frecuencia.iterrows():
            filas.append(
                [
                    Paragraph(escape(str(fila["Intervalo"])), estilos["tabla"]),
                    f"{float(fila['Marca de Clase']):.2f}",
                    str(int(fila["f"])),
                    f"{float(fila['Fr']):.4f}",
                    f"{float(fila['%']):.2f}%",
                    str(int(fila["F"])),
                ]
            )

        tabla = Table(
            filas,
            colWidths=[2.05 * inch, 1.1 * inch, 0.65 * inch, 0.8 * inch, 0.7 * inch, 0.65 * inch],
            repeatRows=1,
            hAlign="LEFT",
        )
        tabla.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), ServicioReportes._TINTA),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8C3A5")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ServicioReportes._FONDO_TABLA]),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("FONTSIZE", (1, 1), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ]
            )
        )
        return tabla

    @classmethod
    def _histograma_con_poligono(
        cls,
        tabla_frecuencia: pd.DataFrame,
        rasgo: str,
        ancho: float = 680,
        alto: float = 238,
    ) -> Drawing:
        """Dibuja la misma distribución por límites que muestra la vista Streamlit."""
        dibujo = Drawing(ancho, alto)
        dibujo.add(
            String(
                0,
                alto - 14,
                f"Distribución de {rasgo}",
                fontName="Helvetica-Bold",
                fontSize=9.5,
                fillColor=cls._TINTA,
            )
        )

        frecuencias = tabla_frecuencia["f"].astype(float).tolist()
        limites = (
            tabla_frecuencia["Intervalo"]
            .str.extract(
                r"^\[\s*([-+]?\d+(?:\.\d+)?)\s*-\s*([-+]?\d+(?:\.\d+)?)",
                expand=True,
            )
            .astype(float)
        )
        limites_inferiores = limites[0].tolist()
        limites_superiores = limites[1].tolist()
        if not limites_inferiores or len(limites_inferiores) != len(frecuencias):
            return dibujo

        limites_eje_x = [*limites_inferiores, limites_superiores[-1]]
        minimo_x, maximo_x = limites_eje_x[0], limites_eje_x[-1]
        izquierda, base = 46, 36
        ancho_grafica, alto_grafica = ancho - 70, alto - 78
        maximo_frecuencia = max(frecuencias, default=1)
        limite_y = max(1, int(ceil(maximo_frecuencia * 1.2)))
        paso_y = max(1, int(ceil(limite_y / 5)))
        limite_y = paso_y * int(ceil(limite_y / paso_y))

        for valor_y in range(0, limite_y + 1, paso_y):
            y = base + (valor_y / limite_y) * alto_grafica
            dibujo.add(
                Line(
                    izquierda,
                    y,
                    izquierda + ancho_grafica,
                    y,
                    strokeColor=colors.HexColor("#E8DDD0"),
                    strokeWidth=0.45,
                )
            )
            dibujo.add(
                String(
                    izquierda - 7,
                    y - 2.5,
                    str(valor_y),
                    fontName="Helvetica",
                    fontSize=6.5,
                    fillColor=cls._GRIS,
                    textAnchor="end",
                )
            )

        dibujo.add(Line(izquierda, base, izquierda, base + alto_grafica, strokeColor=cls._TINTA, strokeWidth=0.7))
        dibujo.add(Line(izquierda, base, izquierda + ancho_grafica, base, strokeColor=cls._TINTA, strokeWidth=0.7))

        def posicion_x(valor: float) -> float:
            if maximo_x == minimo_x:
                return izquierda
            return izquierda + ((valor - minimo_x) / (maximo_x - minimo_x)) * ancho_grafica

        puntos: list[tuple[float, float]] = []
        for frecuencia, limite_inferior, limite_superior in zip(
            frecuencias,
            limites_inferiores,
            limites_superiores,
            strict=True,
        ):
            x = posicion_x(limite_inferior)
            ancho_barra = posicion_x(limite_superior) - x
            altura_barra = (frecuencia / limite_y) * alto_grafica
            dibujo.add(
                Rect(
                    x,
                    base,
                    ancho_barra,
                    altura_barra,
                    fillColor=colors.HexColor("#B876FF"),
                    strokeColor=colors.HexColor("#8A52C7"),
                    strokeWidth=0.55,
                )
            )
            centro_x = x + ancho_barra / 2
            centro_y = base + altura_barra
            puntos.append((centro_x, centro_y))
            dibujo.add(String(centro_x, centro_y + 5, str(int(frecuencia)), fontName="Helvetica-Bold", fontSize=6.5, fillColor=cls._TINTA, textAnchor="middle"))

        if len(puntos) >= 2:
            dibujo.add(PolyLine(puntos, strokeColor=colors.HexColor("#4C286E"), strokeWidth=2.2, fillColor=None))
        for punto_x, punto_y in puntos:
            dibujo.add(Circle(punto_x, punto_y, 3.1, fillColor=colors.HexColor("#4C286E"), strokeColor=colors.white, strokeWidth=0.8))

        for indice, limite in enumerate(limites_eje_x):
            posicion = posicion_x(limite)
            if indice == 0:
                anclaje = "start"
            elif indice == len(limites_eje_x) - 1:
                anclaje = "end"
            else:
                anclaje = "middle"
            dibujo.add(
                String(
                    posicion,
                    base - 10,
                    f"{limite:.2f}",
                    fontName="Helvetica",
                    fontSize=6.2,
                    fillColor=cls._GRIS,
                    textAnchor=anclaje,
                )
            )
        dibujo.add(
            String(
                izquierda + ancho_grafica / 2,
                5,
                f"Puntaje del rasgo {rasgo.lower()}",
                fontName="Helvetica",
                fontSize=7,
                fillColor=cls._GRIS,
                textAnchor="middle",
            )
        )
        etiqueta_y = Label()
        etiqueta_y.x = 8
        etiqueta_y.y = base + alto_grafica / 2
        etiqueta_y.setText("frecuencia")
        etiqueta_y.fontName = "Helvetica"
        etiqueta_y.fontSize = 7
        etiqueta_y.fillColor = cls._GRIS
        etiqueta_y.angle = 90
        etiqueta_y.boxAnchor = "c"
        dibujo.add(etiqueta_y)
        return dibujo

    @staticmethod
    def _tabla_evaluacion_k(
        resultado: ResultadoEntrenamiento, estilos: dict[str, ParagraphStyle]
    ) -> Table:
        """Compara la inercia y el Silhouette de cada valor de K evaluado."""
        if not resultado.evaluaciones:
            return Paragraph(
                "No se evaluaron valores adicionales de K porque este resultado "
                f"proviene de un reentrenamiento. Se conservó K = {resultado.k_usado} "
                "del modelo base.",
                estilos["cuerpo"],
            )

        encabezados = ["K evaluado", "Inercia", "Silhouette", "Resultado"]
        filas = [
            [Paragraph(encabezado, estilos["tabla_encabezado"]) for encabezado in encabezados]
        ]
        for evaluacion in resultado.evaluaciones:
            if evaluacion.k == resultado.k_usado:
                resultado_texto = "Utilizado"
            elif evaluacion.k == resultado.k_recomendado:
                resultado_texto = "Recomendado"
            else:
                resultado_texto = ""
            filas.append(
                [
                    str(evaluacion.k),
                    f"{evaluacion.inercia:,.2f}",
                    f"{evaluacion.silhouette:.3f}",
                    resultado_texto,
                ]
            )
        tabla = Table(
            filas,
            colWidths=[1.3 * inch, 1.7 * inch, 1.7 * inch, 1.7 * inch],
            repeatRows=1,
            hAlign="LEFT",
        )
        tabla.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), ServicioReportes._TINTA),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8C3A5")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ServicioReportes._FONDO_TABLA]),
                    ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return tabla

    @staticmethod
    def _tabla_grupos(
        resumen_grupos: pd.DataFrame, estilos: dict[str, ParagraphStyle]
    ) -> Table:
        """Muestra la cantidad y el porcentaje de registros de cada grupo."""
        encabezados = ["Grupo", "Registros", "Porcentaje"]
        filas = [
            [Paragraph(encabezado, estilos["tabla_encabezado"]) for encabezado in encabezados]
        ]
        for _, fila in resumen_grupos.iterrows():
            filas.append(
                [
                    escape(str(fila["Grupo"])),
                    str(int(fila["Registros"])),
                    f"{float(fila['Porcentaje']) * 100:.1f}%",
                ]
            )
        tabla = Table(
            filas,
            colWidths=[2.6 * inch, 2.2 * inch, 2.2 * inch],
            repeatRows=1,
            hAlign="LEFT",
        )
        tabla.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), ServicioReportes._TINTA),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8C3A5")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ServicioReportes._FONDO_TABLA]),
                    ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return tabla

    @staticmethod
    def _tabla_centros_entrenamiento(
        centros: pd.DataFrame, estilos: dict[str, ParagraphStyle]
    ) -> Table:
        """Muestra los centros de cada grupo en la escala original de las variables.

        Las variables se listan en filas y los grupos en columnas: los datasets
        suelen tener muchas variables pero pocos grupos, así la tabla se
        mantiene legible sin desbordar el ancho de la página, y puede
        continuar en la siguiente página si no cabe completa.
        """
        grupos = centros["Grupo"].astype(str).tolist()
        variables = [columna for columna in centros.columns if columna != "Grupo"]
        valores_por_grupo = [
            centros.loc[centros.index[indice]] for indice in range(len(centros))
        ]

        encabezados = ["Variable"] + grupos
        filas = [
            [Paragraph(escape(encabezado), estilos["tabla_encabezado"]) for encabezado in encabezados]
        ]
        for variable in variables:
            valores_fila = [Paragraph(escape(str(variable)), estilos["tabla"])]
            for fila_centro in valores_por_grupo:
                valores_fila.append(f"{float(fila_centro[variable]):.2f}")
            filas.append(valores_fila)

        ancho_variable = 4.3 * inch
        ancho_disponible = 9.0 * inch - ancho_variable
        ancho_grupo = max(0.7 * inch, ancho_disponible / max(1, len(grupos)))
        colWidths = [ancho_variable] + [ancho_grupo] * len(grupos)

        tabla = Table(filas, colWidths=colWidths, repeatRows=1, hAlign="LEFT")
        tabla.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), ServicioReportes._TINTA),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8C3A5")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ServicioReportes._FONDO_TABLA]),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return tabla

    @staticmethod
    def _parrafos_interpretaciones_grupos(
        interpretaciones: pd.DataFrame,
        perfiles: pd.DataFrame,
        estilos: dict[str, ParagraphStyle],
    ) -> list:
        """Convierte las lecturas por rasgo en texto para el PDF."""
        parrafos = [
            Paragraph(
                "Cada frase compara el promedio del grupo con el promedio general de "
                "referencia. Las explicaciones describen tendencias del grupo, no de "
                "una persona individual.",
                estilos["cuerpo"],
            )
        ]
        perfiles_por_grupo = perfiles.set_index("Grupo")["Perfil"]
        for grupo, lecturas_grupo in interpretaciones.groupby("Grupo", sort=False):
            parrafos.extend(
                [
                    Spacer(1, 7),
                    Paragraph(f"<b>{escape(str(grupo))}</b>", estilos["cuerpo"]),
                    Paragraph(
                        f"<b>Perfil predominante:</b> "
                        f"{escape(str(perfiles_por_grupo[grupo]))}",
                        estilos["cuerpo"],
                    ),
                ]
            )
            for _, fila in lecturas_grupo.iterrows():
                texto = (
                    f"<b>{escape(str(fila['Rasgo']))}:</b> "
                    f"{escape(str(fila['Comparación']))} "
                    f"({float(fila['Valor del grupo']):.2f} frente a "
                    f"{float(fila['Promedio general']):.2f}). "
                    f"{escape(str(fila['Interpretación']))}"
                )
                parrafos.append(Paragraph(texto, estilos["cuerpo"]))
        return parrafos

    @classmethod
    def _grafica_dispersion(
        cls, proyeccion: ProyeccionPCA, ancho: float = 680, alto: float = 260
    ) -> Drawing:
        """Representa los registros y los centros de grupo en un plano PCA."""
        dibujo = Drawing(ancho, alto)
        dibujo.add(
            String(
                0,
                alto - 14,
                "Mapa de similitud entre registros (proyección PCA)",
                fontName="Helvetica-Bold",
                fontSize=9.5,
                fillColor=cls._TINTA,
            )
        )

        grafico = ScatterPlot()
        grafico.x = 55
        grafico.y = 34
        grafico.width = ancho - 90
        grafico.height = alto - 74
        grafico.lineLabelFormat = None
        grafico.xLabel = ""
        grafico.yLabel = ""

        grupos_ordenados = sorted(
            proyeccion.puntos["Grupo"].unique(),
            key=lambda grupo: int(str(grupo).split()[-1]),
        )
        series = []
        for grupo in grupos_ordenados:
            subconjunto = proyeccion.puntos[proyeccion.puntos["Grupo"] == grupo]
            series.append(
                list(zip(subconjunto["Componente 1"], subconjunto["Componente 2"]))
            )
        series.append(
            list(zip(proyeccion.centros["Componente 1"], proyeccion.centros["Componente 2"]))
        )
        grafico.data = series

        for indice, _ in enumerate(grupos_ordenados):
            color = cls._PALETA_GRUPOS[indice % len(cls._PALETA_GRUPOS)]
            grafico.lines[indice].symbol = makeMarker("FilledCircle")
            grafico.lines[indice].symbol.fillColor = color
            grafico.lines[indice].symbol.strokeColor = None
            grafico.lines[indice].symbol.size = 3.4

        indice_centros = len(grupos_ordenados)
        grafico.lines[indice_centros].symbol = makeMarker("FilledCross")
        grafico.lines[indice_centros].symbol.fillColor = cls._TINTA
        grafico.lines[indice_centros].symbol.strokeColor = cls._TINTA
        grafico.lines[indice_centros].symbol.size = 8

        grafico.xValueAxis.labels.fontName = "Helvetica"
        grafico.xValueAxis.labels.fontSize = 6.5
        grafico.yValueAxis.labels.fontName = "Helvetica"
        grafico.yValueAxis.labels.fontSize = 6.5
        grafico.xValueAxis.visibleGrid = True
        grafico.yValueAxis.visibleGrid = True
        grafico.xValueAxis.gridStrokeColor = colors.HexColor("#E7DCC9")
        grafico.yValueAxis.gridStrokeColor = colors.HexColor("#E7DCC9")
        grafico.xValueAxis.strokeColor = cls._GRIS
        grafico.yValueAxis.strokeColor = cls._GRIS
        dibujo.add(grafico)

        varianza_1, varianza_2 = proyeccion.varianza_explicada
        dibujo.add(
            String(
                grafico.x + grafico.width / 2,
                4,
                f"Componente 1 ({varianza_1:.1%} de varianza explicada)",
                textAnchor="middle",
                fontName="Helvetica",
                fontSize=7,
                fillColor=cls._GRIS,
            )
        )
        etiqueta_y = Label()
        etiqueta_y.x = grafico.x - 42
        etiqueta_y.y = grafico.y + (grafico.height / 2)
        etiqueta_y.setText(f"Componente 2 ({varianza_2:.1%})")
        etiqueta_y.angle = 90
        etiqueta_y.fontName = "Helvetica"
        etiqueta_y.fontSize = 7
        etiqueta_y.fillColor = cls._GRIS
        etiqueta_y.boxAnchor = "c"
        etiqueta_y.textAnchor = "middle"
        dibujo.add(etiqueta_y)
        return dibujo

    @staticmethod
    def _interpretacion_entrenamiento(
        resultado: ResultadoEntrenamiento, resumen_grupos: pd.DataFrame
    ) -> str:
        """Produce una interpretación breve y no clínica de los resultados de K-Means."""
        titulo_silhouette, explicacion_silhouette = ServicioResultados.interpretar_silhouette(
            resultado.silhouette
        )
        fila_mayor = resumen_grupos.loc[resumen_grupos["Registros"].idxmax()]
        fila_menor = resumen_grupos.loc[resumen_grupos["Registros"].idxmin()]

        candidatos = [evaluacion.k for evaluacion in resultado.evaluaciones]
        if candidatos:
            rango_evaluado = (
                str(candidatos[0])
                if len(candidatos) == 1
                else f"{candidatos[0]} a {candidatos[-1]}"
            )
            if resultado.k_usado == resultado.k_recomendado:
                seleccion = (
                    f"Se evaluaron valores de K entre {rango_evaluado} y se "
                    f"seleccionaron {resultado.k_usado} grupos por obtener el mayor "
                    "índice de Silhouette."
                )
            else:
                seleccion = (
                    f"Se evaluaron valores de K entre {rango_evaluado}; el valor con "
                    f"mayor Silhouette fue {resultado.k_recomendado}, aunque el "
                    f"entrenamiento final se realizó con {resultado.k_usado} grupos."
                )
        else:
            seleccion = f"El modelo se entrenó con {resultado.k_usado} grupos."

        texto = (
            f"El modelo K-Means agrupó {len(resultado.asignaciones)} registros en "
            f"{resultado.k_usado} grupos utilizando {len(resultado.columnas)} variables. "
            f"{seleccion} El índice de Silhouette obtenido fue de "
            f"<b>{resultado.silhouette:.3f}</b>, lo que indica una "
            f"<b>{titulo_silhouette.lower()}</b>: {explicacion_silhouette.lower()} "
            f"La inercia del modelo fue de {resultado.inercia:,.1f}; este valor resume "
            "la distancia interna de los grupos y es menor cuanto más compactos son. "
            f"El grupo con mayor cantidad de registros fue <b>{escape(str(fila_mayor['Grupo']))}</b> "
            f"con {int(fila_mayor['Registros'])} registros "
            f"({float(fila_mayor['Porcentaje']) * 100:.1f}%), mientras que "
            f"<b>{escape(str(fila_menor['Grupo']))}</b> agrupó la menor cantidad, con "
            f"{int(fila_menor['Registros'])} registros "
            f"({float(fila_menor['Porcentaje']) * 100:.1f}%). "
            "Estos resultados describen patrones estadísticos del conjunto analizado y "
            "no constituyen una clasificación definitiva ni un diagnóstico individual."
        )
        return texto

    @staticmethod
    def _encabezado_pie(canvas, documento) -> None:
        canvas.saveState()
        ancho, alto = landscape(letter)
        canvas.setStrokeColor(colors.HexColor("#D8C3A5"))
        canvas.line(0.5 * inch, alto - 0.43 * inch, ancho - 0.5 * inch, alto - 0.43 * inch)
        canvas.setFillColor(ServicioReportes._GRIS)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(0.5 * inch, alto - 0.32 * inch, "ClusterLab / Análisis Big Five")
        canvas.drawRightString(ancho - 0.5 * inch, 0.34 * inch, f"Página {documento.page}")
        canvas.restoreState()

    def generar_reporte_estadistico(
        self,
        resumen: ResumenEstadistico,
        nombre_dataset: str = "Dataset sin nombre",
        fecha_generacion: datetime | None = None,
    ) -> bytes:
        """Genera un PDF listo para descargar con datos, gráficas e interpretación."""
        if not isinstance(resumen, ResumenEstadistico):
            raise TypeError("El reporte requiere un resumen estadístico válido.")

        fecha = fecha_generacion or datetime.now()
        estilos = self._estilos()
        salida = BytesIO()
        documento = SimpleDocTemplate(
            salida,
            pagesize=landscape(letter),
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
            topMargin=0.62 * inch,
            bottomMargin=0.56 * inch,
            title="Reporte estadístico inicial - ClusterLab",
            author="ClusterLab",
        )

        historia = [
            Paragraph("INFORME ACADÉMICO / BIG FIVE", estilos["subtitulo"]),
            Paragraph("Estadística descriptiva", estilos["titulo"]),
            Paragraph(
                "Resultados previos al entrenamiento del modelo de agrupamiento",
                estilos["subtitulo"],
            ),
            HRFlowable(width="100%", thickness=1, color=self._AZUL, spaceAfter=12),
        ]
        metadatos = [
            ["Dataset", nombre_dataset],
            ["Fecha de generación", fecha.strftime("%d/%m/%Y %H:%M")],
            ["Registros analizados", str(resumen.cantidad_registros)],
            ["Variables analizadas", str(resumen.cantidad_variables)],
            [
                "Valores de rasgo evaluados",
                str(resumen.cantidad_registros * resumen.cantidad_variables),
            ],
            ["Datos faltantes", str(int(resumen.faltantes_por_rasgo.sum()))],
        ]
        tabla_metadatos = Table(
            [
                [
                    Paragraph(escape(str(clave)), estilos["metadato_clave"]),
                    Paragraph(escape(str(valor)), estilos["metadato_valor"]),
                ]
                for clave, valor in metadatos
            ],
            colWidths=[2.05 * inch, 4.8 * inch],
        )
        tabla_metadatos.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), ServicioReportes._FONDO_TABLA),
                    ("TEXTCOLOR", (0, 0), (0, -1), ServicioReportes._TINTA),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#3C332C")),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D8C3A5")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        historia.extend(
            [
                Paragraph("Panorama del conjunto de datos", estilos["seccion"]),
                self._tarjetas_resumen(resumen),
                Spacer(1, 12),
                tabla_metadatos,
                Paragraph("Interpretación breve", estilos["seccion"]),
                Paragraph(self._interpretacion(resumen), estilos["cuerpo"]),
                PageBreak(),
                self._tabla_resumen(resumen, estilos),
                PageBreak(),
                Paragraph("Resumen gráfico", estilos["seccion"]),
                Paragraph(
                    "Promedio observado en cada dimensión Big Five.",
                    estilos["cuerpo"],
                ),
                Spacer(1, 8),
            ]
        )

        dimensiones = resumen.promedio_dimensiones
        historia.extend(
            [
                self._grafica_barras(
                    "Promedio por dimensión Big Five",
                    dimensiones.index.tolist(),
                    dimensiones.astype(float).tolist(),
                    self._VERDE,
                    "Promedio (1 a 5)",
                    maximo_eje=5,
                    ancho=680,
                    alto=200,
                    mostrar_valores=True,
                    etiqueta_eje_x="Dimensión de personalidad",
                ),
            ]
        )

        for rasgo in resumen.dimensiones_por_registro.columns:
            valores_rasgo = resumen.dimensiones_por_registro[rasgo]
            parametros = ServicioEstadisticas.calcular_parametros_intervalos(valores_rasgo)
            tabla_frecuencia = ServicioEstadisticas.calcular_frecuencia_intervalos(
                valores_rasgo,
                parametros=parametros,
            )
            historia.extend(
                [
                    PageBreak(),
                    Paragraph(
                        f"Distribución por intervalos: {escape(str(rasgo))}",
                        estilos["seccion"],
                    ),
                    Paragraph(
                        f"R = {parametros.rango:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                        f"K = {parametros.k} &nbsp;&nbsp;|&nbsp;&nbsp; "
                        f"A = {parametros.amplitud:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                        f"N = {parametros.cantidad_datos}",
                        estilos["cuerpo"],
                    ),
                    Spacer(1, 6),
                    self._tabla_frecuencias(tabla_frecuencia, estilos),
                    Spacer(1, 10),
                    self._histograma_con_poligono(tabla_frecuencia, str(rasgo)),
                ]
            )

        documento.build(historia, onFirstPage=self._encabezado_pie, onLaterPages=self._encabezado_pie)
        return salida.getvalue()

    def generar_reporte(
        self,
        resultados: ResumenEstadistico,
        ruta_destino: str,
        **opciones,
    ) -> Path:
        """Guarda una copia del reporte cuando se requiere conservarla en disco."""
        destino = Path(ruta_destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(self.generar_reporte_estadistico(resultados, **opciones))
        return destino

    def generar_reporte_entrenamiento(
        self,
        resultado: ResultadoEntrenamiento,
        nombre_dataset: str = "Dataset sin nombre",
        fecha_generacion: datetime | None = None,
    ) -> bytes:
        """Genera un PDF descargable con las gráficas, métricas e interpretación
        de un entrenamiento de K-Means (RF de descarga de reporte de resultados)."""
        if not isinstance(resultado, ResultadoEntrenamiento):
            raise TypeError("El reporte requiere un resultado de entrenamiento válido.")

        servicio_resultados = ServicioResultados()
        resumen_grupos = servicio_resultados.crear_resumen_grupos(resultado)
        proyeccion = servicio_resultados.crear_proyeccion_pca(resultado)
        centros = servicio_resultados.crear_tabla_centros(resultado)
        interpretaciones = servicio_resultados.crear_interpretaciones_grupos(
            resultado
        )
        perfiles = servicio_resultados.crear_resumen_perfiles_grupos(resultado)

        fecha = fecha_generacion or datetime.now()
        estilos = self._estilos()
        salida = BytesIO()
        documento = SimpleDocTemplate(
            salida,
            pagesize=landscape(letter),
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
            topMargin=0.62 * inch,
            bottomMargin=0.56 * inch,
            title="Reporte de resultados del entrenamiento - ClusterLab",
            author="ClusterLab",
        )

        historia = [
            Paragraph("INFORME ACADÉMICO / RESULTADOS DEL ENTRENAMIENTO", estilos["subtitulo"]),
            Paragraph("Resultados del entrenamiento K-Means", estilos["titulo"]),
            Paragraph(
                "Métricas, grupos y proyección obtenidos tras entrenar el modelo de agrupamiento",
                estilos["subtitulo"],
            ),
            HRFlowable(width="100%", thickness=1, color=self._AZUL, spaceAfter=12),
        ]

        etiqueta_k = f"{resultado.k_usado}"
        if resultado.k_usado == resultado.k_recomendado:
            etiqueta_k += " (recomendado)"
        else:
            etiqueta_k += f" (recomendado: {resultado.k_recomendado})"

        metadatos = [
            ["Dataset", escape(nombre_dataset)],
            ["Fecha de generación", fecha.strftime("%d/%m/%Y %H:%M")],
            ["Algoritmo", "K-Means"],
            ["Grupos utilizados (K)", etiqueta_k],
            ["Registros agrupados", str(len(resultado.asignaciones))],
            ["Cantidad de variables", str(len(resultado.columnas))],
        ]
        tabla_metadatos = Table(metadatos, colWidths=[1.55 * inch, 5.3 * inch])
        tabla_metadatos.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), ServicioReportes._FONDO_TABLA),
                    ("TEXTCOLOR", (0, 0), (0, -1), ServicioReportes._TINTA),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#3C332C")),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D8C3A5")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        historia.extend(
            [
                Paragraph("Resumen del entrenamiento", estilos["seccion"]),
                self._tarjetas_entrenamiento(resultado),
                Spacer(1, 12),
                tabla_metadatos,
                Paragraph("Interpretación de resultados", estilos["seccion"]),
                Paragraph(
                    self._interpretacion_entrenamiento(resultado, resumen_grupos),
                    estilos["cuerpo"],
                ),
                PageBreak(),
                Paragraph("Evaluación de valores de K", estilos["seccion"]),
                Paragraph(
                    "Comparación de inercia y Silhouette para cada valor de K evaluado "
                    "antes de entrenar el modelo final.",
                    estilos["cuerpo"],
                ),
                Spacer(1, 6),
                self._tabla_evaluacion_k(resultado, estilos),
                Spacer(1, 14),
                Paragraph("Resumen de los grupos", estilos["seccion"]),
                self._tabla_grupos(resumen_grupos, estilos),
                PageBreak(),
                Paragraph("Centros de los grupos (escala original)", estilos["seccion"]),
                Paragraph(
                    "Valor representativo de cada variable en el centro de cada grupo, "
                    "expresado en la escala original de los datos.",
                    estilos["cuerpo"],
                ),
                Spacer(1, 6),
                self._tabla_centros_entrenamiento(centros, estilos),
                PageBreak(),
                Paragraph("Perfil e interpretación por rasgo de los grupos", estilos["seccion"]),
                *self._parrafos_interpretaciones_grupos(interpretaciones, perfiles, estilos),
                PageBreak(),
                Paragraph("Gráficas del entrenamiento", estilos["seccion"]),
                Paragraph(
                    "Tamaño de cada grupo y proyección en dos dimensiones de los "
                    "registros agrupados.",
                    estilos["cuerpo"],
                ),
                Spacer(1, 8),
            ]
        )

        historia.append(
            self._grafica_barras(
                "Tamaño de los grupos",
                resumen_grupos["Grupo"].tolist(),
                resumen_grupos["Registros"].astype(float).tolist(),
                self._AZUL,
                "Registros",
                ancho=680,
                alto=200,
                mostrar_valores=True,
                etiqueta_eje_x="Grupo",
            )
        )
        historia.extend(
            [
                Spacer(1, 10),
                self._grafica_dispersion(proyeccion),
                Spacer(1, 4),
                Paragraph(
                    "Cada color representa un grupo distinto; la marca en forma de "
                    "cruz indica el centro de cada grupo. Esta proyección solo se usa "
                    "para visualizar y no participó en el entrenamiento del modelo.",
                    estilos["cuerpo"],
                ),
            ]
        )

        documento.build(historia, onFirstPage=self._encabezado_pie, onLaterPages=self._encabezado_pie)
        return salida.getvalue()
