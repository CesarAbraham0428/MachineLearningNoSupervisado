"""Generación del reporte estadístico inicial en PDF (RF-06)."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
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
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.shapes import Drawing, String

from services.statistics_service import ResumenEstadistico


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
    def _tarjetas_resumen(cls, resumen: ResumenEstadistico) -> Table:
        """Presenta los indicadores clave como una cuadrícula compacta."""
        indicadores = [
            ("REGISTROS", str(resumen.cantidad_registros), cls._AZUL),
            ("VARIABLES", str(resumen.cantidad_variables), cls._VIOLETA),
            ("RESPUESTAS", str(resumen.cantidad_registros * resumen.cantidad_variables), cls._VERDE),
            ("FALTANTES", str(int(resumen.faltantes_por_pregunta.sum())), cls._AMBAR),
        ]
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
        return Table([celdas], colWidths=[1.72 * inch] * 4, hAlign="LEFT")

    @staticmethod
    def _interpretacion(resumen: ResumenEstadistico) -> str:
        """Produce una interpretación breve, descriptiva y no clínica."""
        frecuencias = resumen.frecuencia_respuestas
        respuesta_predominante = frecuencias.loc[frecuencias["Frecuencia"].idxmax(), "Respuesta"]
        dimension_mayor = resumen.promedio_dimensiones.idxmax()
        valor_mayor = resumen.promedio_dimensiones.max()
        dimension_menor = resumen.promedio_dimensiones.idxmin()
        valor_menor = resumen.promedio_dimensiones.min()
        faltantes = int(resumen.faltantes_por_pregunta.sum())

        definiciones = {
            "Extraversión": "la sociabilidad, la energía en interacciones y la iniciativa para relacionarse con otras personas",
            "Estabilidad emocional": "la calma, la regulación emocional y la recuperación ante situaciones de presión",
            "Apertura": "la curiosidad, el interés por aprender y la disposición hacia ideas o experiencias nuevas",
            "Responsabilidad": "la planificación, el orden y el cumplimiento de tareas o compromisos",
            "Amabilidad": "la cooperación, la empatía y la consideración hacia otras personas",
        }
        texto = (
            f"El análisis reúne {resumen.cantidad_registros} registros y "
            f"{resumen.cantidad_variables} variables del cuestionario. La respuesta "
            f"más frecuente fue <b>{escape(str(respuesta_predominante))}</b>. "
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
        encabezados = ["Pregunta", "Media", "Mediana", "Moda", "Desv. est.", "Mín.", "Máx."]
        filas = [
            [Paragraph(encabezado, estilos["tabla_encabezado"]) for encabezado in encabezados]
        ]
        for pregunta, valores in resumen.estadisticas_por_pregunta.iterrows():
            filas.append(
                [
                    Paragraph(escape(str(pregunta)), estilos["tabla"]),
                    f"{valores['Media']:.2f}",
                    f"{valores['Mediana']:.2f}",
                    f"{valores['Moda']:.2f}",
                    f"{valores['Desviación estándar']:.2f}",
                    f"{valores['Mínimo']:.0f}",
                    f"{valores['Máximo']:.0f}",
                ]
            )

        tabla = Table(
            filas,
            colWidths=[4.05 * inch, 0.62 * inch, 0.65 * inch, 0.55 * inch, 0.75 * inch, 0.5 * inch, 0.5 * inch],
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
            ["Dataset", escape(nombre_dataset)],
            ["Fecha de generación", fecha.strftime("%d/%m/%Y %H:%M")],
            ["Registros analizados", str(resumen.cantidad_registros)],
            ["Variables analizadas", str(resumen.cantidad_variables)],
            ["Respuestas evaluadas", str(resumen.cantidad_registros * resumen.cantidad_variables)],
            ["Datos faltantes", str(int(resumen.faltantes_por_pregunta.sum()))],
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
                    "Distribución global de respuestas y promedio observado en cada dimensión.",
                    estilos["cuerpo"],
                ),
                Spacer(1, 8),
            ]
        )

        frecuencia = resumen.frecuencia_respuestas
        dimensiones = resumen.promedio_dimensiones
        historia.append(
            self._grafica_pastel(
                "Distribución general de respuestas",
                ["T. desacuerdo", "Desacuerdo", "Neutral", "De acuerdo", "T. acuerdo"],
                frecuencia["Frecuencia"].astype(float).tolist(),
                ancho=680,
                alto=200,
            )
        )
        historia.extend(
            [
                Spacer(1, 8),
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
