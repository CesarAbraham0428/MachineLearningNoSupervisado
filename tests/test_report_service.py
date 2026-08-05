"""Pruebas de generación de reportes PDF (RF-06 y reporte de resultados)."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import pandas as pd
from reportlab.graphics.shapes import Rect, String
from reportlab.platypus import Paragraph

from services.report_service import ServicioReportes
from services.dataset_service import crear_perfiles_big_five
from services.statistics_service import ServicioEstadisticas
from services.training_service import ServicioEntrenamiento


def _crear_dataset() -> pd.DataFrame:
    respuestas = ["En desacuerdo", "Neutral", "De acuerdo"]
    filas = []
    for indice, respuesta in enumerate(respuestas, start=1):
        fila = {"Marca temporal": f"2026/07/18 {indice}:00:00 p.m. GMT-6"}
        fila.update(
            {
                f"{numero}. Pregunta de prueba {numero}": respuesta
                for numero in range(1, 26)
            }
        )
        filas.append(fila)
    return pd.DataFrame(filas)


def _crear_dataset_entrenamiento() -> pd.DataFrame:
    """Crea dos grupos claramente separados para un entrenamiento reproducible."""
    return pd.DataFrame(
        {
            "Extraversión": [1.0, 1.2, 0.8, 1.1, 4.8, 5.0, 4.9, 4.7],
            "Responsabilidad": [1.1, 0.9, 1.0, 1.2, 4.9, 4.7, 5.0, 4.8],
        },
        index=[f"persona_{numero}" for numero in range(1, 9)],
    )


class PruebasServicioReportes(unittest.TestCase):
    def test_genera_un_pdf_conteniendo_el_resumen_estadistico(self):
        resumen = ServicioEstadisticas().calcular_resumen(
            crear_perfiles_big_five(_crear_dataset())
        )

        generar_histograma = ServicioReportes._histograma_con_poligono
        with patch.object(
            ServicioReportes,
            "_histograma_con_poligono",
            wraps=generar_histograma,
        ) as histograma:
            contenido = ServicioReportes().generar_reporte_estadistico(
                resumen, nombre_dataset="cuestionario_prueba.csv"
            )

        self.assertTrue(contenido.startswith(b"%PDF"))
        self.assertGreater(len(contenido), 5_000)
        self.assertEqual(
            histograma.call_count,
            resumen.cantidad_variables,
        )

    def test_tabla_estadistica_del_pdf_incluye_dispersion_y_cv(self):
        resumen = ServicioEstadisticas().calcular_resumen(
            crear_perfiles_big_five(_crear_dataset())
        )

        tabla = ServicioReportes._tabla_resumen(
            resumen,
            ServicioReportes._estilos(),
        )
        encabezados = [celda.text for celda in tabla._cellvalues[0]]

        self.assertIn("Varianza", encabezados)
        self.assertIn("Desv. est.", encabezados)
        self.assertIn("CV (%)", encabezados)

    def test_histograma_pdf_usa_limites_de_intervalo_y_ejes_de_la_vista(self):
        tabla = ServicioEstadisticas.calcular_frecuencia_intervalos(
            pd.Series([2.0, 2.1, 2.8, 3.4, 4.0]),
            numero_intervalos=2,
        )

        dibujo = ServicioReportes._histograma_con_poligono(tabla, "Extraversión")
        barras = [elemento for elemento in dibujo.contents if isinstance(elemento, Rect)]
        textos = [elemento.text for elemento in dibujo.contents if isinstance(elemento, String)]

        self.assertEqual(len(barras), 2)
        self.assertAlmostEqual(barras[0].x + barras[0].width, barras[1].x)
        self.assertIn("Distribución de Extraversión", textos)
        self.assertIn("Puntaje del rasgo extraversión", textos)
        self.assertIn("2.00", textos)
        self.assertIn("3.00", textos)
        self.assertIn("4.00", textos)

    def test_genera_un_pdf_con_los_resultados_del_entrenamiento(self):
        resultado = ServicioEntrenamiento(random_state=7).entrenar_modelo(
            _crear_dataset_entrenamiento(), maximo_k=3
        )

        crear_parrafos = ServicioReportes._parrafos_interpretaciones_grupos
        with patch.object(
            ServicioReportes,
            "_parrafos_interpretaciones_grupos",
            wraps=crear_parrafos,
        ) as interpretaciones_pdf:
            contenido = ServicioReportes().generar_reporte_entrenamiento(
                resultado, nombre_dataset="cuestionario_prueba.csv"
            )

        self.assertTrue(contenido.startswith(b"%PDF"))
        self.assertGreater(len(contenido), 5_000)
        self.assertEqual(interpretaciones_pdf.call_count, 1)

    def test_reporte_de_entrenamiento_rechaza_tipos_invalidos(self):
        with self.assertRaises(TypeError):
            ServicioReportes().generar_reporte_entrenamiento(object())



    def test_reporte_reentrenado_explica_cuando_no_hay_evaluacion_de_k(self):
        resultado = ServicioEntrenamiento(random_state=7).entrenar_modelo(
            _crear_dataset_entrenamiento(), maximo_k=3
        )
        reentrenado = replace(resultado, evaluaciones=())

        contenido_k = ServicioReportes._tabla_evaluacion_k(
            reentrenado,
            ServicioReportes._estilos(),
        )

        self.assertIsInstance(contenido_k, Paragraph)
        self.assertIn("No se evaluaron valores adicionales de K", contenido_k.text)
        self.assertIn("K =", contenido_k.text)

if __name__ == "__main__":
    unittest.main()
