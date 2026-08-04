"""Pruebas de generación del PDF estadístico (RF-06)."""

import unittest

import pandas as pd

from services.report_service import ServicioReportes
from services.dataset_service import crear_perfiles_big_five
from services.statistics_service import ServicioEstadisticas


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


class PruebasServicioReportes(unittest.TestCase):
    def test_genera_un_pdf_conteniendo_el_resumen_estadistico(self):
        resumen = ServicioEstadisticas().calcular_resumen(
            crear_perfiles_big_five(_crear_dataset())
        )

        contenido = ServicioReportes().generar_reporte_estadistico(
            resumen, nombre_dataset="cuestionario_prueba.csv"
        )

        self.assertTrue(contenido.startswith(b"%PDF"))
        self.assertGreater(len(contenido), 5_000)


if __name__ == "__main__":
    unittest.main()
