"""Pruebas de los filtros mostrados en la vista de datos."""

import unittest
from datetime import date

import pandas as pd

from views.data_view import (
    _convertir_fechas,
    _detectar_columnas_temporales,
    _etiqueta_columna,
    _obtener_subconjunto,
    _preparar_perfiles_archivo,
)


class PruebasFiltrosVistaDatos(unittest.TestCase):
    def setUp(self):
        self.datos = pd.DataFrame(
            {
                "Marca temporal": [
                    "2026/07/18 1:47:02 p.m. GMT-6",
                    "2026/07/18 8:10:00 a.m. GMT-6",
                    "2026/07/19 3:30:00 p.m. GMT-6",
                    "2026/07/20 9:00:00 a.m. GMT-6",
                ],
                "Respuesta": ["A", "B", "C", "D"],
            }
        )

    def test_convierte_marcas_temporales_en_espanol(self):
        fechas = _convertir_fechas(self.datos["Marca temporal"])

        self.assertTrue(fechas.notna().all())
        self.assertEqual(fechas.iloc[0].hour, 13)
        self.assertEqual(fechas.iloc[1].hour, 8)

    def test_detecta_columna_temporal_y_muestra_nombre_amigable(self):
        self.assertEqual(
            _detectar_columnas_temporales(self.datos),
            ["Marca temporal"],
        )
        self.assertEqual(
            _etiqueta_columna("Marca temporal"),
            "Fecha y hora de respuesta",
        )

    def test_filtra_un_rango_inclusivo_de_fechas(self):
        resultado = _obtener_subconjunto(
            self.datos,
            list(self.datos.columns),
            "Marca temporal",
            (date(2026, 7, 18), date(2026, 7, 19)),
        )

        self.assertEqual(resultado["Respuesta"].tolist(), ["A", "B", "C"])



    def test_acepta_subconjuntos_big_five_ya_exportados(self):
        datos = pd.DataFrame(
            {
                "Estabilidad emocional": [2.8, 3.4],
                "Apertura a la experiencia": [4.0, 3.7],
                "Responsabilidad": [3.6, 4.2],
            }
        )

        perfiles, perfiles_con_contexto, ya_calculados = _preparar_perfiles_archivo(
            datos
        )

        self.assertTrue(ya_calculados)
        pd.testing.assert_frame_equal(perfiles, datos)
        pd.testing.assert_frame_equal(perfiles_con_contexto, datos)
if __name__ == "__main__":
    unittest.main()
