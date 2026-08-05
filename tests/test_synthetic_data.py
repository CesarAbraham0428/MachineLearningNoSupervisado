"""Pruebas de los pasos 3 y 4 de generación de datos sintéticos."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


RUTA_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "aleatorios"
    / "02_generar_respuestas_likert.py"
)
ESPECIFICACION = importlib.util.spec_from_file_location(
    "generador_respuestas_likert",
    RUTA_SCRIPT,
)
GENERADOR = importlib.util.module_from_spec(ESPECIFICACION)
assert ESPECIFICACION.loader is not None
ESPECIFICACION.loader.exec_module(GENERADOR)


class PruebasGeneradorRespuestasLikert(unittest.TestCase):
    def setUp(self):
        self.originales = [
            [1, 2, 3, 4, 5] * 5,
            [5, 4, 3, 2, 1] * 5,
            [3, 3, 3, 3, 3] * 5,
        ]

    def test_genera_25_respuestas_enteras_en_el_rango_likert(self):
        perfiles = [[3.2, 2.8, 4.0, 4.2, 3.6]]

        resultado = GENERADOR.generar_respuestas_sinteticas(
            perfiles,
            self.originales,
        )

        self.assertEqual(len(resultado), 1)
        self.assertEqual(len(resultado[0]), 25)
        self.assertTrue(all(type(valor) is int for valor in resultado[0]))
        self.assertTrue(all(1 <= valor <= 5 for valor in resultado[0]))

    def test_aproxima_cada_rasgo_en_incrementos_de_una_quinta_parte(self):
        perfil = [1.7, 2.4, 3.1, 4.3, 4.9]

        fila = GENERADOR.generar_respuestas_sinteticas(
            [perfil],
            self.originales,
        )[0]

        promedios = [
            sum(fila[inicio:inicio + 5]) / 5
            for inicio in range(0, 25, 5)
        ]
        for observado, esperado in zip(promedios, perfil):
            self.assertLessEqual(abs(observado - esperado), 0.100001)

    def test_convierte_numeros_a_las_etiquetas_originales(self):
        fila = [1, 2, 3, 4, 5] * 5

        resultado = GENERADOR.convertir_respuestas_a_texto([fila])

        self.assertEqual(
            resultado[0][:5],
            [
                "Totalmente en desacuerdo",
                "En desacuerdo",
                "Neutral",
                "De acuerdo",
                "Totalmente de acuerdo",
            ],
        )

if __name__ == "__main__":
    unittest.main()
