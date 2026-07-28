"""Pruebas del resumen estadístico de RF-05."""

import unittest

import pandas as pd

from services.statistics_service import ServicioEstadisticas


def _crear_dataset(respuestas_por_registro: list[str]) -> pd.DataFrame:
    filas = []
    for indice, respuesta in enumerate(respuestas_por_registro, start=1):
        fila = {"Marca temporal": f"2026/07/18 {indice}:00:00 p.m. GMT-6"}
        fila.update(
            {
                f"{numero}. Pregunta de prueba {numero}": respuesta
                for numero in range(1, 26)
            }
        )
        filas.append(fila)
    return pd.DataFrame(filas)


class PruebasServicioEstadisticas(unittest.TestCase):
    def setUp(self):
        self.servicio = ServicioEstadisticas()

    def test_calcula_medidas_estadisticas_relevantes(self):
        resumen = self.servicio.calcular_resumen(
            _crear_dataset(["En desacuerdo", "De acuerdo"])
        )

        primera = resumen.estadisticas_por_pregunta.iloc[0]
        self.assertEqual(resumen.cantidad_registros, 2)
        self.assertEqual(resumen.cantidad_variables, 25)
        self.assertEqual(primera["Media"], 3.0)
        self.assertEqual(primera["Mediana"], 3.0)
        self.assertEqual(primera["Mínimo"], 2.0)
        self.assertEqual(primera["Máximo"], 4.0)

    def test_calcula_frecuencia_y_promedios_big_five(self):
        resumen = self.servicio.calcular_resumen(
            _crear_dataset(["Neutral", "Neutral", "Neutral"])
        )

        frecuencias = resumen.frecuencia_respuestas.set_index("Valor Likert")
        self.assertEqual(int(frecuencias.loc[3, "Frecuencia"]), 75)
        self.assertEqual(float(frecuencias.loc[3, "Porcentaje"]), 100.0)
        self.assertTrue((resumen.promedio_dimensiones == 3.0).all())
        self.assertEqual(int(resumen.faltantes_por_pregunta.sum()), 0)


if __name__ == "__main__":
    unittest.main()
