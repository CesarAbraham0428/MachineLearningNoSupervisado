"""Pruebas del resumen estadístico de perfiles Big Five."""

import unittest

import pandas as pd

from services.dataset_service import DIMENSIONES_BIG_FIVE
from services.statistics_service import ServicioEstadisticas


def _crear_perfiles() -> pd.DataFrame:
    return pd.DataFrame(
        [[2.0, 3.0, 4.0, 2.5, 4.5], [4.0, 3.5, 3.0, 4.5, 3.5]],
        columns=list(DIMENSIONES_BIG_FIVE),
    )


class PruebasServicioEstadisticas(unittest.TestCase):
    def setUp(self):
        self.servicio = ServicioEstadisticas()

    def test_calcula_medidas_sobre_cinco_rasgos(self):
        resumen = self.servicio.calcular_resumen(_crear_perfiles())
        extraversion = resumen.estadisticas_por_rasgo.loc["Extraversión"]

        self.assertEqual(resumen.cantidad_registros, 2)
        self.assertEqual(resumen.cantidad_variables, 5)
        self.assertEqual(extraversion["Media"], 3.0)
        self.assertEqual(extraversion["Mediana"], 3.0)
        self.assertEqual(extraversion["Mínimo"], 2.0)
        self.assertEqual(extraversion["Máximo"], 4.0)

    def test_calcula_promedios_y_faltantes(self):
        perfiles = pd.DataFrame(
            {columna: [3.0, 3.0, 3.0] for columna in DIMENSIONES_BIG_FIVE}
        )
        resumen = self.servicio.calcular_resumen(perfiles)

        self.assertTrue((resumen.promedio_dimensiones == 3.0).all())
        self.assertEqual(int(resumen.faltantes_por_rasgo.sum()), 0)


if __name__ == "__main__":
    unittest.main()
