"""Pruebas del preprocesamiento común del cuestionario Big Five."""

import unittest

import pandas as pd

from services.dataset_service import (
    DIMENSIONES_BIG_FIVE,
    ErrorDatos,
    ServicioConjuntoDatos,
)
from utils.validators import validar_conjunto_datos


def _crear_dataset(respuesta: str = "De acuerdo") -> pd.DataFrame:
    fila = {"Marca temporal": "2026/07/18 1:47:02 p.m. GMT-6"}
    fila.update(
        {
            f"{numero}. Pregunta de prueba {numero}": respuesta
            for numero in range(1, 26)
        }
    )
    return pd.DataFrame([fila])


class PruebasServicioConjuntoDatos(unittest.TestCase):
    def setUp(self):
        self.servicio = ServicioConjuntoDatos()

    def test_convierte_likert_y_excluye_fecha(self):
        resultado = self.servicio.preprocesar(_crear_dataset())

        self.assertEqual(resultado.respuestas_numericas.shape, (1, 25))
        self.assertNotIn(
            "Marca temporal", resultado.respuestas_numericas.columns
        )
        self.assertTrue(
            (resultado.respuestas_numericas.iloc[0] == 4).all()
        )
        self.assertEqual(resultado.columna_temporal, "Marca temporal")

    def test_calcula_las_cinco_dimensiones(self):
        resultado = self.servicio.preprocesar(_crear_dataset())

        self.assertEqual(
            resultado.dimensiones.columns.tolist(),
            list(DIMENSIONES_BIG_FIVE),
        )
        self.assertTrue((resultado.dimensiones.iloc[0] == 4.0).all())

    def test_acepta_variaciones_de_espacios_mayusculas_y_acentos(self):
        datos = _crear_dataset("  TOTALMENTE EN DESACUERDO  ")
        resultado = self.servicio.preprocesar(datos)

        self.assertTrue(
            (resultado.respuestas_numericas.iloc[0] == 1).all()
        )

    def test_rechaza_respuesta_invalida(self):
        datos = _crear_dataset()
        datos.iloc[0, 1] = "A veces"

        with self.assertRaisesRegex(ErrorDatos, "Likert inválidas"):
            self.servicio.preprocesar(datos)
        self.assertFalse(validar_conjunto_datos(datos))

    def test_rechaza_respuesta_faltante(self):
        datos = _crear_dataset()
        datos.iloc[0, 1] = None

        with self.assertRaisesRegex(ErrorDatos, "incompletas"):
            self.servicio.preprocesar(datos)

    def test_rechaza_si_falta_una_pregunta(self):
        datos = _crear_dataset().drop(columns=["25. Pregunta de prueba 25"])

        with self.assertRaisesRegex(ErrorDatos, "faltan preguntas"):
            self.servicio.preprocesar(datos)


if __name__ == "__main__":
    unittest.main()
