"""Pruebas de la generación sintética desde un DataFrame cargado."""

from __future__ import annotations

import unittest

import pandas as pd

from services.dataset_service import obtener_valor_likert
from services.synthetic_data_service import (
    ErrorDatosSinteticos,
    generar_dataset_sintetico,
)


RESPUESTAS = [
    "Totalmente en desacuerdo",
    "En desacuerdo",
    "Neutral",
    "De acuerdo",
    "Totalmente de acuerdo",
]


def _crear_dataset() -> pd.DataFrame:
    filas = []
    for indice, desplazamiento in enumerate((0, 1, 2, 3), start=1):
        fila: dict[str, object] = {
            "Marca temporal": f"2026/07/{17 + indice:02d} 10:00:00 a.m. GMT-6",
            "Grupo": "A" if indice % 2 else "B",
        }
        fila.update(
            {
                f"{numero}. Pregunta de prueba {numero}": RESPUESTAS[
                    (numero + desplazamiento) % len(RESPUESTAS)
                ]
                for numero in range(1, 26)
            }
        )
        filas.append(fila)
    return pd.DataFrame(filas)


class PruebasServicioDatosSinteticos(unittest.TestCase):
    def setUp(self):
        self.datos = _crear_dataset()
        self.columnas_preguntas = [
            f"{numero}. Pregunta de prueba {numero}"
            for numero in range(1, 26)
        ]

    def test_combina_sinteticos_antes_de_los_originales_sin_columna_origen(self):
        resultado = generar_dataset_sintetico(self.datos, cantidad=6, semilla=15)

        self.assertEqual(len(resultado.datos_sinteticos), 6)
        self.assertEqual(len(resultado.datos_combinados), len(self.datos) + 6)
        self.assertEqual(
            resultado.datos_combinados.columns.tolist(), self.datos.columns.tolist()
        )
        self.assertNotIn("Origen", resultado.datos_combinados.columns)
        pd.testing.assert_frame_equal(
            resultado.datos_combinados.iloc[6:].reset_index(drop=True),
            self.datos.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            resultado.datos_combinados.iloc[:6].reset_index(drop=True),
            resultado.datos_sinteticos.reset_index(drop=True),
        )

    def test_genera_respuestas_likert_validas_y_conserva_columnas_auxiliares(self):
        resultado = generar_dataset_sintetico(self.datos, cantidad=9, semilla=23)

        for columna in self.columnas_preguntas:
            valores = resultado.datos_sinteticos[columna].map(obtener_valor_likert)
            self.assertTrue(valores.between(1, 5).all())

        self.assertTrue(
            set(resultado.datos_sinteticos["Grupo"]).issubset(
                set(self.datos["Grupo"])
            )
        )
        self.assertTrue(
            set(resultado.datos_sinteticos["Marca temporal"]).issubset(
                set(self.datos["Marca temporal"])
            )
        )

    def test_es_reproducible_cuando_se_indica_semilla(self):
        primero = generar_dataset_sintetico(self.datos, cantidad=5, semilla=7)
        segundo = generar_dataset_sintetico(self.datos, cantidad=5, semilla=7)

        pd.testing.assert_frame_equal(
            primero.datos_combinados,
            segundo.datos_combinados,
        )

    def test_conserva_las_respuestas_numericas_si_el_original_esta_codificado(self):
        datos_numericos = self.datos.copy()
        conversion = {etiqueta: indice for indice, etiqueta in enumerate(RESPUESTAS, 1)}
        for columna in self.columnas_preguntas:
            datos_numericos[columna] = datos_numericos[columna].map(conversion)

        resultado = generar_dataset_sintetico(datos_numericos, cantidad=7, semilla=11)

        for columna in self.columnas_preguntas:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(resultado.datos_sinteticos[columna])
            )

    def test_rechaza_cantidades_no_positivas(self):
        with self.assertRaisesRegex(ErrorDatosSinteticos, "mayor que cero"):
            generar_dataset_sintetico(self.datos, cantidad=0)


if __name__ == "__main__":
    unittest.main()
