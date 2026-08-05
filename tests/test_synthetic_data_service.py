"""Pruebas de generación normal sobre cinco rasgos Big Five."""

import unittest

import pandas as pd

from services.dataset_service import DIMENSIONES_BIG_FIVE
from services.synthetic_data_service import ErrorDatosSinteticos, generar_dataset_sintetico


def _crear_perfiles() -> pd.DataFrame:
    columnas = list(DIMENSIONES_BIG_FIVE)
    return pd.DataFrame(
        [
            [1.8, 2.4, 3.1, 4.0, 3.6],
            [2.2, 2.8, 3.5, 4.2, 3.8],
            [3.1, 3.2, 4.0, 3.7, 4.1],
            [4.0, 3.8, 4.4, 3.2, 4.5],
            [4.4, 4.1, 4.7, 2.9, 4.8],
        ],
        columns=columnas,
    )


class PruebasServicioDatosSinteticos(unittest.TestCase):
    def setUp(self):
        self.datos = _crear_perfiles()

    def test_combina_sinteticos_y_originales_con_solo_cinco_rasgos(self):
        resultado = generar_dataset_sintetico(self.datos, cantidad=6, semilla=15)

        self.assertEqual(resultado.datos_sinteticos.shape, (6, 5))
        self.assertEqual(resultado.datos_combinados.shape, (11, 5))
        self.assertEqual(resultado.datos_combinados.columns.tolist(), list(DIMENSIONES_BIG_FIVE))
        pd.testing.assert_frame_equal(
            resultado.datos_combinados.iloc[6:].reset_index(drop=True),
            self.datos.reset_index(drop=True),
        )

    def test_genera_valores_continuos_validos(self):
        resultado = generar_dataset_sintetico(self.datos, cantidad=200, semilla=23)

        self.assertTrue(
            resultado.datos_sinteticos.apply(lambda serie: serie.between(1, 5).all()).all()
        )
        self.assertTrue(
            all(
                pd.api.types.is_float_dtype(resultado.datos_sinteticos[columna])
                for columna in resultado.datos_sinteticos
            )
        )

    def test_es_reproducible_cuando_se_indica_semilla(self):
        primero = generar_dataset_sintetico(self.datos, cantidad=20, semilla=7)
        segundo = generar_dataset_sintetico(self.datos, cantidad=20, semilla=7)
        pd.testing.assert_frame_equal(primero.datos_combinados, segundo.datos_combinados)

    def test_reporta_media_y_desviacion_muestral_originales(self):
        resultado = generar_dataset_sintetico(self.datos, cantidad=10, semilla=11)
        pd.testing.assert_series_equal(resultado.medias, self.datos.mean())
        pd.testing.assert_series_equal(resultado.desviaciones, self.datos.std(ddof=1))

    def test_dimensiones_sin_dispersion_permanecen_constantes(self):
        constantes = pd.DataFrame(
            {columna: [3.0, 3.0, 3.0] for columna in DIMENSIONES_BIG_FIVE}
        )
        resultado = generar_dataset_sintetico(constantes, cantidad=12, semilla=3)
        self.assertTrue((resultado.datos_sinteticos == 3.0).all().all())

    def test_rechaza_cantidades_no_positivas(self):
        with self.assertRaisesRegex(ErrorDatosSinteticos, "mayor que cero"):
            generar_dataset_sintetico(self.datos, cantidad=0)

    def test_requiere_al_menos_dos_perfiles(self):
        with self.assertRaisesRegex(ErrorDatosSinteticos, "al menos dos"):
            generar_dataset_sintetico(self.datos.iloc[:1], cantidad=3)


if __name__ == "__main__":
    unittest.main()
