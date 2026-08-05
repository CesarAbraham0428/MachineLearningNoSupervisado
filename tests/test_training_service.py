"""Pruebas unitarias del flujo de entrenamiento K-Means."""

import unittest

import pandas as pd

from services.training_service import ErrorEntrenamiento, ServicioEntrenamiento


def _datos_de_prueba() -> pd.DataFrame:
    """Crea dos grupos claramente separados para una prueba reproducible."""
    return pd.DataFrame(
        {
            "Extraversión": [1.0, 1.2, 0.8, 1.1, 4.8, 5.0, 4.9, 4.7],
            "Responsabilidad": [1.1, 0.9, 1.0, 1.2, 4.9, 4.7, 5.0, 4.8],
        },
        index=[f"persona_{numero}" for numero in range(1, 9)],
    )


class PruebasServicioEntrenamiento(unittest.TestCase):
    def setUp(self):
        self.servicio = ServicioEntrenamiento(random_state=7)

    def test_estandariza_y_conserva_nombres(self):
        datos = _datos_de_prueba()

        estandarizados, _ = self.servicio.estandarizar(datos)

        self.assertEqual(estandarizados.columns.tolist(), datos.columns.tolist())
        self.assertEqual(estandarizados.index.tolist(), datos.index.tolist())
        self.assertAlmostEqual(float(estandarizados.mean().iloc[0]), 0.0)

    def test_evalua_k_y_recomienda_por_silhouette(self):
        evaluaciones = self.servicio.evaluar_k(_datos_de_prueba(), maximo_k=3)

        self.assertEqual([evaluacion.k for evaluacion in evaluaciones], [2, 3])
        recomendado = self.servicio.recomendar_k(evaluaciones)
        self.assertIn(recomendado, [2, 3])
        self.assertGreaterEqual(
            max(evaluacion.silhouette for evaluacion in evaluaciones), -1.0
        )

    def test_limita_candidatos_k_segun_tamano_de_muestra(self):
        self.assertEqual(
            list(self.servicio._candidatos_k(cantidad_registros=10, maximo_k=8)),
            [2, 3],
        )
        self.assertEqual(
            list(self.servicio._candidatos_k(cantidad_registros=51, maximo_k=8)),
            [2, 3, 4, 5, 6, 7],
        )

    def test_entrena_y_devuelve_centroides_en_escala_original(self):
        resultado = self.servicio.entrenar_modelo(_datos_de_prueba(), maximo_k=3)

        self.assertEqual(len(resultado.asignaciones), 8)
        self.assertEqual(resultado.centroides_originales.shape[1], 2)
        self.assertEqual(resultado.tamanos_clusters.sum(), 8)
        self.assertEqual(resultado.k_usado, resultado.k_recomendado)
        self.assertGreater(resultado.silhouette, 0.0)

    def test_rechaza_columnas_de_texto(self):
        datos = _datos_de_prueba().assign(Categoría="A")

        with self.assertRaisesRegex(ErrorEntrenamiento, "numéricas"):
            self.servicio.entrenar_modelo(datos)

    def test_reutilizar_modelo_asigna_sin_reentrenar(self):
        resultado = self.servicio.entrenar_modelo(_datos_de_prueba(), maximo_k=3)

        datos_nuevos = pd.DataFrame(
            {
                "Extraversión": [1.05, 0.95, 4.85, 4.95],
                "Responsabilidad": [1.0, 1.05, 4.85, 4.9],
            }
        )
        reutilizado = self.servicio.reutilizar_modelo(
            datos_nuevos,
            resultado.columnas,
            resultado.modelo,
            resultado.escalador,
        )

        self.assertIs(reutilizado.modelo, resultado.modelo)
        self.assertIs(reutilizado.escalador, resultado.escalador)
        self.assertEqual(len(reutilizado.asignaciones), 4)
        self.assertEqual(reutilizado.k_usado, resultado.k_usado)
        self.assertTrue(
            (
                reutilizado.centroides_estandarizados.values
                == resultado.centroides_estandarizados.values
            ).all()
        )

    def test_reutilizar_modelo_rechaza_columnas_faltantes(self):
        resultado = self.servicio.entrenar_modelo(_datos_de_prueba(), maximo_k=3)
        datos_incompatibles = pd.DataFrame({"Otra columna": [1, 2, 3, 4]})

        with self.assertRaisesRegex(ErrorEntrenamiento, "variables requeridas"):
            self.servicio.reutilizar_modelo(
                datos_incompatibles,
                resultado.columnas,
                resultado.modelo,
                resultado.escalador,
            )

    def test_continuar_entrenamiento_parte_de_los_centros_previos(self):
        resultado = self.servicio.entrenar_modelo(_datos_de_prueba(), maximo_k=3)

        datos_nuevos = pd.DataFrame(
            {
                "Extraversión": [1.1, 0.9, 1.0, 4.9, 5.1, 4.8],
                "Responsabilidad": [1.0, 1.1, 0.95, 4.95, 4.85, 5.0],
            }
        )
        continuado = self.servicio.continuar_entrenamiento(
            datos_nuevos,
            resultado.columnas,
            resultado.modelo,
            resultado.escalador,
        )

        self.assertIsNot(continuado.modelo, resultado.modelo)
        self.assertIs(continuado.escalador, resultado.escalador)
        self.assertEqual(len(continuado.asignaciones), 6)
        self.assertEqual(continuado.k_usado, resultado.k_usado)
        self.assertGreater(continuado.silhouette, 0.0)

    def test_continuar_entrenamiento_requiere_mas_registros_que_grupos(self):
        resultado = self.servicio.entrenar_modelo(
            _datos_de_prueba(), k=3, maximo_k=3
        )
        datos_insuficientes = pd.DataFrame(
            {
                "Extraversión": [1.0, 1.1, 4.9],
                "Responsabilidad": [1.1, 1.0, 4.8],
            }
        )

        with self.assertRaisesRegex(ErrorEntrenamiento, "más registros"):
            self.servicio.continuar_entrenamiento(
                datos_insuficientes,
                resultado.columnas,
                resultado.modelo,
                resultado.escalador,
            )


if __name__ == "__main__":
    unittest.main()