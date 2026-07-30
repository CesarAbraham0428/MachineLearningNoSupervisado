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


if __name__ == "__main__":
    unittest.main()
