"""Pruebas para seleccionar y reutilizar un modelo guardado (models_view)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from services.model_service import ServicioModelo
from services.training_service import ServicioEntrenamiento
from views.models_view import (
    _continuar_entrenamiento_en_dataset_activo,
    _predecir_clusters,
    _preparar_datos_para_prediccion,
    _tabla_centros_modelo,
)


def _entrenar_y_guardar(servicio_modelo: ServicioModelo):
    """Entrena un modelo simple y separable y lo guarda, como haría la app."""
    datos_entrenamiento = pd.DataFrame(
        {
            "Extraversión": [1.0, 1.2, 0.8, 1.1, 4.8, 5.0, 4.9, 4.7],
            "Responsabilidad": [1.1, 0.9, 1.0, 1.2, 4.9, 4.7, 5.0, 4.8],
        }
    )
    resultado = ServicioEntrenamiento(random_state=7).entrenar_modelo(
        datos_entrenamiento, maximo_k=3
    )
    guardado = servicio_modelo.guardar_modelo(
        resultado,
        nombre="Modelo de prueba",
        categoria="Pruebas",
        dataset_origen="dataset_prueba.csv",
    )
    return guardado, resultado


class PruebasReutilizarModeloGuardado(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self._directorio_temporal = tempfile.TemporaryDirectory()
        raiz = Path(self._directorio_temporal.name)
        self.servicio_modelo = ServicioModelo(
            ruta_db=raiz / "catalogo.db", directorio_modelos=raiz / "modelos"
        )
        self.guardado, self.resultado = _entrenar_y_guardar(self.servicio_modelo)
        self.artefacto = self.servicio_modelo.cargar_modelo(self.guardado.id)

    def tearDown(self):
        self._directorio_temporal.cleanup()

    def test_tabla_de_centros_reconstruye_la_escala_original(self):
        tabla = _tabla_centros_modelo(self.artefacto)

        self.assertEqual(len(tabla), self.resultado.k_usado)
        self.assertIn("Extraversión", tabla.columns)
        self.assertIn("Responsabilidad", tabla.columns)
        # Los centros reconstruidos deben coincidir con los originales.
        centros_originales = self.resultado.centroides_originales.round(3)
        pd.testing.assert_series_equal(
            tabla["Extraversión"].reset_index(drop=True),
            centros_originales["Extraversión"].reset_index(drop=True),
            check_names=False,
        )

    def test_predice_grupos_para_un_dataset_nuevo_compatible(self):
        dataset_nuevo = pd.DataFrame(
            {
                "Extraversión": [1.05, 4.95],
                "Responsabilidad": [1.0, 4.85],
                "Comentario": ["A", "B"],  # columna extra, debe ignorarse
            }
        )
        datos_listos, error = _preparar_datos_para_prediccion(
            dataset_nuevo, self.artefacto
        )
        self.assertEqual(error, "")
        self.assertIsNotNone(datos_listos)

        asignaciones = _predecir_clusters(datos_listos, self.artefacto)
        self.assertEqual(len(asignaciones), 2)
        # Los dos registros son claramente distintos y deben caer en grupos distintos.
        self.assertNotEqual(asignaciones.iloc[0], asignaciones.iloc[1])
        self.assertTrue(asignaciones.iloc[0].startswith("Grupo "))

    def test_rechaza_dataset_al_que_le_faltan_columnas(self):
        dataset_incompleto = pd.DataFrame({"Extraversión": [1.0, 4.5]})

        datos_listos, error = _preparar_datos_para_prediccion(
            dataset_incompleto, self.artefacto
        )
        self.assertIsNone(datos_listos)
        self.assertIn("Responsabilidad", error)

    def test_rechaza_dataset_con_valores_faltantes(self):
        dataset_con_nulos = pd.DataFrame(
            {
                "Extraversión": [1.0, None],
                "Responsabilidad": [1.1, 4.8],
            }
        )
        datos_listos, error = _preparar_datos_para_prediccion(
            dataset_con_nulos, self.artefacto
        )
        self.assertIsNone(datos_listos)
        self.assertIn("faltantes", error)



    def test_continuar_entrenamiento_usa_centros_y_escalador_guardados(self):
        datos = pd.DataFrame(
            {
                "Extraversión": [1.0, 1.1, 4.8, 4.9],
                "Responsabilidad": [1.1, 1.0, 4.9, 4.8],
            }
        )
        modelo_guardado = SimpleNamespace(nombre="Modelo de prueba")
        modelo_previo = object()
        escalador_previo = object()
        artefacto = {
            "columnas": ("Extraversión", "Responsabilidad"),
            "modelo": modelo_previo,
            "escalador": escalador_previo,
        }
        servicio = MagicMock()
        resultado = object()
        servicio.continuar_entrenamiento.return_value = resultado
        streamlit = MagicMock()
        streamlit.session_state = {}

        with (
            patch("views.models_view.ServicioEntrenamiento", return_value=servicio),
            patch("views.models_view.st", streamlit),
        ):
            _continuar_entrenamiento_en_dataset_activo(
                modelo_guardado, artefacto, datos
            )

        servicio.continuar_entrenamiento.assert_called_once_with(
            datos,
            ("Extraversión", "Responsabilidad"),
            modelo_previo,
            escalador_previo,
        )
        self.assertIs(streamlit.session_state["resultado_entrenamiento"], resultado)
        streamlit.rerun.assert_called_once()

if __name__ == "__main__":
    unittest.main()
