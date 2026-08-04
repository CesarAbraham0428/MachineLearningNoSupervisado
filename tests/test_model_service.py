"""Pruebas del almacenamiento persistente de modelos entrenados."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from services.model_service import ErrorModelo, ServicioModelo
from services.training_service import ServicioEntrenamiento


def _resultado_de_prueba():
    datos = pd.DataFrame(
        {
            "Extraversión": [1.0, 1.2, 0.8, 1.1, 4.8, 5.0, 4.9, 4.7],
            "Responsabilidad": [1.1, 0.9, 1.0, 1.2, 4.9, 4.7, 5.0, 4.8],
        }
    )
    return ServicioEntrenamiento(random_state=7).entrenar_modelo(
        datos,
        maximo_k=3,
    )


class PruebasServicioModelo(unittest.TestCase):
    def setUp(self):
        self.temporal = TemporaryDirectory()
        raiz = Path(self.temporal.name)
        self.fecha = datetime(2026, 7, 30, 16, 30, tzinfo=timezone.utc)
        self.servicio = ServicioModelo(
            ruta_db=raiz / "catalogo.db",
            directorio_modelos=raiz / "modelos",
            reloj=lambda: self.fecha,
        )
        self.resultado = _resultado_de_prueba()

    def tearDown(self):
        self.temporal.cleanup()

    def test_guarda_archivo_y_metadatos_del_entrenamiento(self):
        guardado = self.servicio.guardar_modelo(
            self.resultado,
            nombre="Personalidad Big Five",
            categoria="Cuestionario académico",
            dataset_origen="personalidad.csv",
            mapeo_likert={"De acuerdo": 4},
            columnas_likert=["Extraversión"],
        )

        ruta = self.servicio.directorio_modelos / guardado.archivo
        self.assertTrue(ruta.is_file())
        self.assertEqual(guardado.fecha_creacion, self.fecha)
        self.assertEqual(guardado.cantidad_registros, 8)
        self.assertEqual(guardado.cantidad_variables, 2)
        self.assertEqual(guardado.algoritmo, "K-Means")
        self.assertEqual(guardado.cantidad_grupos, self.resultado.k_usado)

        listado = self.servicio.listar_modelos()
        self.assertEqual(len(listado), 1)
        self.assertEqual(listado[0].categoria, "Cuestionario académico")
        self.assertEqual(self.servicio.contar_modelos(), 1)

    def test_paquete_conserva_modelo_escalador_columnas_y_mapeo(self):
        guardado = self.servicio.guardar_modelo(
            self.resultado,
            nombre="Modelo reutilizable",
            categoria="Personalidad",
            dataset_origen="personalidad.csv",
            mapeo_likert={"Neutral": 3},
            columnas_likert=["Extraversión", "Responsabilidad"],
        )

        artefacto = self.servicio.cargar_modelo(guardado.id)

        self.assertEqual(artefacto["algoritmo"], "K-Means")
        self.assertEqual(artefacto["columnas"], tuple(self.resultado.columnas))
        self.assertEqual(artefacto["mapeo_likert"], {"Neutral": 3})
        self.assertEqual(
            artefacto["columnas_likert"],
            ("Extraversión", "Responsabilidad"),
        )
        self.assertTrue(hasattr(artefacto["modelo"], "predict"))
        self.assertTrue(hasattr(artefacto["escalador"], "transform"))

    def test_no_sobrescribe_un_modelo_con_el_mismo_nombre(self):
        argumentos = {
            "nombre": "Modelo repetido",
            "categoria": "Prueba",
            "dataset_origen": "datos.csv",
        }
        primero = self.servicio.guardar_modelo(self.resultado, **argumentos)

        with self.assertRaisesRegex(ErrorModelo, "Ya existe"):
            self.servicio.guardar_modelo(self.resultado, **argumentos)

        archivos = list(self.servicio.directorio_modelos.glob("*.joblib"))
        self.assertEqual(archivos, [self.servicio.directorio_modelos / primero.archivo])
        self.assertEqual(self.servicio.contar_modelos(), 1)

    def test_rechaza_nombre_o_categoria_vacios(self):
        with self.assertRaisesRegex(ErrorModelo, "nombre"):
            self.servicio.guardar_modelo(
                self.resultado,
                nombre=" ",
                categoria="Personalidad",
            )
        with self.assertRaisesRegex(ErrorModelo, "categoría"):
            self.servicio.guardar_modelo(
                self.resultado,
                nombre="Modelo",
                categoria=" ",
            )

    def test_lista_solo_modelos_compatibles_con_las_columnas_disponibles(self):
        self.servicio.guardar_modelo(
            self.resultado,
            nombre="Modelo compatible",
            categoria="Personalidad",
            dataset_origen="personalidad.csv",
        )

        compatibles = self.servicio.listar_modelos_compatibles(
            ["Extraversión", "Responsabilidad", "Otra columna extra"]
        )
        self.assertEqual([modelo.nombre for modelo in compatibles], ["Modelo compatible"])

        incompatibles = self.servicio.listar_modelos_compatibles(["Solo esta columna"])
        self.assertEqual(incompatibles, [])

    def test_columnas_del_modelo_quedan_disponibles_en_el_catalogo(self):
        guardado = self.servicio.guardar_modelo(
            self.resultado,
            nombre="Modelo con columnas",
            categoria="Personalidad",
            dataset_origen="personalidad.csv",
        )

        self.assertEqual(guardado.columnas, tuple(self.resultado.columnas))
        listado = self.servicio.listar_modelos()
        self.assertEqual(listado[0].columnas, tuple(self.resultado.columnas))


if __name__ == "__main__":
    unittest.main()