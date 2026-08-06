"""Pruebas del almacenamiento persistente de modelos entrenados."""

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
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
            dataset_origen="personalidad.csv",
            mapeo_likert={"De acuerdo": 4},
            columnas_likert=["Extraversión"],
        )

        ruta = self.servicio.directorio_modelos / guardado.archivo
        self.assertTrue(ruta.is_file())
        self.assertEqual(guardado.fecha_creacion, self.fecha)
        self.assertEqual(guardado.fecha_modificacion, self.fecha)
        self.assertEqual(guardado.cantidad_registros, 8)
        self.assertEqual(guardado.cantidad_variables, 2)
        self.assertEqual(guardado.algoritmo, "K-Means")
        self.assertEqual(guardado.cantidad_grupos, self.resultado.k_usado)

        listado = self.servicio.listar_modelos()
        self.assertEqual(len(listado), 1)
        self.assertEqual(self.servicio.contar_modelos(), 1)

    def test_paquete_conserva_modelo_escalador_columnas_y_mapeo(self):
        guardado = self.servicio.guardar_modelo(
            self.resultado,
            nombre="Modelo reutilizable",
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
            "dataset_origen": "datos.csv",
        }
        primero = self.servicio.guardar_modelo(self.resultado, **argumentos)

        with self.assertRaisesRegex(ErrorModelo, "Ya existe"):
            self.servicio.guardar_modelo(self.resultado, **argumentos)

        archivos = list(self.servicio.directorio_modelos.glob("*.joblib"))
        self.assertEqual(archivos, [self.servicio.directorio_modelos / primero.archivo])
        self.assertEqual(self.servicio.contar_modelos(), 1)

    def test_rechaza_nombre_vacio(self):
        with self.assertRaisesRegex(ErrorModelo, "nombre"):
            self.servicio.guardar_modelo(
                self.resultado,
                nombre=" ",
            )

    def test_lista_solo_modelos_compatibles_con_las_columnas_disponibles(self):
        self.servicio.guardar_modelo(
            self.resultado,
            nombre="Modelo compatible",
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
            dataset_origen="personalidad.csv",
        )

        self.assertEqual(guardado.columnas, tuple(self.resultado.columnas))
        listado = self.servicio.listar_modelos()
        self.assertEqual(listado[0].columnas, tuple(self.resultado.columnas))

    def test_reentrenar_actualiza_artefacto_metadatos_y_fecha_modificacion(self):
        guardado = self.servicio.guardar_modelo(
            self.resultado,
            nombre="Modelo que cambia",
            dataset_origen="personalidad.csv",
        )
        archivo_anterior = self.servicio.directorio_modelos / guardado.archivo
        fecha_modificacion = datetime(2026, 8, 6, 18, 45, tzinfo=timezone.utc)
        self.servicio.reloj = lambda: fecha_modificacion

        nuevos_datos = pd.DataFrame(
            {
                "Extraversión": [1.0, 1.1, 1.2, 4.7, 4.8, 4.9],
                "Responsabilidad": [1.2, 1.0, 1.1, 4.8, 5.0, 4.9],
            }
        )
        artefacto_previo = self.servicio.cargar_modelo(guardado.id)
        resultado_nuevo = ServicioEntrenamiento(random_state=11).continuar_entrenamiento(
            nuevos_datos,
            tuple(artefacto_previo["columnas"]),
            artefacto_previo["modelo"],
            artefacto_previo["escalador"],
        )

        actualizado = self.servicio.actualizar_modelo_reentrenado(
            guardado.id, resultado_nuevo
        )

        self.assertEqual(actualizado.fecha_creacion, self.fecha)
        self.assertEqual(actualizado.fecha_modificacion, fecha_modificacion)
        self.assertEqual(actualizado.cantidad_registros, len(nuevos_datos))
        self.assertEqual(actualizado.silhouette, resultado_nuevo.silhouette)
        self.assertNotEqual(actualizado.archivo, guardado.archivo)
        self.assertFalse(archivo_anterior.exists())
        artefacto_actualizado = self.servicio.cargar_modelo(guardado.id)
        self.assertEqual(
            datetime.fromisoformat(artefacto_actualizado["fecha_modificacion"]),
            fecha_modificacion,
        )

    def test_migra_catalogo_existente_eliminando_columna_categoria(self):
        ruta_db = self.servicio.ruta_db.parent / "catalogo_legado.db"
        fecha = self.fecha.isoformat()
        with closing(sqlite3.connect(ruta_db)) as conexion:
            conexion.execute(
                """
                CREATE TABLE modelos_guardados (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    categoria TEXT NOT NULL,
                    fecha_creacion TEXT NOT NULL,
                    cantidad_registros INTEGER NOT NULL,
                    cantidad_variables INTEGER NOT NULL,
                    algoritmo TEXT NOT NULL,
                    cantidad_grupos INTEGER NOT NULL,
                    silhouette REAL NOT NULL,
                    dataset_origen TEXT NOT NULL,
                    archivo TEXT NOT NULL UNIQUE
                )
                """
            )
            conexion.execute(
                """
                INSERT INTO modelos_guardados (
                    nombre, categoria, fecha_creacion, cantidad_registros,
                    cantidad_variables, algoritmo, cantidad_grupos, silhouette,
                    dataset_origen, archivo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "Modelo legado",
                    "Categoría que se eliminará",
                    fecha,
                    8,
                    2,
                    "K-Means",
                    2,
                    0.75,
                    "legado.csv",
                    "legado.joblib",
                ),
            )
            conexion.commit()

        servicio_migrado = ServicioModelo(
            ruta_db=ruta_db,
            directorio_modelos=self.servicio.directorio_modelos / "legado",
            reloj=lambda: self.fecha,
        )

        with closing(sqlite3.connect(ruta_db)) as conexion:
            columnas = {
                fila[1]
                for fila in conexion.execute("PRAGMA table_info(modelos_guardados)")
            }
        self.assertNotIn("categoria", columnas)
        self.assertIn("fecha_modificacion", columnas)
        self.assertIn("columnas", columnas)
        listado = servicio_migrado.listar_modelos()
        self.assertEqual([modelo.nombre for modelo in listado], ["Modelo legado"])
        self.assertEqual(listado[0].fecha_modificacion, self.fecha)


if __name__ == "__main__":
    unittest.main()
