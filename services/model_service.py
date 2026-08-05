"""Persistencia de modelos entrenados y sus metadatos."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import sqlite3
from typing import Callable, Iterable
from uuid import uuid4

import joblib

from database.database import BaseDeDatos
from services.training_service import ResultadoEntrenamiento


_RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
_DIRECTORIO_MODELOS_PREDETERMINADO = _RAIZ_PROYECTO / "storage" / "models"
_RUTA_DB_PREDETERMINADA = (
    _DIRECTORIO_MODELOS_PREDETERMINADO / "catalogo_modelos.db"
)


class ErrorModelo(ValueError):
    """Indica que un modelo no pudo guardarse o consultarse."""


@dataclass(frozen=True)
class ModeloGuardado:
    """Metadatos visibles de un modelo almacenado."""

    id: int
    nombre: str
    categoria: str
    fecha_creacion: datetime
    cantidad_registros: int
    cantidad_variables: int
    algoritmo: str
    cantidad_grupos: int
    silhouette: float
    dataset_origen: str
    archivo: str
    columnas: tuple[str, ...] = field(default_factory=tuple)


class ServicioModelo:
    """Guarda artefactos de entrenamiento y mantiene un catálogo en SQLite."""

    def __init__(
        self,
        ruta_db: str | Path = _RUTA_DB_PREDETERMINADA,
        directorio_modelos: str | Path = _DIRECTORIO_MODELOS_PREDETERMINADO,
        reloj: Callable[[], datetime] | None = None,
    ):
        self.ruta_db = Path(ruta_db)
        self.directorio_modelos = Path(directorio_modelos)
        self.reloj = reloj or (lambda: datetime.now().astimezone())
        self.ruta_db.parent.mkdir(parents=True, exist_ok=True)
        self.directorio_modelos.mkdir(parents=True, exist_ok=True)
        self.base_datos = BaseDeDatos(str(self.ruta_db))
        self._crear_tabla()

    @contextmanager
    def _conexion(self):
        """Cierra siempre SQLite para no bloquear el archivo en Windows."""
        conexion = self.base_datos.obtener_conexion()
        try:
            yield conexion
            conexion.commit()
        except Exception:
            conexion.rollback()
            raise
        finally:
            conexion.close()

    def _crear_tabla(self) -> None:
        """Crea el catálogo sin alterar registros existentes."""
        try:
            with self._conexion() as conexion:
                conexion.execute(
                    """
                    CREATE TABLE IF NOT EXISTS modelos_guardados (
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
                columnas_existentes = {
                    str(fila[1])
                    for fila in conexion.execute("PRAGMA table_info(modelos_guardados)")
                }
                if "columnas" not in columnas_existentes:
                    # Migración: los modelos guardados antes de esta versión no
                    # registran sus variables; quedan con '[]' y no se ofrecen
                    # como compatibles hasta que se vuelvan a guardar.
                    conexion.execute(
                        "ALTER TABLE modelos_guardados ADD COLUMN columnas TEXT NOT NULL DEFAULT '[]'"
                    )
        except sqlite3.Error as error:
            raise ErrorModelo(
                "No fue posible preparar el catálogo de modelos guardados."
            ) from error

    @staticmethod
    def _validar_texto(valor: object, etiqueta: str, limite: int = 100) -> str:
        texto = str(valor or "").strip()
        if not texto:
            raise ErrorModelo(f"{etiqueta} es obligatorio.")
        if len(texto) > limite:
            raise ErrorModelo(f"{etiqueta} no puede superar {limite} caracteres.")
        return texto

    @staticmethod
    def _desde_fila(fila: sqlite3.Row) -> ModeloGuardado:
        try:
            columnas = tuple(json.loads(fila["columnas"]))
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            columnas = ()
        return ModeloGuardado(
            id=int(fila["id"]),
            nombre=str(fila["nombre"]),
            categoria=str(fila["categoria"]),
            fecha_creacion=datetime.fromisoformat(str(fila["fecha_creacion"])),
            cantidad_registros=int(fila["cantidad_registros"]),
            cantidad_variables=int(fila["cantidad_variables"]),
            algoritmo=str(fila["algoritmo"]),
            cantidad_grupos=int(fila["cantidad_grupos"]),
            silhouette=float(fila["silhouette"]),
            dataset_origen=str(fila["dataset_origen"]),
            archivo=str(fila["archivo"]),
            columnas=columnas,
        )

    def guardar_modelo(
        self,
        resultado: ResultadoEntrenamiento,
        *,
        nombre: str,
        categoria: str,
        dataset_origen: str = "Dataset sin nombre",
        mapeo_likert: dict[str, int] | None = None,
        columnas_likert: list[str] | tuple[str, ...] | None = None,
    ) -> ModeloGuardado:
        """Guarda el paquete necesario para reutilizar un entrenamiento."""
        if not isinstance(resultado, ResultadoEntrenamiento):
            raise ErrorModelo("No hay un modelo entrenado válido para guardar.")

        nombre_limpio = self._validar_texto(nombre, "El nombre")
        categoria_limpia = self._validar_texto(categoria, "La categoría")
        dataset_limpio = self._validar_texto(
            dataset_origen,
            "El dataset de origen",
            limite=255,
        )
        fecha_creacion = self.reloj().astimezone().replace(microsecond=0)
        nombre_archivo = f"{uuid4().hex}.joblib"
        ruta_final = self.directorio_modelos / nombre_archivo
        ruta_temporal = self.directorio_modelos / f".{nombre_archivo}.tmp"

        artefacto = {
            "version_formato": 1,
            "algoritmo": "K-Means",
            "modelo": resultado.modelo,
            "escalador": resultado.escalador,
            "columnas": tuple(resultado.columnas),
            "cantidad_grupos": resultado.k_usado,
            "fecha_creacion": fecha_creacion.isoformat(),
            "dataset_origen": dataset_limpio,
            "categoria": categoria_limpia,
            "mapeo_likert": dict(mapeo_likert or {}),
            "columnas_likert": tuple(columnas_likert or ()),
        }

        try:
            joblib.dump(artefacto, ruta_temporal)
            ruta_temporal.replace(ruta_final)
        except Exception as error:
            ruta_temporal.unlink(missing_ok=True)
            ruta_final.unlink(missing_ok=True)
            raise ErrorModelo("No fue posible crear el archivo del modelo.") from error

        try:
            with self._conexion() as conexion:
                cursor = conexion.execute(
                    """
                    INSERT INTO modelos_guardados (
                        nombre,
                        categoria,
                        fecha_creacion,
                        cantidad_registros,
                        cantidad_variables,
                        algoritmo,
                        cantidad_grupos,
                        silhouette,
                        dataset_origen,
                        archivo,
                        columnas
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        nombre_limpio,
                        categoria_limpia,
                        fecha_creacion.isoformat(),
                        len(resultado.asignaciones),
                        len(resultado.columnas),
                        "K-Means",
                        resultado.k_usado,
                        resultado.silhouette,
                        dataset_limpio,
                        nombre_archivo,
                        json.dumps(list(resultado.columnas)),
                    ),
                )
                modelo_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as error:
            ruta_final.unlink(missing_ok=True)
            raise ErrorModelo(
                "Ya existe un modelo con ese nombre. Utiliza un nombre diferente."
            ) from error
        except sqlite3.Error as error:
            ruta_final.unlink(missing_ok=True)
            raise ErrorModelo(
                "El archivo se creó, pero no fue posible registrar el modelo."
            ) from error

        return ModeloGuardado(
            id=modelo_id,
            nombre=nombre_limpio,
            categoria=categoria_limpia,
            fecha_creacion=fecha_creacion,
            cantidad_registros=len(resultado.asignaciones),
            cantidad_variables=len(resultado.columnas),
            algoritmo="K-Means",
            cantidad_grupos=resultado.k_usado,
            silhouette=float(resultado.silhouette),
            dataset_origen=dataset_limpio,
            archivo=nombre_archivo,
            columnas=tuple(resultado.columnas),
        )

    def listar_modelos(self) -> list[ModeloGuardado]:
        """Devuelve el catálogo persistente del modelo más reciente al más antiguo."""
        try:
            with self._conexion() as conexion:
                conexion.row_factory = sqlite3.Row
                filas = conexion.execute(
                    """
                    SELECT
                        id,
                        nombre,
                        categoria,
                        fecha_creacion,
                        cantidad_registros,
                        cantidad_variables,
                        algoritmo,
                        cantidad_grupos,
                        silhouette,
                        dataset_origen,
                        archivo,
                        columnas
                    FROM modelos_guardados
                    ORDER BY fecha_creacion DESC, id DESC
                    """
                ).fetchall()
        except sqlite3.Error as error:
            raise ErrorModelo("No fue posible consultar los modelos guardados.") from error
        return [self._desde_fila(fila) for fila in filas]

    def listar_modelos_compatibles(
        self, columnas_disponibles: Iterable[str]
    ) -> list[ModeloGuardado]:
        """Filtra el catálogo a los modelos cuyas variables existen en el dataset activo.

        Un modelo se considera compatible cuando todas las variables con las
        que fue entrenado están presentes (por nombre) entre las columnas
        numéricas disponibles del nuevo conjunto de datos. Los modelos
        guardados antes de registrar esta información no se consideran
        compatibles hasta que se vuelvan a guardar.
        """
        disponibles = {str(columna) for columna in columnas_disponibles}
        return [
            modelo
            for modelo in self.listar_modelos()
            if modelo.columnas and set(modelo.columnas).issubset(disponibles)
        ]

    def contar_modelos(self) -> int:
        """Cuenta los modelos registrados sin cargar sus archivos."""
        try:
            with self._conexion() as conexion:
                fila = conexion.execute(
                    "SELECT COUNT(*) FROM modelos_guardados"
                ).fetchone()
        except sqlite3.Error as error:
            raise ErrorModelo("No fue posible contar los modelos guardados.") from error
        return int(fila[0]) if fila else 0

    def cargar_modelo(self, modelo_id: int) -> dict:
        """Carga un artefacto interno por identificador para su futura reutilización."""
        try:
            with self._conexion() as conexion:
                fila = conexion.execute(
                    "SELECT archivo FROM modelos_guardados WHERE id = ?",
                    (int(modelo_id),),
                ).fetchone()
        except (sqlite3.Error, TypeError, ValueError) as error:
            raise ErrorModelo("No fue posible localizar el modelo solicitado.") from error

        if fila is None:
            raise ErrorModelo("El modelo solicitado no existe.")

        ruta = self.directorio_modelos / str(fila[0])
        if not ruta.is_file():
            raise ErrorModelo("El archivo asociado al modelo ya no está disponible.")
        try:
            artefacto = joblib.load(ruta)
        except Exception as error:
            raise ErrorModelo("El archivo del modelo no pudo abrirse.") from error
        if not isinstance(artefacto, dict):
            raise ErrorModelo("El archivo del modelo no tiene un formato compatible.")
        return artefacto