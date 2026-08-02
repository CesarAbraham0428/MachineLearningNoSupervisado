"""Preparación, evaluación y entrenamiento de K-Means."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class ErrorEntrenamiento(ValueError):
    """Indica que los datos no están listos para entrenar K-Means."""


@dataclass(frozen=True)
class EvaluacionK:
    """Métricas de un valor candidato de K."""

    k: int
    inercia: float
    silhouette: float


@dataclass(frozen=True)
class ResultadoEntrenamiento:
    """Resultado reproducible del entrenamiento final de K-Means."""

    columnas: tuple[str, ...]
    datos_estandarizados: pd.DataFrame
    asignaciones: pd.Series
    tamanos_clusters: pd.Series
    centroides_estandarizados: pd.DataFrame
    centroides_originales: pd.DataFrame
    evaluaciones: tuple[EvaluacionK, ...]
    k_recomendado: int
    k_usado: int
    inercia: float
    silhouette: float
    modelo: KMeans
    escalador: StandardScaler


class ServicioEntrenamiento:
    """Encapsula el flujo numérico necesario para entrenar K-Means."""

    def __init__(self, random_state: int = 42, n_init: int = 10):
        self.random_state = random_state
        self.n_init = n_init

    @staticmethod
    def validar_datos(datos: pd.DataFrame) -> pd.DataFrame:
        """Valida y devuelve una copia numérica apta para clustering."""
        if not isinstance(datos, pd.DataFrame):
            raise TypeError("Los datos de entrenamiento deben ser un DataFrame.")
        if datos.empty:
            raise ErrorEntrenamiento("No hay registros para entrenar el modelo.")
        if len(datos) < 3:
            raise ErrorEntrenamiento(
                "Se requieren al menos 3 registros para evaluar clústeres."
            )
        if datos.shape[1] < 2:
            raise ErrorEntrenamiento(
                "Selecciona al menos 2 variables numéricas para entrenar."
            )

        columnas_no_numericas = [
            str(columna)
            for columna in datos.columns
            if not pd.api.types.is_numeric_dtype(datos[columna])
        ]
        if columnas_no_numericas:
            raise ErrorEntrenamiento(
                "Todas las variables deben ser numéricas. Faltan convertir: "
                + ", ".join(columnas_no_numericas)
                + "."
            )
        if datos.isna().any().any():
            raise ErrorEntrenamiento(
                "Existen valores faltantes. Corrígelos antes de entrenar."
            )

        copia = datos.astype(float).copy()
        if not np.isfinite(copia.to_numpy()).all():
            raise ErrorEntrenamiento(
                "Las variables contienen valores infinitos o no válidos."
            )
        return copia

    def estandarizar(self, datos: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
        """Estandariza las variables y conserva sus nombres e índices."""
        datos_validos = self.validar_datos(datos)
        escalador = StandardScaler()
        transformados = escalador.fit_transform(datos_validos)
        return (
            pd.DataFrame(
                transformados,
                index=datos_validos.index,
                columns=datos_validos.columns,
            ),
            escalador,
        )

    @staticmethod
    def _candidatos_k(cantidad_registros: int, maximo_k: int) -> range:
        """Limita K para evitar demasiados clústeres en muestras pequeñas.

        La raíz cuadrada del número de registros funciona como un límite
        conservador de candidatos. La elección final dentro de ese rango sigue
        realizándose con Silhouette.
        """
        limite_muestra = max(2, round(sqrt(cantidad_registros)))
        limite = min(maximo_k, cantidad_registros - 1, limite_muestra)
        if limite < 2:
            raise ErrorEntrenamiento(
                "No hay suficientes registros para comparar valores de K."
            )
        return range(2, limite + 1)

    def evaluar_k(
        self, datos: pd.DataFrame, maximo_k: int = 8
    ) -> tuple[EvaluacionK, ...]:
        """Evalúa K candidatos con inercia y Silhouette sobre datos estandarizados."""
        if maximo_k < 2:
            raise ValueError("El máximo de K debe ser al menos 2.")

        estandarizados, _ = self.estandarizar(datos)
        evaluaciones: list[EvaluacionK] = []
        for k in self._candidatos_k(len(estandarizados), maximo_k):
            modelo = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            etiquetas = modelo.fit_predict(estandarizados)
            if len(np.unique(etiquetas)) < 2:
                continue
            evaluaciones.append(
                EvaluacionK(
                    k=k,
                    inercia=float(modelo.inertia_),
                    silhouette=float(silhouette_score(estandarizados, etiquetas)),
                )
            )

        if not evaluaciones:
            raise ErrorEntrenamiento(
                "No fue posible formar al menos dos clústeres distintos con los datos."
            )
        return tuple(evaluaciones)

    @staticmethod
    def recomendar_k(evaluaciones: Iterable[EvaluacionK]) -> int:
        """Recomienda el K con mayor Silhouette; ante empate, el menor K."""
        opciones = tuple(evaluaciones)
        if not opciones:
            raise ErrorEntrenamiento("No existen métricas para recomendar K.")
        mejor = min(opciones, key=lambda item: (-item.silhouette, item.k))
        return mejor.k

    def entrenar_modelo(
        self,
        datos: pd.DataFrame,
        k: int | None = None,
        maximo_k: int = 8,
    ) -> ResultadoEntrenamiento:
        """Estandariza los datos, recomienda K y ajusta el modelo final."""
        estandarizados, escalador = self.estandarizar(datos)
        evaluaciones = self.evaluar_k(datos, maximo_k=maximo_k)
        recomendado = self.recomendar_k(evaluaciones)
        k_usado = recomendado if k is None else int(k)

        opciones_validas = {evaluacion.k for evaluacion in evaluaciones}
        if k_usado not in opciones_validas:
            raise ErrorEntrenamiento(
                "El valor de K debe estar dentro de los candidatos evaluados: "
                + ", ".join(map(str, sorted(opciones_validas)))
                + "."
            )

        modelo = KMeans(
            n_clusters=k_usado,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        etiquetas = modelo.fit_predict(estandarizados)
        if len(np.unique(etiquetas)) < 2:
            raise ErrorEntrenamiento(
                "El modelo no pudo separar los registros en clústeres distintos."
            )

        centroides_estandarizados = pd.DataFrame(
            modelo.cluster_centers_, columns=estandarizados.columns
        )
        centroides_originales = pd.DataFrame(
            escalador.inverse_transform(modelo.cluster_centers_),
            columns=estandarizados.columns,
        )
        asignaciones = pd.Series(
            etiquetas + 1,
            index=estandarizados.index,
            name="Cluster",
        )
        tamanos = asignaciones.value_counts().sort_index()
        silhouette_final = silhouette_score(estandarizados, etiquetas)

        return ResultadoEntrenamiento(
            columnas=tuple(map(str, estandarizados.columns)),
            datos_estandarizados=estandarizados,
            asignaciones=asignaciones,
            tamanos_clusters=tamanos,
            centroides_estandarizados=centroides_estandarizados,
            centroides_originales=centroides_originales,
            evaluaciones=evaluaciones,
            k_recomendado=recomendado,
            k_usado=k_usado,
            inercia=float(modelo.inertia_),
            silhouette=float(silhouette_final),
            modelo=modelo,
            escalador=escalador,
        )
