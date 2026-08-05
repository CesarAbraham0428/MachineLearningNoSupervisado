"""Generación de perfiles sintéticos de cinco dimensiones Big Five."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from services.dataset_service import ErrorDatos, validar_perfiles_big_five


VALOR_MINIMO = 1.0
VALOR_MAXIMO = 5.0
MAXIMO_INTENTOS_LOTE = 100


class ErrorDatosSinteticos(ValueError):
    """Indica que no es posible crear el resultado sintético solicitado."""


@dataclass(frozen=True)
class ResultadoDatosSinteticos:
    """Perfiles originales, sintéticos y combinados listos para analizar."""

    datos_originales: pd.DataFrame
    datos_sinteticos: pd.DataFrame
    datos_combinados: pd.DataFrame
    medias: pd.Series
    desviaciones: pd.Series


def _generar_dimension_normal_limitada(
    media: float,
    desviacion: float,
    cantidad: int,
    generador: np.random.Generator,
) -> np.ndarray:
    """Muestrea una normal y rechaza valores fuera de la escala 1--5."""
    if np.isclose(desviacion, 0.0):
        return np.full(cantidad, media, dtype=float)

    aceptados: list[np.ndarray] = []
    total = 0
    for _ in range(MAXIMO_INTENTOS_LOTE):
        pendientes = cantidad - total
        if pendientes <= 0:
            break
        candidatos = generador.normal(media, desviacion, size=max(32, pendientes * 2))
        validos = candidatos[
            (candidatos >= VALOR_MINIMO) & (candidatos <= VALOR_MAXIMO)
        ]
        if validos.size:
            lote = validos[:pendientes]
            aceptados.append(lote)
            total += len(lote)

    if total < cantidad:
        raise ErrorDatosSinteticos(
            "No fue posible generar suficientes valores dentro de la escala 1 a 5."
        )
    return np.concatenate(aceptados)[:cantidad]


def generar_dataset_sintetico(
    datos_originales: pd.DataFrame,
    cantidad: int,
) -> ResultadoDatosSinteticos:
    """Genera perfiles normales y los antepone a los perfiles originales.

    La entrada debe representar a cada persona mediante al menos dos rasgos
    Big Five activos. Cada dimensión sintética se genera con la media y
    desviación estándar muestral observadas, conservando valores continuos
    entre 1 y 5.
    """
    if isinstance(cantidad, bool) or not isinstance(cantidad, int) or cantidad < 1:
        raise ErrorDatosSinteticos(
            "La cantidad de registros debe ser un entero mayor que cero."
        )

    try:
        perfiles_originales = validar_perfiles_big_five(
            datos_originales,
            permitir_subconjunto=True,
            minimo_columnas=2,
        )
    except ErrorDatos as error:
        raise ErrorDatosSinteticos(str(error)) from error

    if len(perfiles_originales) < 2:
        raise ErrorDatosSinteticos(
            "Se requieren al menos dos perfiles originales para calcular la desviación."
        )

    # 1. Se obtienen parámetros de los datos reales
    medias = perfiles_originales.mean(axis=0)
    desviaciones = perfiles_originales.std(axis=0, ddof=1)
    if desviaciones.isna().any():
        raise ErrorDatosSinteticos(
            "No fue posible calcular la desviación estándar de los rasgos."
        )

    # 2. Generación de perfiles sintéticos
    generador = np.random.default_rng()
    sinteticos = pd.DataFrame(
        {
            columna: _generar_dimension_normal_limitada(
                media=float(medias[columna]),
                desviacion=float(desviaciones[columna]),
                cantidad=cantidad,
                generador=generador,
            )
            for columna in perfiles_originales.columns
        }
    ).round(2)
    sinteticos = sinteticos.loc[:, perfiles_originales.columns]
    #3 Unión con los datos originales
    combinados = pd.concat(
        [sinteticos, perfiles_originales.reset_index(drop=True)],
        ignore_index=True,
    )
    return ResultadoDatosSinteticos(
        datos_originales=perfiles_originales,
        datos_sinteticos=sinteticos,
        datos_combinados=combinados,
        medias=medias,
        desviaciones=desviaciones,
    )
