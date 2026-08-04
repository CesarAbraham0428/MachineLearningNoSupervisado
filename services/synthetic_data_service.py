"""Generación de respuestas sintéticas para cuestionarios Big Five.

El servicio toma el DataFrame original cargado en la aplicación, crea nuevas
respuestas para sus 25 preguntas Likert y devuelve tanto las filas sintéticas
como el archivo final en el orden solicitado: sintéticos y después originales.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Sequence

import numpy as np
import pandas as pd

from services.dataset_service import (
    ETIQUETAS_LIKERT,
    ErrorDatos,
    ServicioConjuntoDatos,
    obtener_valor_likert,
)


PREGUNTAS_POR_RASGO = 5
CANTIDAD_RASGOS = 5
CANTIDAD_PREGUNTAS = PREGUNTAS_POR_RASGO * CANTIDAD_RASGOS


class ErrorDatosSinteticos(ValueError):
    """Indica que no es posible crear el resultado sintético solicitado."""


@dataclass(frozen=True)
class ResultadoDatosSinteticos:
    """Contiene las filas nuevas y el dataset final listo para exportarse."""

    datos_sinteticos: pd.DataFrame
    datos_combinados: pd.DataFrame


def _redondear_likert(valor: float) -> int:
    """Redondea y limita un valor a la escala Likert de 1 a 5."""

    redondeado = int(math.floor(valor + 0.5))
    return max(1, min(5, redondeado))


def _ajustar_promedio(
    respuestas: list[int],
    promedio_objetivo: float,
    generador: random.Random,
) -> list[int]:
    """Ajusta cinco respuestas al promedio alcanzable más cercano."""

    suma_objetivo = max(
        PREGUNTAS_POR_RASGO,
        min(
            5 * PREGUNTAS_POR_RASGO,
            int(math.floor(promedio_objetivo * PREGUNTAS_POR_RASGO + 0.5)),
        ),
    )
    ajustadas = list(respuestas)

    while sum(ajustadas) < suma_objetivo:
        candidatas = [indice for indice, valor in enumerate(ajustadas) if valor < 5]
        ajustadas[generador.choice(candidatas)] += 1
    while sum(ajustadas) > suma_objetivo:
        candidatas = [indice for indice, valor in enumerate(ajustadas) if valor > 1]
        ajustadas[generador.choice(candidatas)] -= 1
    return ajustadas


def _generar_perfiles_sinteticos(
    dimensiones: pd.DataFrame,
    cantidad: int,
    generador: np.random.Generator,
) -> np.ndarray:
    """Genera perfiles Big Five conservando sus medias y correlaciones.

    Se reutiliza el enfoque de perfiles normales del generador original, pero
    se emplea la matriz de covarianza completa para no independizar los cinco
    rasgos. Si una muestra normal no puede producir suficientes perfiles dentro
    del rango Likert, se completa mediante remuestreo de perfiles reales.
    """

    valores = dimensiones.to_numpy(dtype=float)
    if len(valores) == 1 or np.allclose(valores, valores[0]):
        return np.repeat(valores[:1], cantidad, axis=0)

    media = valores.mean(axis=0)
    covarianza = np.cov(valores, rowvar=False, ddof=1)
    covarianza = (covarianza + covarianza.T) / 2

    perfiles: list[np.ndarray] = []
    for _ in range(20):
        pendientes = cantidad - len(perfiles)
        if pendientes <= 0:
            break
        candidatas = generador.multivariate_normal(
            media,
            covarianza,
            size=max(16, pendientes * 2),
            check_valid="ignore",
        )
        validas = candidatas[np.all((candidatas >= 1) & (candidatas <= 5), axis=1)]
        perfiles.extend(validas[:pendientes])

    if len(perfiles) < cantidad:
        pendientes = cantidad - len(perfiles)
        indices = generador.integers(0, len(valores), size=pendientes)
        perfiles.extend(valores[indice].copy() for indice in indices)

    return np.asarray(perfiles[:cantidad], dtype=float)


def _generar_respuestas_sinteticas(
    perfiles: Sequence[Sequence[float]],
    respuestas_originales: Sequence[Sequence[int]],
    generador: random.Random,
) -> tuple[list[list[int]], list[int]]:
    """Convierte perfiles en 25 respuestas y registra su fila de referencia."""

    if not respuestas_originales:
        raise ErrorDatosSinteticos("Se requieren respuestas originales como referencia.")
    if any(len(fila) != CANTIDAD_PREGUNTAS for fila in respuestas_originales):
        raise ErrorDatosSinteticos("Cada respuesta original debe contener 25 valores.")
    if any(len(perfil) != CANTIDAD_RASGOS for perfil in perfiles):
        raise ErrorDatosSinteticos("Cada perfil sintético debe contener cinco rasgos.")

    resultado: list[list[int]] = []
    referencias: list[int] = []
    for perfil in perfiles:
        indice_referencia = generador.randrange(len(respuestas_originales))
        referencia = list(respuestas_originales[indice_referencia])
        fila_sintetica: list[int] = []

        for indice_rasgo, promedio_objetivo in enumerate(perfil):
            inicio = indice_rasgo * PREGUNTAS_POR_RASGO
            fin = inicio + PREGUNTAS_POR_RASGO
            bloque_referencia = referencia[inicio:fin]
            promedio_referencia = sum(bloque_referencia) / PREGUNTAS_POR_RASGO
            bloque_inicial = [
                _redondear_likert(
                    float(promedio_objetivo) + (respuesta - promedio_referencia)
                )
                for respuesta in bloque_referencia
            ]
            fila_sintetica.extend(
                _ajustar_promedio(
                    bloque_inicial,
                    float(promedio_objetivo),
                    generador,
                )
            )

        resultado.append(fila_sintetica)
        referencias.append(indice_referencia)

    return resultado, referencias


def _representantes_likert(
    datos: pd.DataFrame,
    columnas_preguntas: Sequence[str],
) -> dict[str, dict[int, object]]:
    """Obtiene por pregunta la representación más frecuente de cada valor 1--5."""

    representantes: dict[str, dict[int, object]] = {}
    for columna in columnas_preguntas:
        conteos: dict[int, Counter[object]] = {valor: Counter() for valor in range(1, 6)}
        for respuesta in datos[columna].dropna():
            valor_likert = obtener_valor_likert(respuesta)
            if valor_likert is not None:
                conteos[valor_likert][respuesta] += 1

        es_numerica = pd.api.types.is_numeric_dtype(datos[columna])
        representantes[columna] = {
            valor: (
                conteos[valor].most_common(1)[0][0]
                if conteos[valor]
                else valor if es_numerica else ETIQUETAS_LIKERT[valor]
            )
            for valor in range(1, 6)
        }
    return representantes


def generar_dataset_sintetico(
    datos_originales: pd.DataFrame,
    cantidad: int,
    semilla: int | None = None,
) -> ResultadoDatosSinteticos:
    """Genera ``cantidad`` filas sintéticas y las antepone al dataset original.

    El dataset debe cumplir el contrato Big Five de la aplicación: preguntas
    numeradas del 1 al 25 con respuestas Likert válidas. Las columnas que no
    son preguntas se conservan desde una fila real de referencia para mantener
    el mismo esquema y distribuciones auxiliares del archivo cargado.
    """

    if isinstance(cantidad, bool) or not isinstance(cantidad, int) or cantidad < 1:
        raise ErrorDatosSinteticos("La cantidad de registros debe ser un entero mayor que cero.")
    if not isinstance(datos_originales, pd.DataFrame):
        raise TypeError("Los datos originales deben recibirse como un DataFrame de pandas.")

    try:
        preprocesamiento = ServicioConjuntoDatos().preprocesar(datos_originales)
    except ErrorDatos as error:
        raise ErrorDatosSinteticos(str(error)) from error

    generador_perfiles = np.random.default_rng(semilla)
    perfiles = _generar_perfiles_sinteticos(
        preprocesamiento.dimensiones,
        cantidad,
        generador_perfiles,
    )
    respuestas_numericas = preprocesamiento.respuestas_numericas.to_numpy(
        dtype=int
    ).tolist()
    respuestas_sinteticas, referencias = _generar_respuestas_sinteticas(
        perfiles.tolist(),
        respuestas_numericas,
        random.Random(semilla),
    )

    datos_sinteticos = datos_originales.iloc[referencias].copy(deep=True)
    datos_sinteticos.index = pd.RangeIndex(cantidad)
    representantes = _representantes_likert(
        datos_originales,
        preprocesamiento.columnas_preguntas,
    )
    for indice, columna in enumerate(preprocesamiento.columnas_preguntas):
        datos_sinteticos[columna] = [
            representantes[columna][fila[indice]]
            for fila in respuestas_sinteticas
        ]

    datos_sinteticos = datos_sinteticos.loc[:, datos_originales.columns]
    datos_combinados = pd.concat(
        [datos_sinteticos, datos_originales.copy(deep=True)],
        ignore_index=True,
    )
    return ResultadoDatosSinteticos(
        datos_sinteticos=datos_sinteticos,
        datos_combinados=datos_combinados,
    )
