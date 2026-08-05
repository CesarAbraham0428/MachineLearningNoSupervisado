"""Generacion de perfiles sinteticos Big Five con perfiles base separados."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from services.dataset_service import (
    DIMENSIONES_BIG_FIVE,
    ErrorDatos,
    validar_perfiles_big_five,
)


VALOR_MINIMO = 1.0
VALOR_MAXIMO = 5.0
MAXIMO_INTENTOS_LOTE = 100
COLUMNA_ORIGEN = "Origen"
ETIQUETA_ORIGINAL = "Original"
PREFIJO_SINTETICO = "Sintetico"


class ErrorDatosSinteticos(ValueError):
    """Indica que no es posible crear el resultado sintetico solicitado."""


@dataclass(frozen=True)
class ResultadoDatosSinteticos:
    """Perfiles originales, sinteticos y combinados listos para analizar."""

    datos_originales: pd.DataFrame
    datos_sinteticos: pd.DataFrame
    datos_combinados: pd.DataFrame
    medias: pd.Series
    desviaciones: pd.Series
    resumen_grupos: pd.DataFrame


def _estandarizar(valores: np.ndarray) -> np.ndarray:
    """Estandariza para medir separacion sin sesgarse por un rasgo disperso."""
    desviaciones = valores.std(axis=0, ddof=1)
    desviaciones = np.where(np.isclose(desviaciones, 0.0), 1.0, desviaciones)
    return (valores - valores.mean(axis=0)) / desviaciones


def _seleccionar_perfiles_base(
    perfiles: np.ndarray,
    cantidad_perfiles: int,
    generador: np.random.Generator,
) -> list[int]:
    """Elige un perfil aleatorio y despues los mas alejados de los ya elegidos.

    Es una variante determinista de la idea de separacion de K-Means++: cada
    nuevo perfil maximiza su distancia al perfil base mas cercano.
    """
    estandarizados = _estandarizar(perfiles)
    indices = [int(generador.integers(len(estandarizados)))]

    while len(indices) < cantidad_perfiles:
        elegidos = estandarizados[indices]
        distancias_cuadradas = (
            (estandarizados[:, np.newaxis, :] - elegidos[np.newaxis, :, :]) ** 2
        ).sum(axis=2)
        distancia_minima = distancias_cuadradas.min(axis=1)
        distancia_minima[indices] = -np.inf
        indice_nuevo = int(np.argmax(distancia_minima))
        if not np.isfinite(distancia_minima[indice_nuevo]) or np.isclose(
            distancia_minima[indice_nuevo], 0.0
        ):
            break
        indices.append(indice_nuevo)

    return indices


def _asignar_grupos(perfiles: np.ndarray, indices_base: list[int]) -> np.ndarray:
    """Asigna cada perfil real al perfil base estandarizado mas cercano."""
    estandarizados = _estandarizar(perfiles)
    bases = estandarizados[indices_base]
    distancias = (
        (estandarizados[:, np.newaxis, :] - bases[np.newaxis, :, :]) ** 2
    ).sum(axis=2)
    return distancias.argmin(axis=1)


def _covarianza_para_grupo(
    perfiles_grupo: np.ndarray,
    covarianza_global: np.ndarray,
) -> np.ndarray:
    """Obtiene una covarianza estable, incluso para grupos pequenos."""
    if np.allclose(covarianza_global, 0.0):
        return np.zeros_like(covarianza_global)
    if len(perfiles_grupo) >= 2:
        covarianza = np.cov(perfiles_grupo, rowvar=False, ddof=1)
    else:
        covarianza = covarianza_global * 0.25

    covarianza = np.asarray(covarianza, dtype=float)
    covarianza = np.nan_to_num(covarianza, nan=0.0, posinf=0.0, neginf=0.0)
    covarianza = (covarianza + covarianza.T) / 2
    escala = float(np.mean(np.diag(covarianza_global)))
    regularizacion = max(escala * 1e-3, 1e-4)
    return covarianza + np.eye(covarianza.shape[0]) * regularizacion


def _generar_perfiles_limitados(
    media: np.ndarray,
    covarianza: np.ndarray,
    cantidad: int,
    generador: np.random.Generator,
) -> np.ndarray:
    """Genera perfiles multivariados y solo conserva los que estan en 1--5."""
    aceptados: list[np.ndarray] = []
    total = 0
    for _ in range(MAXIMO_INTENTOS_LOTE):
        pendientes = cantidad - total
        if pendientes <= 0:
            break
        candidatos = generador.multivariate_normal(
            media,
            covarianza,
            size=max(32, pendientes * 3),
            check_valid="ignore",
        )
        validos = candidatos[
            ((candidatos >= VALOR_MINIMO) & (candidatos <= VALOR_MAXIMO)).all(axis=1)
        ]
        if len(validos):
            lote = validos[:pendientes]
            aceptados.append(lote)
            total += len(lote)

    if total < cantidad:
        raise ErrorDatosSinteticos(
            "No fue posible generar suficientes perfiles dentro de la escala 1 a 5."
        )
    return np.vstack(aceptados)[:cantidad]


def _agregar_origen(datos: pd.DataFrame, origen: pd.Series | str) -> pd.DataFrame:
    """Antepone la procedencia sin modificar los cinco rasgos."""
    resultado = datos.reset_index(drop=True).copy()
    resultado.insert(0, COLUMNA_ORIGEN, origen)
    return resultado


def generar_dataset_sintetico(
    datos_originales: pd.DataFrame,
    cantidad: int,
    semilla: int | None = None,
    cantidad_perfiles_base: int = 2,
) -> ResultadoDatosSinteticos:
    """Genera datos alrededor de perfiles originales separados.

    Los originales nunca se modifican. Se eligen hasta dos perfiles base lejanos
    entre si, se agrupan los registros reales alrededor de ellos y se generan
    perfiles multivariados para cada grupo. ``Origen`` identifica cada fila.
    """
    if isinstance(cantidad, bool) or not isinstance(cantidad, int) or cantidad < 1:
        raise ErrorDatosSinteticos(
            "La cantidad de registros debe ser un entero mayor que cero."
        )
    if (
        isinstance(cantidad_perfiles_base, bool)
        or not isinstance(cantidad_perfiles_base, int)
        or cantidad_perfiles_base < 1
    ):
        raise ErrorDatosSinteticos("Debe seleccionarse al menos un perfil base.")

    try:
        perfiles_originales = validar_perfiles_big_five(datos_originales)
    except ErrorDatos as error:
        raise ErrorDatosSinteticos(str(error)) from error

    if len(perfiles_originales) < 2:
        raise ErrorDatosSinteticos(
            "Se requieren al menos dos perfiles originales para generar datos."
        )

    medias = perfiles_originales.mean(axis=0)
    desviaciones = perfiles_originales.std(axis=0, ddof=1)
    valores = perfiles_originales.to_numpy(dtype=float)
    covarianza_global = np.cov(valores, rowvar=False, ddof=1)
    generador = np.random.default_rng(semilla)
    cantidad_bases = min(cantidad_perfiles_base, len(perfiles_originales))
    indices_base = _seleccionar_perfiles_base(valores, cantidad_bases, generador)
    asignaciones = _asignar_grupos(valores, indices_base)

    cantidad_grupos = len(indices_base)
    cuotas = np.full(cantidad_grupos, cantidad // cantidad_grupos, dtype=int)
    cuotas[: cantidad % cantidad_grupos] += 1
    partes: list[pd.DataFrame] = []
    resumen: list[dict[str, object]] = []
    for indice_grupo, indice_base in enumerate(indices_base):
        perfiles_grupo = valores[asignaciones == indice_grupo]
        cantidad_grupo = int(cuotas[indice_grupo])
        generados = _generar_perfiles_limitados(
            perfiles_grupo.mean(axis=0),
            _covarianza_para_grupo(perfiles_grupo, covarianza_global),
            cantidad_grupo,
            generador,
        )
        partes.append(
            _agregar_origen(
                pd.DataFrame(generados, columns=DIMENSIONES_BIG_FIVE).round(2),
                f"{PREFIJO_SINTETICO} - perfil base {indice_grupo + 1}",
            )
        )
        resumen.append(
            {
                "Perfil base": indice_grupo + 1,
                "Fila original de referencia": int(indice_base) + 1,
                "Originales cercanos": int((asignaciones == indice_grupo).sum()),
                "Sinteticos generados": cantidad_grupo,
            }
        )

    sinteticos = pd.concat(partes, ignore_index=True)
    originales_con_origen = _agregar_origen(perfiles_originales, ETIQUETA_ORIGINAL)
    combinados = pd.concat([sinteticos, originales_con_origen], ignore_index=True)
    return ResultadoDatosSinteticos(
        datos_originales=originales_con_origen,
        datos_sinteticos=sinteticos,
        datos_combinados=combinados,
        medias=medias,
        desviaciones=desviaciones,
        resumen_grupos=pd.DataFrame(resumen),
    )
