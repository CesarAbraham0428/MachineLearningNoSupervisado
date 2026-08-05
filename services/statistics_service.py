"""Estadística descriptiva de perfiles Big Five de cinco dimensiones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from services.dataset_service import validar_perfiles_big_five


@dataclass(frozen=True)
class ResumenEstadistico:
    """Datos descriptivos que consumen la vista y el reporte PDF."""

    cantidad_registros: int
    cantidad_variables: int
    estadisticas_por_rasgo: pd.DataFrame
    faltantes_por_rasgo: pd.Series
    promedio_dimensiones: pd.Series
    dimensiones_por_registro: pd.DataFrame


@dataclass(frozen=True)
class ParametrosIntervalos:
    """Parámetros calculados para agrupar un rasgo en intervalos."""

    cantidad_datos: int
    minimo: float
    maximo: float
    rango: float
    k_formula: float
    k: int
    amplitud: float


class ServicioEstadisticas:
    """Resume la matriz numérica de cinco rasgos Big Five."""

    @staticmethod
    def _calcular_moda(perfiles: pd.DataFrame) -> pd.Series:
        modas = {}
        for columna in perfiles.columns:
            valores = perfiles[columna].mode(dropna=True)
            modas[columna] = valores.iloc[0] if not valores.empty else pd.NA
        return pd.Series(modas, name="Moda")

    @classmethod
    def _crear_tabla_estadisticas(cls, perfiles: pd.DataFrame) -> pd.DataFrame:
        medias = perfiles.mean()
        desviaciones = perfiles.std(ddof=1)
        tabla = pd.DataFrame(
            {
                "Media": medias,
                "Mediana": perfiles.median(),
                "Moda": cls._calcular_moda(perfiles),
                "Varianza": perfiles.var(ddof=1),
                "Desviación estándar": desviaciones,
                "Coeficiente de variación (%)": desviaciones.divide(
                    medias.replace(0, np.nan)
                )
                * 100,
                "Mínimo": perfiles.min(),
                "Máximo": perfiles.max(),
            }
        )
        tabla.index.name = "Rasgo"
        return tabla.round(2)

    @staticmethod
    def _serie_numerica(valores: pd.Series) -> pd.Series:
        return pd.to_numeric(pd.Series(valores), errors="coerce").dropna()

    @classmethod
    def calcular_parametros_intervalos(
        cls,
        valores: pd.Series,
    ) -> ParametrosIntervalos:
        """Calcula N, mínimo, máximo, R, K y A del rasgo seleccionado."""
        serie = cls._serie_numerica(valores)
        if serie.empty:
            return ParametrosIntervalos(0, np.nan, np.nan, 0.0, 1.0, 1, 0.0)

        cantidad_datos = len(serie)
        minimo = float(serie.min())
        maximo = float(serie.max())
        rango = maximo - minimo
        k_formula = 1 + 1.3322 * np.log(cantidad_datos)
        k = max(1, int(round(k_formula)))
        if rango == 0:
            k = 1
        amplitud = rango / k if k else 0.0
        return ParametrosIntervalos(
            cantidad_datos=cantidad_datos,
            minimo=minimo,
            maximo=maximo,
            rango=rango,
            k_formula=float(k_formula),
            k=k,
            amplitud=amplitud,
        )

    @classmethod
    def calcular_frecuencia_intervalos(
        cls,
        valores: pd.Series,
        numero_intervalos: int | None = None,
        parametros: ParametrosIntervalos | None = None,
    ) -> pd.DataFrame:
        """Calcula frecuencias agrupadas en intervalos del rasgo seleccionado."""
        serie = cls._serie_numerica(valores)
        columnas = [
            "Intervalo",
            "Marca de Clase",
            "f",
            "Fr",
            "%",
            "F",
        ]
        if serie.empty:
            return pd.DataFrame(columns=columnas)

        parametros_calculados = parametros or cls.calcular_parametros_intervalos(serie)
        cantidad_intervalos = (
            parametros_calculados.k if numero_intervalos is None else numero_intervalos
        )
        if cantidad_intervalos < 1:
            raise ValueError("El número de intervalos debe ser positivo.")

        if parametros_calculados.rango == 0:
            limites = np.array(
                [parametros_calculados.minimo - 0.5, parametros_calculados.maximo + 0.5]
            )
            cantidad_intervalos = 1
        else:
            limites = np.linspace(
                parametros_calculados.minimo,
                parametros_calculados.maximo,
                cantidad_intervalos + 1,
            )
        frecuencias, _ = np.histogram(serie.to_numpy(dtype=float), bins=limites)
        marcas = (limites[:-1] + limites[1:]) / 2
        intervalos = [
            f"[{inicio:.2f} - {fin:.2f}{']' if indice == cantidad_intervalos - 1 else ')'}"
            for indice, (inicio, fin) in enumerate(zip(limites[:-1], limites[1:]))
        ]
        frecuencia_acumulada = frecuencias.cumsum()
        frecuencia_relativa = frecuencias / len(serie)

        return pd.DataFrame(
            {
                "Intervalo": intervalos,
                "Marca de Clase": marcas.round(2),
                "f": frecuencias.astype(int),
                "Fr": frecuencia_relativa.round(4),
                "%": (frecuencia_relativa * 100).round(2),
                "F": frecuencia_acumulada.astype(int),
            }
        )

    def calcular_resumen(self, datos: pd.DataFrame) -> ResumenEstadistico:
        """Calcula estadísticas directamente sobre los cinco rasgos activos."""
        perfiles = validar_perfiles_big_five(datos)
        faltantes = perfiles.isna().sum()
        faltantes.name = "Faltantes"
        return ResumenEstadistico(
            cantidad_registros=len(perfiles),
            cantidad_variables=len(perfiles.columns),
            estadisticas_por_rasgo=self._crear_tabla_estadisticas(perfiles),
            faltantes_por_rasgo=faltantes,
            promedio_dimensiones=perfiles.mean().round(2),
            dimensiones_por_registro=perfiles,
        )
