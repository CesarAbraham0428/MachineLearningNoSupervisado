"""Estadística descriptiva de perfiles Big Five de cinco dimensiones."""

from __future__ import annotations

from dataclasses import dataclass

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
        tabla = pd.DataFrame(
            {
                "Media": perfiles.mean(),
                "Mediana": perfiles.median(),
                "Moda": cls._calcular_moda(perfiles),
                "Desviación estándar": perfiles.std(ddof=1),
                "Mínimo": perfiles.min(),
                "Máximo": perfiles.max(),
            }
        )
        tabla.index.name = "Rasgo"
        return tabla.round(2)

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
