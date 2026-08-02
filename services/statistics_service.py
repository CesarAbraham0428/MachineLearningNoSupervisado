"""Cálculo de estadísticas descriptivas para el cuestionario Big Five."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from services.dataset_service import ResultadoPreprocesamiento, ServicioConjuntoDatos


@dataclass(frozen=True)
class ResumenEstadistico:
    """Datos calculados que consumen la vista y el reporte estadístico."""

    cantidad_registros: int
    cantidad_variables: int
    estadisticas_por_pregunta: pd.DataFrame
    faltantes_por_pregunta: pd.Series
    frecuencia_respuestas: pd.DataFrame
    promedio_dimensiones: pd.Series
    respuestas_numericas: pd.DataFrame
    dimensiones_por_registro: pd.DataFrame


class ServicioEstadisticas:
    """Genera el resumen descriptivo requerido por RF-05."""

    def __init__(self, servicio_datos: ServicioConjuntoDatos | None = None):
        self._servicio_datos = servicio_datos or ServicioConjuntoDatos()

    @staticmethod
    def _calcular_moda(respuestas: pd.DataFrame) -> pd.Series:
        """Calcula la primera moda por pregunta de manera determinista."""
        modas = {}
        for columna in respuestas.columns:
            valores = respuestas[columna].mode(dropna=True)
            modas[columna] = valores.iloc[0] if not valores.empty else pd.NA
        return pd.Series(modas, name="Moda")

    @staticmethod
    def _crear_tabla_estadisticas(respuestas: pd.DataFrame) -> pd.DataFrame:
        """Agrupa las medidas descriptivas relevantes por variable."""
        tabla = pd.DataFrame(
            {
                "Media": respuestas.mean(),
                "Mediana": respuestas.median(),
                "Moda": ServicioEstadisticas._calcular_moda(respuestas),
                "Desviación estándar": respuestas.std(ddof=1),
                "Mínimo": respuestas.min(),
                "Máximo": respuestas.max(),
            }
        )
        tabla.index.name = "Pregunta"
        return tabla.round(2)

    @staticmethod
    def _calcular_frecuencias(respuestas: pd.DataFrame) -> pd.DataFrame:
        """Cuenta respuestas Likert y muestra también el porcentaje global."""
        conteos = (
            respuestas.stack()
            .value_counts()
            .reindex(range(1, 6), fill_value=0)
            .sort_index()
        )
        total = int(conteos.sum())
        etiquetas = {
            1: "Totalmente en desacuerdo",
            2: "En desacuerdo",
            3: "Neutral",
            4: "De acuerdo",
            5: "Totalmente de acuerdo",
        }
        frecuencia = pd.DataFrame(
            {
                "Valor Likert": conteos.index,
                "Respuesta": [etiquetas[valor] for valor in conteos.index],
                "Frecuencia": conteos.values,
                "Porcentaje": (conteos.values / total * 100) if total else 0,
            }
        )
        return frecuencia.round({"Porcentaje": 2})

    def calcular_resumen(self, datos: pd.DataFrame) -> ResumenEstadistico:
        """Calcula todas las estadísticas del dataset antes del entrenamiento."""
        preprocesado: ResultadoPreprocesamiento = self._servicio_datos.preprocesar(datos)
        respuestas = preprocesado.respuestas_numericas

        return ResumenEstadistico(
            cantidad_registros=len(respuestas),
            cantidad_variables=len(respuestas.columns),
            estadisticas_por_pregunta=self._crear_tabla_estadisticas(respuestas),
            faltantes_por_pregunta=datos.loc[:, preprocesado.columnas_preguntas]
            .isna()
            .sum(),
            frecuencia_respuestas=self._calcular_frecuencias(respuestas),
            promedio_dimensiones=preprocesado.dimensiones.mean().round(2),
            respuestas_numericas=respuestas,
            dimensiones_por_registro=preprocesado.dimensiones,
        )
