"""Preparación de métricas y visualizaciones para resultados de K-Means."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA

from services.training_service import ResultadoEntrenamiento


class ErrorResultados(ValueError):
    """Indica que el resultado entrenado no puede visualizarse."""


@dataclass(frozen=True)
class ProyeccionPCA:
    """Representación bidimensional de registros y centros de los grupos."""

    puntos: pd.DataFrame
    centros: pd.DataFrame
    varianza_explicada: tuple[float, float]


class ServicioResultados:
    """Transforma un entrenamiento en información comprensible para la vista."""

    @staticmethod
    def _validar(resultado: ResultadoEntrenamiento) -> None:
        if not isinstance(resultado, ResultadoEntrenamiento):
            raise ErrorResultados("No existe un resultado de entrenamiento válido.")
        if resultado.datos_estandarizados.empty:
            raise ErrorResultados("El resultado no contiene registros agrupados.")
        if resultado.datos_estandarizados.shape[1] < 2:
            raise ErrorResultados(
                "Se necesitan al menos dos variables para visualizar los grupos."
            )

    @staticmethod
    def interpretar_silhouette(valor: float) -> tuple[str, str]:
        """Devuelve una lectura breve y una explicación de Silhouette."""
        if valor < 0:
            return (
                "Separación deficiente",
                "Algunos registros podrían estar asignados al grupo incorrecto.",
            )
        if valor < 0.25:
            return (
                "Separación débil",
                "Los grupos se mezclan considerablemente y deben interpretarse con cautela.",
            )
        if valor < 0.50:
            return (
                "Separación aceptable",
                "Existen diferencias entre los grupos, aunque todavía hay registros cercanos.",
            )
        if valor < 0.70:
            return (
                "Buena separación",
                "Los grupos presentan diferencias claras en la mayoría de los registros.",
            )
        return (
            "Separación muy clara",
            "Los grupos están bien diferenciados según las variables utilizadas.",
        )

    def crear_resumen_grupos(
        self, resultado: ResultadoEntrenamiento
    ) -> pd.DataFrame:
        """Calcula cantidad y porcentaje de registros de cada grupo."""
        self._validar(resultado)
        total = int(resultado.tamanos_clusters.sum())
        return pd.DataFrame(
            {
                "Grupo": [
                    f"Grupo {int(grupo)}"
                    for grupo in resultado.tamanos_clusters.index
                ],
                "Registros": resultado.tamanos_clusters.astype(int).to_list(),
                "Porcentaje": (
                    resultado.tamanos_clusters.astype(float) / total
                ).to_list(),
            }
        )

    def crear_proyeccion_pca(
        self, resultado: ResultadoEntrenamiento
    ) -> ProyeccionPCA:
        """Reduce los datos a dos componentes para mostrar los grupos en un plano."""
        self._validar(resultado)
        datos = resultado.datos_estandarizados.astype(float)
        asignaciones = resultado.asignaciones.reindex(datos.index)
        if asignaciones.isna().any():
            raise ErrorResultados(
                "Las asignaciones de grupo no coinciden con los registros entrenados."
            )

        pca = PCA(n_components=2)
        coordenadas = pca.fit_transform(datos)
        centros = pca.transform(resultado.centroides_estandarizados)

        puntos = pd.DataFrame(
            {
                "Componente 1": coordenadas[:, 0],
                "Componente 2": coordenadas[:, 1],
                "Grupo": asignaciones.astype(int).map(
                    lambda grupo: f"Grupo {grupo}"
                ),
                "Registro": [
                    f"Registro {posicion}"
                    for posicion in range(1, len(datos) + 1)
                ],
            },
            index=datos.index,
        )
        centros_df = pd.DataFrame(
            {
                "Componente 1": centros[:, 0],
                "Componente 2": centros[:, 1],
                "Grupo": [
                    f"Grupo {grupo}"
                    for grupo in range(1, resultado.k_usado + 1)
                ],
            }
        )
        return ProyeccionPCA(
            puntos=puntos,
            centros=centros_df,
            varianza_explicada=tuple(
                float(valor) for valor in pca.explained_variance_ratio_
            ),
        )

    def crear_tabla_centros(
        self, resultado: ResultadoEntrenamiento
    ) -> pd.DataFrame:
        """Devuelve los centros en la escala original de las variables."""
        self._validar(resultado)
        centros = resultado.centroides_originales.copy()
        centros.insert(
            0,
            "Grupo",
            [f"Grupo {grupo}" for grupo in range(1, len(centros) + 1)],
        )
        return centros

    def crear_tabla_asignaciones(
        self, resultado: ResultadoEntrenamiento
    ) -> pd.DataFrame:
        """Relaciona cada registro entrenado con el grupo encontrado."""
        self._validar(resultado)
        asignaciones = resultado.asignaciones.reindex(
            resultado.datos_estandarizados.index
        )
        return pd.DataFrame(
            {
                "Registro": [
                    f"Registro {posicion}"
                    for posicion in range(1, len(asignaciones) + 1)
                ],
                "Grupo asignado": asignaciones.astype(int).map(
                    lambda grupo: f"Grupo {grupo}"
                ),
            }
        )
