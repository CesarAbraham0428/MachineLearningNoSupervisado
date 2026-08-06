"""Preparación de métricas y visualizaciones para resultados de K-Means."""

from __future__ import annotations

from numbers import Integral

import pandas as pd

from services.training_service import ResultadoEntrenamiento


class ErrorResultados(ValueError):
    """Indica que el resultado entrenado no puede visualizarse."""


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

    def crear_interpretaciones_grupos(
        self, resultado: ResultadoEntrenamiento
    ) -> pd.DataFrame:
        """Crea una lectura clara de cada rasgo para cada grupo."""
        self._validar(resultado)
        centros = resultado.centroides_originales
        promedios_generales = pd.Series(
            resultado.escalador.mean_, index=resultado.columnas, dtype=float
        )
        escalas = pd.Series(
            resultado.escalador.scale_, index=resultado.columnas, dtype=float
        )
        umbral = 0.35
        lecturas: list[dict[str, object]] = []

        for posicion, (_, centro) in enumerate(centros.iterrows(), start=1):
            for rasgo in resultado.columnas:
                valor_grupo = float(centro[rasgo])
                promedio_general = float(promedios_generales[rasgo])
                diferencia_estandarizada = (
                    valor_grupo - promedio_general
                ) / float(escalas[rasgo])
                if diferencia_estandarizada >= umbral:
                    nivel = "alto"
                    comparacion = "Supera el promedio"
                elif diferencia_estandarizada <= -umbral:
                    nivel = "bajo"
                    comparacion = "Está por debajo del promedio"
                else:
                    nivel = "cercano"
                    comparacion = "Está cerca del promedio"

                lecturas.append(
                    {
                        "Grupo": f"Grupo {posicion}",
                        "Rasgo": rasgo,
                        "Valor del grupo": valor_grupo,
                        "Promedio general": promedio_general,
                        "Comparación": comparacion,
                        "Interpretación": self._descripcion_rasgo(rasgo, nivel),
                    }
                )

        return pd.DataFrame(lecturas)

    def crear_resumen_perfiles_grupos(
        self, resultado: ResultadoEntrenamiento
    ) -> pd.DataFrame:
        """Resume los rasgos que más distinguen el perfil de cada grupo."""
        self._validar(resultado)
        promedios_generales = pd.Series(
            resultado.escalador.mean_, index=resultado.columnas, dtype=float
        )
        escalas = pd.Series(
            resultado.escalador.scale_, index=resultado.columnas, dtype=float
        )
        umbral = 0.35
        perfiles: list[dict[str, str]] = []

        for posicion, (_, centro) in enumerate(
            resultado.centroides_originales.iterrows(), start=1
        ):
            diferencias = (centro - promedios_generales) / escalas
            rasgos = (
                diferencias[abs(diferencias) >= umbral]
                .abs()
                .sort_values(ascending=False)
                .head(3)
                .index
                .tolist()
            )
            descripciones = [
                self._frase_perfil(
                    rasgo,
                    "alto" if diferencias[rasgo] >= 0 else "bajo",
                )
                for rasgo in rasgos
            ]
            if descripciones:
                perfil = (
                    "Este grupo reúne principalmente perfiles "
                    f"{self._unir_descripciones(descripciones)}."
                )
            else:
                perfil = (
                    "Este grupo tiene un perfil cercano al promedio en los rasgos "
                    "analizados."
                )
            perfiles.append({"Grupo": f"Grupo {posicion}", "Perfil": perfil})

        return pd.DataFrame(perfiles)

    @staticmethod
    def _frase_perfil(rasgo: str, nivel: str) -> str:
        """Convierte un rasgo destacado en una frase breve de perfil."""
        frases = {
            "extraversión": {
                "alto": "más sociables y comunicativos",
                "bajo": "más reservados al interactuar socialmente",
            },
            "estabilidad emocional": {
                "alto": "más tranquilos ante situaciones de presión",
                "bajo": "más sensibles al estrés y la presión",
            },
            "apertura a la experiencia": {
                "alto": "más creativos, imaginativos y abiertos a ideas nuevas",
                "bajo": "más prácticos y con preferencia por lo conocido",
            },
            "responsabilidad": {
                "alto": "más organizados, disciplinados y constantes",
                "bajo": "más flexibles y menos estructurados al organizar tareas",
            },
            "amabilidad": {
                "alto": "más empáticos y cooperativos",
                "bajo": "más directos y competitivos al relacionarse",
            },
        }
        frase = frases.get(str(rasgo).strip().casefold())
        if frase:
            return frase[nivel]
        direccion = "por encima" if nivel == "alto" else "por debajo"
        return f"con {rasgo} {direccion} del promedio"

    @staticmethod
    def _unir_descripciones(descripciones: list[str]) -> str:
        """Une descripciones de perfil en una oración legible."""
        if len(descripciones) == 1:
            return descripciones[0]
        if len(descripciones) == 2:
            return f"{descripciones[0]} y {descripciones[1]}"
        return f"{', '.join(descripciones[:-1])} y {descripciones[-1]}"
    @staticmethod
    def _descripcion_rasgo(rasgo: str, nivel: str) -> str:
        """Devuelve una frase prefabricada para cada rasgo y comparación."""
        lecturas = {
            "extraversión": {
                "alto": "Esto sugiere perfiles más sociables, comunicativos y participativos.",
                "bajo": "Esto sugiere perfiles más reservados y con menor interés por la interacción social.",
            },
            "estabilidad emocional": {
                "alto": "Esto sugiere perfiles más tranquilos y seguros al afrontar situaciones de presión.",
                "bajo": "Esto sugiere perfiles más sensibles al estrés y a la presión.",
            },
            "apertura a la experiencia": {
                "alto": "Esto sugiere perfiles más creativos, imaginativos y abiertos a ideas nuevas.",
                "bajo": "Esto sugiere perfiles más prácticos y con preferencia por lo conocido.",
            },
            "responsabilidad": {
                "alto": "Esto sugiere perfiles más organizados, disciplinados y constantes en tareas o estudios.",
                "bajo": "Esto sugiere perfiles más flexibles y menos estructurados al organizar tareas o estudios.",
            },
            "amabilidad": {
                "alto": "Esto sugiere perfiles más empáticos, cooperativos y considerados con los demás.",
                "bajo": "Esto sugiere perfiles más directos y competitivos al relacionarse con los demás.",
            },
        }
        if nivel == "cercano":
            return (
                "Este rasgo se mantiene cercano al promedio, por lo que no "
                "distingue especialmente a este grupo."
            )
        lectura = lecturas.get(str(rasgo).strip().casefold())
        if lectura:
            return lectura[nivel]
        direccion = "más alto" if nivel == "alto" else "más bajo"
        return f"El valor de este rasgo es {direccion} que el promedio general."

    def crear_tabla_asignaciones(
        self,
        resultado: ResultadoEntrenamiento,
        datos_originales: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Relaciona cada registro entrenado, sus valores y el grupo encontrado."""
        self._validar(resultado)
        indices_entrenados = resultado.datos_estandarizados.index
        asignaciones = resultado.asignaciones.reindex(indices_entrenados)
        if asignaciones.isna().any():
            raise ErrorResultados(
                "Las asignaciones de grupo no coinciden con los registros entrenados."
            )

        identificadores = pd.Series(
            [
                f"Registro {int(indice) + 1}"
                if isinstance(indice, Integral) and int(indice) >= 0
                else str(indice)
                for indice in indices_entrenados
            ],
            index=indices_entrenados,
            name="Identificador",
        )
        grupos = asignaciones.astype(int).map(lambda grupo: f"Grupo {grupo}")

        if datos_originales is None:
            return pd.DataFrame(
                {
                    "Identificador": identificadores.to_list(),
                    "Grupo asignado": grupos.to_list(),
                }
            )
        if not isinstance(datos_originales, pd.DataFrame):
            raise TypeError("Los datos originales deben ser un DataFrame.")
        if not datos_originales.index.is_unique:
            raise ErrorResultados(
                "Los datos originales contienen identificadores de fila repetidos."
            )

        indices_faltantes = indices_entrenados.difference(datos_originales.index)
        if not indices_faltantes.empty:
            raise ErrorResultados(
                "No fue posible localizar todos los registros entrenados en los "
                "datos cargados."
            )

        detalle = datos_originales.reindex(indices_entrenados).copy()
        nombres_reservados = {"Identificador", "Grupo asignado"}
        detalle = detalle.rename(
            columns={
                columna: f"{columna} (dato original)"
                for columna in detalle.columns
                if columna in nombres_reservados
            }
        )
        detalle.insert(0, "Identificador", identificadores.to_list())
        detalle["Grupo asignado"] = grupos.to_list()
        return detalle.reset_index(drop=True)
