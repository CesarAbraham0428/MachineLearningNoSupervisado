"""Carga, validación y preprocesamiento de conjuntos de datos Big Five."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

import pandas as pd


LIKERT_A_NUMERO: Final[dict[str, int]] = {
    "totalmente en desacuerdo": 1,
    "en desacuerdo": 2,
    "neutral": 3,
    "de acuerdo": 4,
    "totalmente de acuerdo": 5,
}

DIMENSIONES_BIG_FIVE: Final[dict[str, range]] = {
    "Extraversión": range(1, 6),
    "Estabilidad emocional": range(6, 11),
    "Apertura": range(11, 16),
    "Responsabilidad": range(16, 21),
    "Amabilidad": range(21, 26),
}

_PATRON_NUMERO_PREGUNTA = re.compile(r"^\s*(\d{1,2})\s*\.")


class ErrorDatos(ValueError):
    """Indica que el dataset no cumple el contrato esperado."""


@dataclass(frozen=True)
class ResultadoPreprocesamiento:
    """Resultado común que consumirán estadísticas y entrenamiento."""

    respuestas_numericas: pd.DataFrame
    dimensiones: pd.DataFrame
    columnas_preguntas: tuple[str, ...]
    columna_temporal: str | None


def _normalizar_texto(valor: object) -> str:
    """Normaliza espacios, mayúsculas y acentos para comparar respuestas."""
    texto = " ".join(str(valor).strip().lower().split())
    return "".join(
        caracter
        for caracter in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(caracter)
    )


_LIKERT_NORMALIZADO: Final[dict[str, int]] = {
    _normalizar_texto(etiqueta): numero
    for etiqueta, numero in LIKERT_A_NUMERO.items()
}


class ServicioConjuntoDatos:
    """Prepara el contrato de datos que comparten RF-05, RF-06 y RF-07."""

    def cargar_datos(self, ruta_archivo: str) -> pd.DataFrame:
        """Carga un CSV o Excel y rechaza archivos vacíos."""
        ruta = str(ruta_archivo)
        if ruta.lower().endswith(".csv"):
            datos = pd.read_csv(ruta)
        elif ruta.lower().endswith(".xlsx"):
            datos = pd.read_excel(ruta, engine="openpyxl")
        else:
            raise ErrorDatos("Formato no compatible. Usa un archivo CSV o XLSX.")

        if datos.empty:
            raise ErrorDatos("El archivo no contiene registros.")
        return datos

    @staticmethod
    def identificar_columnas_preguntas(datos: pd.DataFrame) -> list[str]:
        """Encuentra y ordena las preguntas numeradas del 1 al 25."""
        numeradas: list[tuple[int, str]] = []
        numeros_repetidos: set[int] = set()
        numeros_vistos: set[int] = set()

        for columna in datos.columns:
            coincidencia = _PATRON_NUMERO_PREGUNTA.match(str(columna))
            if not coincidencia:
                continue
            numero = int(coincidencia.group(1))
            if numero in numeros_vistos:
                numeros_repetidos.add(numero)
            numeros_vistos.add(numero)
            numeradas.append((numero, columna))

        if numeros_repetidos:
            raise ErrorDatos(
                "Hay números de pregunta repetidos: "
                + ", ".join(map(str, sorted(numeros_repetidos)))
                + "."
            )

        esperados = set(range(1, 26))
        encontrados = {numero for numero, _ in numeradas}
        faltantes = sorted(esperados - encontrados)
        adicionales = sorted(encontrados - esperados)
        if faltantes or adicionales:
            detalles = []
            if faltantes:
                detalles.append(f"faltan preguntas: {faltantes}")
            if adicionales:
                detalles.append(f"sobran preguntas numeradas: {adicionales}")
            raise ErrorDatos(
                "Se requieren exactamente las preguntas 1 a 25; "
                + "; ".join(detalles)
                + "."
            )

        return [columna for _, columna in sorted(numeradas)]

    @staticmethod
    def identificar_columna_temporal(datos: pd.DataFrame) -> str | None:
        """Localiza la marca temporal sin incluirla en el análisis."""
        nombres_aceptados = {
            "marca temporal",
            "timestamp",
            "fecha",
            "fecha de respuesta",
        }
        for columna in datos.columns:
            if _normalizar_texto(columna) in nombres_aceptados:
                return columna
        return None

    @staticmethod
    def convertir_likert(
        datos: pd.DataFrame, columnas_preguntas: list[str]
    ) -> pd.DataFrame:
        """Convierte las 25 respuestas Likert a enteros entre 1 y 5."""
        respuestas = datos.loc[:, columnas_preguntas].copy()

        faltantes_por_columna = respuestas.isna().sum()
        faltantes_por_columna = faltantes_por_columna[faltantes_por_columna > 0]
        if not faltantes_por_columna.empty:
            detalle = ", ".join(
                f"pregunta {columnas_preguntas.index(columna) + 1}: {cantidad}"
                for columna, cantidad in faltantes_por_columna.items()
            )
            raise ErrorDatos(f"Existen respuestas incompletas ({detalle}).")

        convertidas = pd.DataFrame(index=respuestas.index)
        invalidos: dict[int, list[str]] = {}
        for indice, columna in enumerate(columnas_preguntas, start=1):
            normalizada = respuestas[columna].map(_normalizar_texto)
            mascara_invalida = ~normalizada.isin(_LIKERT_NORMALIZADO)
            if mascara_invalida.any():
                invalidos[indice] = sorted(
                    respuestas.loc[mascara_invalida, columna]
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )
            convertidas[columna] = normalizada.map(_LIKERT_NORMALIZADO)

        if invalidos:
            detalle = "; ".join(
                f"pregunta {numero}: {valores}"
                for numero, valores in invalidos.items()
            )
            raise ErrorDatos(f"Se encontraron respuestas Likert inválidas ({detalle}).")

        return convertidas.astype("int8")

    @staticmethod
    def calcular_dimensiones(
        respuestas_numericas: pd.DataFrame,
        columnas_preguntas: list[str],
    ) -> pd.DataFrame:
        """Calcula el promedio de las cinco preguntas de cada dimensión."""
        dimensiones = pd.DataFrame(index=respuestas_numericas.index)
        for dimension, numeros in DIMENSIONES_BIG_FIVE.items():
            columnas = [columnas_preguntas[numero - 1] for numero in numeros]
            dimensiones[dimension] = respuestas_numericas[columnas].mean(axis=1)
        return dimensiones

    def preprocesar(self, datos: pd.DataFrame) -> ResultadoPreprocesamiento:
        """Valida el dataset y devuelve respuestas y dimensiones numéricas."""
        if not isinstance(datos, pd.DataFrame):
            raise TypeError("Los datos deben recibirse como un DataFrame de pandas.")
        if datos.empty:
            raise ErrorDatos("El conjunto de datos no contiene registros.")

        columnas = self.identificar_columnas_preguntas(datos)
        respuestas = self.convertir_likert(datos, columnas)
        dimensiones = self.calcular_dimensiones(respuestas, columnas)

        return ResultadoPreprocesamiento(
            respuestas_numericas=respuestas,
            dimensiones=dimensiones,
            columnas_preguntas=tuple(columnas),
            columna_temporal=self.identificar_columna_temporal(datos),
        )
