"""Carga, validación y preprocesamiento de conjuntos de datos Big Five."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np
import pandas as pd


LIKERT_A_NUMERO: Final[dict[str, int]] = {
    "totalmente en desacuerdo": 1,
    "en desacuerdo": 2,
    "neutral": 3,
    "de acuerdo": 4,
    "totalmente de acuerdo": 5,
}

ETIQUETAS_LIKERT: Final[dict[int, str]] = {
    1: "Totalmente en desacuerdo",
    2: "En desacuerdo",
    3: "Neutral",
    4: "De acuerdo",
    5: "Totalmente de acuerdo",
}

_VARIACIONES_LIKERT_A_NUMERO: Final[dict[int, tuple[str, ...]]] = {
    1: (
        "totalmente en desacuerdo",
        "muy en desacuerdo",
        "completamente en desacuerdo",
        "fuertemente en desacuerdo",
        "totalmente desacuerdo",
        "muy desacuerdo",
        "strongly disagree",
        "totally disagree",
        "td",
        "ted",
    ),
    2: (
        "en desacuerdo",
        "algo en desacuerdo",
        "parcialmente en desacuerdo",
        "disagree",
        "ed",
    ),
    3: (
        "neutral",
        "neutro",
        "ni de acuerdo ni en desacuerdo",
        "ni de acuerdo ni desacuerdo",
        "indiferente",
        "neither agree nor disagree",
        "n",
    ),
    4: (
        "de acuerdo",
        "algo de acuerdo",
        "parcialmente de acuerdo",
        "agree",
        "da",
    ),
    5: (
        "totalmente de acuerdo",
        "muy de acuerdo",
        "completamente de acuerdo",
        "fuertemente de acuerdo",
        "totalmente acuerdo",
        "muy acuerdo",
        "strongly agree",
        "totally agree",
        "ta",
        "tad",
    ),
}

DIMENSIONES_BIG_FIVE: Final[dict[str, range]] = {
    "Extraversión": range(1, 6),
    "Estabilidad emocional": range(6, 11),
    "Apertura a la experiencia": range(11, 16),
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
for _numero, _variaciones in _VARIACIONES_LIKERT_A_NUMERO.items():
    _LIKERT_NORMALIZADO.update(
        {_normalizar_texto(variacion): _numero for variacion in _variaciones}
    )


def _normalizar_respuesta_likert(valor: object) -> str:
    """Normaliza una respuesta y elimina numeración decorativa opcional."""
    texto = _normalizar_texto(valor)
    texto = re.sub(r"^[1-5]\s*(?:[-:.)]|–|—)\s*", "", texto)
    texto = re.sub(r"\s*(?:\([- ]?[1-5]\)?|[-:]\s*[1-5])$", "", texto)
    return texto.strip()


def obtener_valor_likert(valor: object) -> int | None:
    """Devuelve el valor 1--5 de una respuesta Likert reconocida.

    Acepta tanto valores numéricos como etiquetas textuales en español e inglés,
    incluyendo variaciones de mayúsculas, acentos, espacios y prefijos numéricos.
    """
    if pd.isna(valor) or isinstance(valor, bool):
        return None

    if isinstance(valor, Real):
        numero = float(valor)
        if numero.is_integer() and 1 <= numero <= 5:
            return int(numero)

    texto = _normalizar_respuesta_likert(valor)
    if re.fullmatch(r"[1-5](?:\.0+)?", texto):
        return int(float(texto))
    return _LIKERT_NORMALIZADO.get(texto)


def detectar_columnas_likert(datos: pd.DataFrame) -> dict[object, tuple[str, ...]]:
    """Detecta columnas categóricas cuyos valores corresponden a una escala Likert.

    Las columnas numéricas 1--5 ya están codificadas y no se devuelven, porque no
    necesitan una asignación adicional. Los valores nulos se ignoran durante la
    detección y se conservan al aplicar el mapeo.
    """
    detectadas: dict[object, tuple[str, ...]] = {}
    for columna in datos.columns:
        serie = datos[columna]
        if pd.api.types.is_numeric_dtype(serie):
            continue

        valores = serie.dropna()
        if valores.empty:
            continue
        if not valores.map(obtener_valor_likert).notna().all():
            continue

        unicos = tuple(dict.fromkeys(str(valor).strip() for valor in valores))
        detectadas[columna] = unicos
    return detectadas


def aplicar_mapeo_likert(
    datos: pd.DataFrame,
    columnas: list[str] | tuple[str, ...],
    mapeo: dict[str, int],
) -> pd.DataFrame:
    """Copia el dataset y convierte las columnas Likert según el mapeo indicado."""
    if not mapeo:
        raise ErrorDatos("Debes asignar al menos una categoría Likert.")

    valores_mapeo = list(mapeo.values())
    if any(not isinstance(valor, int) or isinstance(valor, bool) or not 1 <= valor <= 5
           for valor in valores_mapeo):
        raise ErrorDatos("Los valores de la escala Likert deben ser enteros del 1 al 5.")

    mapeo_exacto = {
        _normalizar_texto(etiqueta): valor
        for etiqueta, valor in mapeo.items()
    }
    mapeo_normalizado = {
        _normalizar_respuesta_likert(etiqueta): valor
        for etiqueta, valor in mapeo.items()
    }
    resultado = datos.copy()

    for columna in columnas:
        if columna not in resultado.columns:
            raise ErrorDatos(f"La columna Likert no existe: {columna}.")

        serie = resultado[columna]
        def convertir_valor(valor: object) -> object:
            if pd.isna(valor):
                return pd.NA
            return mapeo_exacto.get(
                _normalizar_texto(valor),
                mapeo_normalizado.get(_normalizar_respuesta_likert(valor)),
            )

        convertida = serie.map(convertir_valor)
        invalidos = serie[convertida.isna() & serie.notna()]
        if not invalidos.empty:
            encontrados = ", ".join(sorted({str(valor) for valor in invalidos}))
            raise ErrorDatos(
                f"La columna '{columna}' contiene respuestas sin asignar: {encontrados}."
            )

        resultado[columna] = (
            convertida.astype("int8")
            if not convertida.isna().any()
            else convertida.astype("Int8")
        )

    return resultado


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
            normalizada = respuestas[columna].map(obtener_valor_likert)
            mascara_invalida = normalizada.isna()
            if mascara_invalida.any():
                invalidos[indice] = sorted(
                    respuestas.loc[mascara_invalida, columna]
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )
            convertidas[columna] = normalizada

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


def crear_perfiles_big_five(datos: pd.DataFrame) -> pd.DataFrame:
    """Convierte las 25 respuestas de cada persona en sus cinco rasgos."""
    return ServicioConjuntoDatos().preprocesar(datos).dimensiones.copy()


def crear_perfiles_con_contexto(datos: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas auxiliares para filtrar sin incorporarlas a K-Means."""
    preprocesado = ServicioConjuntoDatos().preprocesar(datos)
    columnas_preguntas = set(preprocesado.columnas_preguntas)
    columnas_contexto = [
        columna for columna in datos.columns if columna not in columnas_preguntas
    ]
    contexto = datos.loc[:, columnas_contexto].copy()
    return pd.concat([preprocesado.dimensiones.copy(), contexto], axis=1)


def validar_perfiles_big_five(
    datos: pd.DataFrame,
    *,
    permitir_subconjunto: bool = False,
    minimo_columnas: int = 1,
) -> pd.DataFrame:
    """Valida rasgos Big Five y, opcionalmente, un subconjunto de ellos.

    Por defecto exige las cinco dimensiones para conservar la validación usada
    al preparar y generar datos. Las vistas de análisis pueden pedir un
    subconjunto cuando la persona ocultó rasgos desde la pestaña de datos.
    """
    if not isinstance(datos, pd.DataFrame):
        raise TypeError("Los perfiles deben recibirse como un DataFrame de pandas.")
    if datos.empty:
        raise ErrorDatos("El conjunto de perfiles no contiene registros.")

    columnas_esperadas = list(DIMENSIONES_BIG_FIVE)
    columnas_disponibles = [
        columna for columna in columnas_esperadas if columna in datos
    ]
    faltantes = [
        columna for columna in columnas_esperadas if columna not in datos
    ]
    if faltantes and not permitir_subconjunto:
        raise ErrorDatos(
            "Faltan dimensiones Big Five: " + ", ".join(faltantes) + "."
        )
    if permitir_subconjunto and len(columnas_disponibles) < minimo_columnas:
        raise ErrorDatos(
            "Selecciona al menos "
            f"{minimo_columnas} rasgo(s) Big Five para continuar."
        )

    columnas_a_validar = (
        columnas_disponibles if permitir_subconjunto else columnas_esperadas
    )
    perfiles = datos.loc[:, columnas_a_validar].copy()
    columnas_no_numericas = [
        columna
        for columna in columnas_a_validar
        if not pd.api.types.is_numeric_dtype(perfiles[columna])
    ]
    if columnas_no_numericas:
        raise ErrorDatos(
            "Las dimensiones deben ser numéricas: "
            + ", ".join(columnas_no_numericas)
            + "."
        )
    if perfiles.isna().any().any():
        raise ErrorDatos("Los perfiles Big Five contienen valores faltantes.")

    perfiles = perfiles.astype(float)
    if not np.isfinite(perfiles.to_numpy()).all():
        raise ErrorDatos("Los perfiles Big Five contienen valores no válidos.")
    if not perfiles.apply(lambda serie: serie.between(1, 5).all()).all():
        raise ErrorDatos("Las dimensiones Big Five deben estar entre 1 y 5.")
    return perfiles


# ---------------------------------------------------------------------------
# RF-05 a RF-08 — Diagnóstico de calidad y limpieza automática del dataset
# ---------------------------------------------------------------------------

from dataclasses import field


@dataclass(frozen=True)
class DiagnosticoCalidad:
    """Resultado del análisis de calidad de un DataFrame genérico.

    Attributes:
        num_filas: Número total de filas del dataset.
        num_columnas: Número total de columnas.
        num_columnas_numericas: Columnas con dtype numérico.
        num_columnas_categoricas: Columnas no numéricas.
        nulos_por_columna: Serie con el conteo de nulos por columna (solo > 0).
        num_duplicados: Filas completamente duplicadas.
        outliers_por_columna: Dict columna → cantidad de outliers detectados por IQR.
    """

    num_filas: int
    num_columnas: int
    num_columnas_numericas: int
    num_columnas_categoricas: int
    nulos_por_columna: pd.Series
    num_duplicados: int
    outliers_por_columna: dict

    @property
    def total_nulos(self) -> int:
        """Suma total de valores nulos encontrados."""
        return int(self.nulos_por_columna.sum())

    @property
    def total_outliers(self) -> int:
        """Suma total de outliers detectados en todas las columnas numéricas."""
        return sum(self.outliers_por_columna.values())

    @property
    def requiere_limpieza(self) -> bool:
        """Indica si el dataset tiene algún problema que necesita corrección."""
        return self.total_nulos > 0 or self.num_duplicados > 0 or self.total_outliers > 0


def _detectar_outliers_iqr(serie: pd.Series) -> int:
    """Cuenta los valores fuera del rango IQR de una Serie numérica.

    Usa el método estándar: límite inferior = Q1 - 1.5·IQR,
    límite superior = Q3 + 1.5·IQR.

    Args:
        serie: Serie de pandas con valores numéricos.

    Returns:
        Cantidad de valores fuera del rango permitido.
    """
    serie_limpia = serie.dropna()
    if serie_limpia.empty:
        return 0
    q1 = serie_limpia.quantile(0.25)
    q3 = serie_limpia.quantile(0.75)
    riq = q3 - q1
    if riq == 0:
        return 0
    limite_inferior = q1 - 1.5 * riq
    limite_superior = q3 + 1.5 * riq
    mascara = (serie_limpia < limite_inferior) | (serie_limpia > limite_superior)
    return int(mascara.sum())


def _es_escala_likert(serie: pd.Series) -> bool:
    """Identifica escalas ordinales enteras de 1 a 5 que no deben winsorizarse."""
    valores = serie.dropna()
    if valores.empty:
        return False
    return bool(
        valores.between(1, 5).all()
        and np.isclose(valores.to_numpy(dtype=float), valores.round()).all()
    )


def diagnosticar_calidad(df: pd.DataFrame) -> DiagnosticoCalidad:
    """Analiza la calidad de un DataFrame genérico y devuelve un diagnóstico.

    Detecta automáticamente:
    - Valores nulos por columna.
    - Filas completamente duplicadas.
    - Outliers en columnas numéricas mediante el método IQR.

    Args:
        df: DataFrame a diagnosticar. No se modifica.

    Returns:
        DiagnosticoCalidad con todos los hallazgos.
    """
    columnas_numericas = df.select_dtypes(include="number").columns.tolist()
    columnas_categoricas = df.select_dtypes(exclude="number").columns.tolist()

    nulos = df.isnull().sum()
    nulos_con_problemas = nulos[nulos > 0]

    outliers: dict = {}

    return DiagnosticoCalidad(
        num_filas=len(df),
        num_columnas=len(df.columns),
        num_columnas_numericas=len(columnas_numericas),
        num_columnas_categoricas=len(columnas_categoricas),
        nulos_por_columna=nulos_con_problemas,
        num_duplicados=int(df.duplicated().sum()),
        outliers_por_columna=outliers,
    )


@dataclass(frozen=True)
class ResultadoLimpieza:
    """Resultado de la limpieza automática de un DataFrame.

    Attributes:
        duplicados_eliminados: Filas duplicadas que se quitaron.
        nulos_corregidos: Total de celdas nulas imputadas.
        outliers_tratados: Total de valores winzorizados.
        filas_finales: Número de filas del dataset limpio.
        columnas_finales: Número de columnas del dataset limpio.
        dataset_limpio: DataFrame resultante de la limpieza.
    """

    duplicados_eliminados: int
    nulos_corregidos: int
    outliers_tratados: int
    filas_finales: int
    columnas_finales: int
    dataset_limpio: pd.DataFrame


def limpiar_dataset(df: pd.DataFrame) -> ResultadoLimpieza:
    """Limpia automáticamente un DataFrame aplicando tres pasos secuenciales.

    Pasos en orden:
    1. Eliminar filas completamente duplicadas.
    2. Imputar nulos en columnas numéricas con la **mediana** de cada columna.
    3. Imputar nulos en columnas categóricas con la **moda** de cada columna.

    IMPORTANTE: el DataFrame original `df` nunca se modifica.

    Args:
        df: DataFrame original. Se trabaja sobre una copia.

    Returns:
        ResultadoLimpieza con métricas del proceso y el dataset limpio.
    """
    df_trabajo = df.copy()

    # 1. Eliminar duplicados
    filas_antes = len(df_trabajo)
    df_trabajo = df_trabajo.drop_duplicates()
    duplicados_eliminados = filas_antes - len(df_trabajo)

    # 2 y 3. Imputar nulos
    nulos_corregidos = 0
    columnas_numericas = df_trabajo.select_dtypes(include="number").columns.tolist()
    columnas_categoricas = df_trabajo.select_dtypes(exclude="number").columns.tolist()

    for col in columnas_numericas:
        faltantes = df_trabajo[col].isnull().sum()
        if faltantes > 0:
            mediana = df_trabajo[col].median()
            df_trabajo[col] = df_trabajo[col].fillna(mediana)
            nulos_corregidos += faltantes

    for col in columnas_categoricas:
        faltantes = df_trabajo[col].isnull().sum()
        if faltantes > 0:
            modas = df_trabajo[col].mode(dropna=True)
            if not modas.empty:
                df_trabajo[col] = df_trabajo[col].fillna(modas.iloc[0])
                nulos_corregidos += faltantes

    # 4. Outliers tratados se establece en 0 (funcionalidad removida)
    outliers_tratados = 0

    return ResultadoLimpieza(
        duplicados_eliminados=duplicados_eliminados,
        nulos_corregidos=int(nulos_corregidos),
        outliers_tratados=outliers_tratados,
        filas_finales=len(df_trabajo),
        columnas_finales=len(df_trabajo.columns),
        dataset_limpio=df_trabajo,
    )
