"""Pasos 3 y 4: convertir perfiles Big Five en respuestas Likert.

El script toma los 349 perfiles sinteticos de cinco rasgos, genera para cada
uno una fila de 25 respuestas enteras y despues recupera las etiquetas
textuales del cuestionario. Tambien crea un dataset ampliado con las 51 filas
originales y las 349 sinteticas.

Uso:
    python aleatorios/02_generar_respuestas_likert.py \
        --entrada-original "ruta/al/cuestionario.csv"

Los cinco reactivos de cada rasgo se generan usando como referencia el patron
de respuestas de una persona real. De esta forma se conserva la diferencia
observada entre preguntas y el promedio resultante queda lo mas cerca posible
del perfil sintetico (en pasos de 0.2, porque cada rasgo tiene cinco reactivos).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence


DIRECTORIO_ALEATORIOS = Path(__file__).resolve().parent
RUTA_PERFILES_PREDETERMINADA = (
    DIRECTORIO_ALEATORIOS / "perfiles_sinteticos.csv"
)
RUTA_NUMERICA_PREDETERMINADA = (
    DIRECTORIO_ALEATORIOS / "respuestas_sinteticas_numericas.csv"
)
RUTA_TEXTO_PREDETERMINADA = (
    DIRECTORIO_ALEATORIOS / "respuestas_sinteticas_texto.csv"
)
RUTA_COMPLETA_PREDETERMINADA = (
    DIRECTORIO_ALEATORIOS / "cuestionario_ampliado.csv"
)

PREGUNTAS_POR_RASGO = 5
CANTIDAD_RASGOS = 5
CANTIDAD_PREGUNTAS = PREGUNTAS_POR_RASGO * CANTIDAD_RASGOS
PERFILES_SINTETICOS_ESPERADOS = 349

NOMBRES_RASGOS = (
    "extraversion",
    "estabilidad_emocional",
    "apertura",
    "responsabilidad",
    "amabilidad",
)

ETIQUETAS_LIKERT = {
    1: "Totalmente en desacuerdo",
    2: "En desacuerdo",
    3: "Neutral",
    4: "De acuerdo",
    5: "Totalmente de acuerdo",
}

_PATRON_PREGUNTA = re.compile(r"^\s*(\d{1,2})\s*\.")


def _normalizar_texto(valor: object) -> str:
    """Normaliza espacios, mayusculas y acentos para comparar etiquetas."""

    texto = " ".join(str(valor).strip().casefold().split())
    return "".join(
        caracter
        for caracter in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(caracter)
    )


_VALORES_LIKERT = {
    _normalizar_texto(etiqueta): valor
    for valor, etiqueta in ETIQUETAS_LIKERT.items()
}


def convertir_respuesta_a_numero(respuesta: object) -> int:
    """Convierte una etiqueta Likert valida en un entero de 1 a 5."""

    if isinstance(respuesta, (int, float)) and not isinstance(respuesta, bool):
        numero = float(respuesta)
        if numero.is_integer() and 1 <= numero <= 5:
            return int(numero)

    texto = _normalizar_texto(respuesta)
    if texto in _VALORES_LIKERT:
        return _VALORES_LIKERT[texto]
    if re.fullmatch(r"[1-5](?:\.0+)?", texto):
        return int(float(texto))
    raise ValueError(f"Respuesta Likert no reconocida: {respuesta!r}.")


def _indices_preguntas(encabezados: Sequence[str]) -> list[int]:
    """Localiza y ordena exactamente las preguntas numeradas del 1 al 25."""

    encontradas: dict[int, int] = {}
    for indice, encabezado in enumerate(encabezados):
        coincidencia = _PATRON_PREGUNTA.match(str(encabezado))
        if not coincidencia:
            continue
        numero = int(coincidencia.group(1))
        if numero in encontradas:
            raise ValueError(f"La pregunta {numero} aparece mas de una vez.")
        encontradas[numero] = indice

    esperadas = set(range(1, CANTIDAD_PREGUNTAS + 1))
    if set(encontradas) != esperadas:
        faltantes = sorted(esperadas - set(encontradas))
        raise ValueError(
            "El cuestionario debe contener las preguntas 1 a 25. "
            f"Faltantes: {faltantes}."
        )
    return [encontradas[numero] for numero in sorted(encontradas)]


def leer_cuestionario_original(
    ruta: Path,
) -> tuple[list[str], list[list[str]], list[list[int]]]:
    """Lee las 25 preguntas originales y devuelve texto y escala numerica."""

    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        filas = list(csv.reader(archivo))
    if len(filas) < 2:
        raise ValueError("El cuestionario original no contiene respuestas.")

    encabezados, filas_datos = filas[0], filas[1:]
    indices = _indices_preguntas(encabezados)
    encabezados_preguntas = [encabezados[indice].strip() for indice in indices]
    respuestas_texto: list[list[str]] = []
    respuestas_numericas: list[list[int]] = []

    for numero_fila, fila in enumerate(filas_datos, start=2):
        if len(fila) != len(encabezados):
            raise ValueError(
                f"La fila {numero_fila} tiene {len(fila)} columnas; "
                f"se esperaban {len(encabezados)}."
            )
        textos = [fila[indice].strip() for indice in indices]
        try:
            numeros = [convertir_respuesta_a_numero(valor) for valor in textos]
        except ValueError as error:
            raise ValueError(f"Error en la fila {numero_fila}: {error}") from error
        respuestas_texto.append(textos)
        respuestas_numericas.append(numeros)

    return encabezados_preguntas, respuestas_texto, respuestas_numericas


def leer_perfiles_sinteticos(ruta: Path) -> list[list[float]]:
    """Lee y valida la matriz de cinco rasgos creada en los pasos 1 y 2."""

    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if lector.fieldnames != list(NOMBRES_RASGOS):
            raise ValueError(
                "El archivo de perfiles debe contener, en orden: "
                + ", ".join(NOMBRES_RASGOS)
                + "."
            )
        perfiles: list[list[float]] = []
        for numero_fila, fila in enumerate(lector, start=2):
            try:
                perfil = [float(fila[rasgo]) for rasgo in NOMBRES_RASGOS]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"El perfil de la fila {numero_fila} contiene valores invalidos."
                ) from error
            if any(not 1 <= valor <= 5 for valor in perfil):
                raise ValueError(
                    f"El perfil de la fila {numero_fila} contiene valores fuera "
                    "del rango 1 a 5."
                )
            perfiles.append(perfil)

    if not perfiles:
        raise ValueError("El archivo de perfiles sinteticos esta vacio.")
    return perfiles


def _redondear_likert(valor: float) -> int:
    """Redondea al entero mas cercano y limita el resultado al rango 1 a 5."""

    redondeado = int(math.floor(valor + 0.5))
    return max(1, min(5, redondeado))


def _ajustar_promedio(
    respuestas: list[int],
    promedio_objetivo: float,
    generador: random.Random,
) -> list[int]:
    """Ajusta la suma para aproximar el promedio objetivo en pasos de 0.2."""

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


def generar_respuestas_sinteticas(
    perfiles: Sequence[Sequence[float]],
    respuestas_originales: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Convierte perfiles 5D en filas de 25 respuestas Likert numericas.

    Para cada perfil se selecciona una fila real como referencia. En cada rasgo
    se conserva el patron relativo de sus cinco preguntas, se desplaza hacia el
    promedio sintetico y finalmente se ajusta la suma al promedio alcanzable
    mas cercano.
    """

    if not respuestas_originales:
        raise ValueError("Se requieren respuestas originales como referencia.")
    if any(len(fila) != CANTIDAD_PREGUNTAS for fila in respuestas_originales):
        raise ValueError("Cada respuesta original debe contener 25 valores.")
    if any(len(perfil) != CANTIDAD_RASGOS for perfil in perfiles):
        raise ValueError("Cada perfil sintetico debe contener cinco rasgos.")

    generador = random.Random()
    resultado: list[list[int]] = []

    for perfil in perfiles:
        referencia = list(generador.choice(respuestas_originales))
        fila_sintetica: list[int] = []

        for indice_rasgo, promedio_objetivo in enumerate(perfil):
            inicio = indice_rasgo * PREGUNTAS_POR_RASGO
            fin = inicio + PREGUNTAS_POR_RASGO
            bloque_referencia = referencia[inicio:fin]
            promedio_referencia = sum(bloque_referencia) / PREGUNTAS_POR_RASGO
            bloque_inicial = [
                _redondear_likert(
                    float(promedio_objetivo)
                    + (respuesta - promedio_referencia)
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

    return resultado


def convertir_respuestas_a_texto(
    respuestas: Iterable[Sequence[int]],
) -> list[list[str]]:
    """Recupera las etiquetas textuales de todas las respuestas sinteticas."""

    resultado: list[list[str]] = []
    for numero_fila, fila in enumerate(respuestas, start=1):
        if len(fila) != CANTIDAD_PREGUNTAS:
            raise ValueError(
                f"La fila sintetica {numero_fila} debe contener 25 respuestas."
            )
        try:
            resultado.append([ETIQUETAS_LIKERT[int(valor)] for valor in fila])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"La fila sintetica {numero_fila} contiene un valor fuera de 1 a 5."
            ) from error
    return resultado


def guardar_csv(
    ruta: Path,
    encabezados: Sequence[str],
    filas: Iterable[Sequence[object]],
) -> None:
    """Guarda una tabla CSV con codificacion compatible con Excel."""

    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezados)
        escritor.writerows(filas)


def leer_argumentos() -> argparse.Namespace:
    """Configura las entradas y salidas del proceso reproducible."""

    analizador = argparse.ArgumentParser(
        description=(
            "Genera 25 respuestas Likert para cada perfil sintetico y crea "
            "el cuestionario ampliado."
        )
    )
    analizador.add_argument(
        "--entrada-original",
        type=Path,
        required=True,
        help="Ruta del cuestionario original con las 51 respuestas.",
    )
    analizador.add_argument(
        "--entrada-perfiles",
        type=Path,
        default=RUTA_PERFILES_PREDETERMINADA,
        help="CSV de perfiles sinteticos de cinco rasgos.",
    )
    analizador.add_argument(
        "--salida-numerica",
        type=Path,
        default=RUTA_NUMERICA_PREDETERMINADA,
        help="CSV de 349 filas por 25 respuestas enteras.",
    )
    analizador.add_argument(
        "--salida-texto",
        type=Path,
        default=RUTA_TEXTO_PREDETERMINADA,
        help="CSV de respuestas sinteticas con etiquetas Likert.",
    )
    analizador.add_argument(
        "--salida-completa",
        type=Path,
        default=RUTA_COMPLETA_PREDETERMINADA,
        help="CSV final con filas originales y sinteticas.",
    )
    return analizador.parse_args()


def ejecutar() -> None:
    """Ejecuta los pasos 3 y 4 y guarda los tres resultados auditables."""

    argumentos = leer_argumentos()
    encabezados, originales_texto, originales_numericas = (
        leer_cuestionario_original(argumentos.entrada_original)
    )
    perfiles = leer_perfiles_sinteticos(argumentos.entrada_perfiles)

    if len(perfiles) != PERFILES_SINTETICOS_ESPERADOS:
        raise ValueError(
            f"Se esperaban {PERFILES_SINTETICOS_ESPERADOS} perfiles sinteticos "
            f"y se encontraron {len(perfiles)}."
        )

    sinteticas_numericas = generar_respuestas_sinteticas(
        perfiles,
        originales_numericas,
    )
    sinteticas_texto = convertir_respuestas_a_texto(sinteticas_numericas)
    cuestionario_ampliado = [
        ["Original", *fila] for fila in originales_texto
    ] + [
        ["Sintetico", *fila] for fila in sinteticas_texto
    ]

    guardar_csv(argumentos.salida_numerica, encabezados, sinteticas_numericas)
    guardar_csv(argumentos.salida_texto, encabezados, sinteticas_texto)
    guardar_csv(
        argumentos.salida_completa,
        ["Origen", *encabezados],
        cuestionario_ampliado,
    )

    print(
        f"Paso 3 completado: {len(sinteticas_numericas)} filas x "
        f"{len(encabezados)} respuestas numericas."
    )
    print(
        f"Paso 4 completado: {len(sinteticas_texto)} filas convertidas "
        "a etiquetas Likert."
    )
    print(
        f"Dataset final: {len(cuestionario_ampliado)} perfiles x "
        f"{len(encabezados)} preguntas = "
        f"{len(cuestionario_ampliado) * len(encabezados):,} respuestas."
    )


if __name__ == "__main__":
    ejecutar()
