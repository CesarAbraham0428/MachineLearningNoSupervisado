"""Paso 1: convertir las respuestas del cuestionario en promedios Big Five.

El resultado se conserva en memoria como una matriz de 51 filas x 5 columnas
numericas y se imprime en consola. Los valores de respuesta se convierten a la
escala 1 a 5:

    Totalmente en desacuerdo = 1
    En desacuerdo            = 2
    Neutral                  = 3
    De acuerdo               = 4
    Totalmente de acuerdo    = 5

Uso:
    python 01_promedios_big_five.py
    python 01_promedios_big_five.py --entrada "ruta/al/cuestionario.csv"
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path


RUTA_ENTRADA_PREDETERMINADA = Path(
    r"C:\Users\lopez\Downloads\Cuestionario de Personalidad.csv"
)
RUTA_SALIDA_PREDETERMINADA = (
    Path(__file__).resolve().parent / "perfiles_sinteticos.csv"
)
ENCUESTADOS_ESPERADOS = 51
PREGUNTAS_POR_RASGO = 5
PERFILES_SINTETICOS_ESPERADOS = 349
VALOR_MINIMO = 1.0
VALOR_MAXIMO = 5.0
SEMILLA_ALEATORIA = 42

VALORES_RESPUESTA = {
    "Totalmente en desacuerdo": 1,
    "En desacuerdo": 2,
    "Neutral": 3,
    "De acuerdo": 4,
    "Totalmente de acuerdo": 5,
}

RASGOS_CINCO_GRANDES = {
    "extraversion": range(0, PREGUNTAS_POR_RASGO),
    "estabilidad_emocional": range(
        PREGUNTAS_POR_RASGO, 2 * PREGUNTAS_POR_RASGO
    ),
    "apertura": range(2 * PREGUNTAS_POR_RASGO, 3 * PREGUNTAS_POR_RASGO),
    "responsabilidad": range(
        3 * PREGUNTAS_POR_RASGO, 4 * PREGUNTAS_POR_RASGO
    ),
    "amabilidad": range(4 * PREGUNTAS_POR_RASGO, 5 * PREGUNTAS_POR_RASGO),
}

NOMBRES_RASGOS = list(RASGOS_CINCO_GRANDES)
ENCABEZADOS_MATRIZ = [
    "id_encuestado",
    "marca_temporal",
    *NOMBRES_RASGOS,
]


def convertir_respuesta_a_numero(
    respuesta: str, numero_fila: int, numero_pregunta: int
) -> int:
    """Convierte una respuesta textual a su valor numerico Likert."""

    respuesta_normalizada = respuesta.strip()
    try:
        return VALORES_RESPUESTA[respuesta_normalizada]
    except KeyError as error:
        valores_validos = ", ".join(VALORES_RESPUESTA)
        raise ValueError(
            f"Respuesta no reconocida en la fila {numero_fila}, "
            f"pregunta {numero_pregunta}: {respuesta!r}. "
            f"Valores validos: {valores_validos}."
        ) from error


def calcular_promedios_cinco_grandes(ruta_entrada: Path) -> list[list[object]]:
    """Calcula una fila de promedios por cada encuestado, sin guardar archivos."""

    with ruta_entrada.open("r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.reader(archivo)
        filas = list(lector)

    if not filas:
        raise ValueError("El archivo CSV esta vacio.")

    encabezados, filas_datos = filas[0], filas[1:]
    encabezados_preguntas = encabezados[1:]
    preguntas_esperadas = len(RASGOS_CINCO_GRANDES) * PREGUNTAS_POR_RASGO

    if len(encabezados_preguntas) != preguntas_esperadas:
        raise ValueError(
            f"Se esperaban {preguntas_esperadas} preguntas y se encontraron "
            f"{len(encabezados_preguntas)}."
        )

    if len(filas_datos) != ENCUESTADOS_ESPERADOS:
        raise ValueError(
            f"Se esperaban {ENCUESTADOS_ESPERADOS} encuestados y se encontraron "
            f"{len(filas_datos)}."
        )

    matriz: list[list[object]] = []

    for numero_fila, fila in enumerate(filas_datos, start=2):
        if len(fila) != len(encabezados):
            raise ValueError(
                f"La fila {numero_fila} tiene {len(fila)} columnas; "
                f"se esperaban {len(encabezados)}."
            )

        respuestas_numericas = [
            convertir_respuesta_a_numero(respuesta, numero_fila, numero_pregunta)
            for numero_pregunta, respuesta in enumerate(fila[1:], start=1)
        ]

        promedios = [
            round(
                sum(respuestas_numericas[indice] for indice in indices_preguntas)
                / PREGUNTAS_POR_RASGO,
                2,
            )
            for indices_preguntas in RASGOS_CINCO_GRANDES.values()
        ]

        matriz.append([len(matriz) + 1, fila[0], *promedios])

    return matriz


def calcular_estadisticas_dimensiones(
    matriz: list[list[object]],
) -> list[dict[str, object]]:
    """Calcula la media y la desviacion estandar muestral de cada rasgo."""

    estadisticas: list[dict[str, object]] = []

    for indice_rasgo, nombre_rasgo in enumerate(NOMBRES_RASGOS):
        valores_rasgo = [
            float(fila[indice_rasgo + 2])
            for fila in matriz
        ]
        estadisticas.append(
            {
                "rasgo": nombre_rasgo,
                "media": statistics.mean(valores_rasgo),
                "desviacion_estandar": statistics.stdev(valores_rasgo),
            }
        )

    return estadisticas


def imprimir_estadisticas_dimensiones(
    estadisticas: list[dict[str, object]],
) -> None:
    """Imprime las estadisticas del paso 2 en consola."""

    print("\nPaso 2: media y desviacion estandar muestral por dimension")
    print("Rasgo\tMedia\tDesviacion estandar")

    for estadistica in estadisticas:
        nombre_rasgo = str(estadistica["rasgo"])
        media = float(estadistica["media"])
        desviacion_estandar = float(estadistica["desviacion_estandar"])
        print(f"{nombre_rasgo}\t{media:.2f}\t{desviacion_estandar:.2f}")


def generar_valor_normal_limitado(
    media: float,
    desviacion_estandar: float,
    generador_aleatorio: random.Random,
) -> float:
    """Genera un valor normal y repite el intento si sale del rango 1 a 5."""

    if desviacion_estandar == 0:
        return round(media, 2)

    while True:
        valor_generado = generador_aleatorio.gauss(media, desviacion_estandar)
        if VALOR_MINIMO <= valor_generado <= VALOR_MAXIMO:
            return round(valor_generado, 2)


def generar_perfiles_sinteticos(
    estadisticas: list[dict[str, object]],
    cantidad_perfiles: int = PERFILES_SINTETICOS_ESPERADOS,
    semilla: int = SEMILLA_ALEATORIA,
) -> list[list[float]]:
    """Genera perfiles independientes con una normal por cada dimension."""

    generador_aleatorio = random.Random(semilla)
    perfiles: list[list[float]] = []

    for _ in range(cantidad_perfiles):
        perfil = [
            generar_valor_normal_limitado(
                media=float(estadistica["media"]),
                desviacion_estandar=float(estadistica["desviacion_estandar"]),
                generador_aleatorio=generador_aleatorio,
            )
            for estadistica in estadisticas
        ]
        perfiles.append(perfil)

    return perfiles


def imprimir_perfiles_sinteticos(perfiles: list[list[float]]) -> None:
    """Imprime los perfiles sintéticos en consola sin guardarlos en archivos."""

    print(
        f"\nPaso 3: perfiles sinteticos con distribucion normal "
        f"({len(perfiles)} x {len(NOMBRES_RASGOS)})"
    )
    print("id_perfil\t" + "\t".join(NOMBRES_RASGOS))

    for numero_perfil, perfil in enumerate(perfiles, start=1):
        valores = "\t".join(f"{valor:.2f}" for valor in perfil)
        print(f"{numero_perfil}\t{valores}")


def guardar_matriz_sintetica(
    perfiles: list[list[float]],
    ruta_salida: Path,
) -> None:
    """Guarda la matriz 349 x 5 en un CSV para su verificacion."""

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(NOMBRES_RASGOS)
        escritor.writerows(perfiles)

    print(f"\nMatriz guardada en: {ruta_salida.resolve()}")


def imprimir_matriz(matriz: list[list[object]]) -> None:
    """Imprime en consola la matriz numerica y su correspondencia."""

    print(
        f"\nMatriz numerica de promedios Big Five "
        f"({len(matriz)} x {len(NOMBRES_RASGOS)}):"
    )
    print("\t".join(NOMBRES_RASGOS))
    for fila in matriz:
        promedios = fila[2:]
        print("\t".join(f"{promedio:.2f}" for promedio in promedios))

    print("\nCorrespondencia de cada fila:")
    print("\t".join(ENCABEZADOS_MATRIZ[:2]))
    for fila in matriz:
        print("\t".join(str(valor) for valor in fila[:2]))


def leer_argumentos() -> argparse.Namespace:
    """Lee la ruta opcional del archivo de respuestas."""

    analizador = argparse.ArgumentParser(
        description="Calcula los promedios de los cinco rasgos Big Five."
    )
    analizador.add_argument(
        "--entrada",
        type=Path,
        default=RUTA_ENTRADA_PREDETERMINADA,
        help="Ruta del cuestionario CSV.",
    )
    analizador.add_argument(
        "--salida",
        type=Path,
        default=RUTA_SALIDA_PREDETERMINADA,
        help="Ruta del CSV con la matriz sintetica.",
    )
    return analizador.parse_args()


def ejecutar() -> None:
    """Ejecuta los pasos del procesamiento."""

    # Paso 1: leer la ruta del cuestionario.
    argumentos = leer_argumentos()

    # Paso 1: convertir respuestas y calcular los cinco promedios.
    matriz = calcular_promedios_cinco_grandes(argumentos.entrada)

    # Paso 2: calcular la media y la desviacion estandar de cada dimension.
    estadisticas = calcular_estadisticas_dimensiones(matriz)

    # Paso 3: generar perfiles sinteticos y mostrarlos en consola.
    perfiles = generar_perfiles_sinteticos(estadisticas)

    # Paso 4: mostrar y guardar la matriz sintetica para su verificacion.
    imprimir_matriz(matriz)
    imprimir_estadisticas_dimensiones(estadisticas)
    imprimir_perfiles_sinteticos(perfiles)
    guardar_matriz_sintetica(perfiles, argumentos.salida)


if __name__ == "__main__":
    ejecutar()
