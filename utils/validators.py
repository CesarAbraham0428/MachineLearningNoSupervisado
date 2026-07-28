"""Funciones de validación de datos y parámetros."""

from __future__ import annotations

import pandas as pd

from services.dataset_service import ErrorDatos, ServicioConjuntoDatos


def validar_conjunto_datos(df: pd.DataFrame) -> bool:
    """Indica si el dataset puede utilizarse en el análisis Big Five."""
    try:
        ServicioConjuntoDatos().preprocesar(df)
    except (ErrorDatos, TypeError, ValueError):
        return False
    return True


def obtener_error_conjunto_datos(df: pd.DataFrame) -> str | None:
    """Devuelve un mensaje descriptivo cuando el dataset no es compatible."""
    try:
        ServicioConjuntoDatos().preprocesar(df)
    except (ErrorDatos, TypeError, ValueError) as error:
        return str(error)
    return None


def validar_parametros_entrenamiento(parametros: dict) -> bool:
    """Valida el número de clústeres requerido por K-Means."""
    if not isinstance(parametros, dict):
        return False
    numero_clusters = parametros.get("numero_clusters")
    return isinstance(numero_clusters, int) and numero_clusters >= 2
