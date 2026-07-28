"""
Manejo de conexión y transacciones con SQLite / base de datos.
"""
import sqlite3


class BaseDeDatos:
    def __init__(self, ruta_db: str = "database/app.db"):
        self.ruta_db = ruta_db

    def obtener_conexion(self):
        """Retorna una conexión activa a la base de datos."""
        return sqlite3.connect(self.ruta_db)
