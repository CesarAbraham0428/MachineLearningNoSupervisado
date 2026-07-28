"""
Gestión del estado de la sesión de la aplicación.
"""


class EstadoSesion:
    def __init__(self):
        self.datos = None
        self.modelo_actual = None
        self.resultados = None

    def reiniciar_estado(self):
        """Reinicia los datos almacenados en el estado de la sesión."""
        self.datos = None
        self.modelo_actual = None
        self.resultados = None
