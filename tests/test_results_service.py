"""Pruebas de la preparación de resultados y gráficas de agrupamiento."""

import unittest

import pandas as pd

from services.results_service import ServicioResultados
from services.training_service import ServicioEntrenamiento


def _resultado_de_prueba():
    datos = pd.DataFrame(
        {
            "Extraversión": [1.0, 1.2, 0.8, 1.1, 4.8, 5.0, 4.9, 4.7],
            "Responsabilidad": [1.1, 0.9, 1.0, 1.2, 4.9, 4.7, 5.0, 4.8],
            "Apertura": [1.0, 1.1, 0.9, 1.2, 4.7, 4.8, 5.0, 4.9],
        }
    )
    return ServicioEntrenamiento(random_state=7).entrenar_modelo(
        datos,
        maximo_k=3,
    )


class PruebasServicioResultados(unittest.TestCase):
    def setUp(self):
        self.servicio = ServicioResultados()
        self.resultado = _resultado_de_prueba()

    def test_crea_resumen_con_cantidades_y_porcentajes(self):
        resumen = self.servicio.crear_resumen_grupos(self.resultado)

        self.assertEqual(resumen["Registros"].sum(), 8)
        self.assertAlmostEqual(resumen["Porcentaje"].sum(), 1.0)
        self.assertEqual(len(resumen), self.resultado.k_usado)

    def test_crea_proyeccion_pca_para_registros_y_centros(self):
        proyeccion = self.servicio.crear_proyeccion_pca(self.resultado)

        self.assertEqual(len(proyeccion.puntos), 8)
        self.assertEqual(len(proyeccion.centros), self.resultado.k_usado)
        self.assertEqual(len(proyeccion.varianza_explicada), 2)
        self.assertGreater(sum(proyeccion.varianza_explicada), 0.0)

    def test_crea_tablas_de_centros_y_asignaciones(self):
        centros = self.servicio.crear_tabla_centros(self.resultado)
        asignaciones = self.servicio.crear_tabla_asignaciones(self.resultado)

        self.assertEqual(len(centros), self.resultado.k_usado)
        self.assertIn("Grupo", centros.columns)
        self.assertEqual(len(asignaciones), 8)
        self.assertIn("Identificador", asignaciones.columns)
        self.assertIn("Grupo asignado", asignaciones.columns)

    def test_asignaciones_incluyen_valores_originales_en_el_orden_entrenado(self):
        datos_originales = pd.DataFrame(
            {
                "Nombre": [f"Persona {numero}" for numero in range(1, 9)],
                "Edad": [20, 21, 22, 23, 30, 31, 32, 33],
            },
            index=self.resultado.datos_estandarizados.index,
        )

        asignaciones = self.servicio.crear_tabla_asignaciones(
            self.resultado,
            datos_originales,
        )

        self.assertEqual(
            asignaciones.columns.tolist(),
            ["Identificador", "Nombre", "Edad", "Grupo asignado"],
        )
        self.assertEqual(asignaciones.loc[0, "Nombre"], "Persona 1")
        self.assertEqual(asignaciones.loc[7, "Edad"], 33)
        self.assertTrue(
            asignaciones["Grupo asignado"].str.startswith("Grupo ").all()
        )

    def test_identificador_conserva_la_fila_original_al_entrenar_un_filtro(self):
        datos_filtrados = pd.DataFrame(
            {
                "Extraversión": [1.0, 1.2, 0.8, 4.8, 5.0, 4.9],
                "Responsabilidad": [1.1, 0.9, 1.0, 4.9, 4.7, 5.0],
            },
            index=[2, 3, 4, 8, 9, 10],
        )
        resultado = ServicioEntrenamiento(random_state=7).entrenar_modelo(
            datos_filtrados,
            maximo_k=2,
        )

        asignaciones = self.servicio.crear_tabla_asignaciones(
            resultado,
            datos_filtrados.assign(Nombre=list("ABCDEF")),
        )

        self.assertEqual(
            asignaciones["Identificador"].tolist(),
            [
                "Registro 3",
                "Registro 4",
                "Registro 5",
                "Registro 9",
                "Registro 10",
                "Registro 11",
            ],
        )

    def test_interpreta_una_separacion_debil(self):
        titulo, explicacion = self.servicio.interpretar_silhouette(0.10)

        self.assertEqual(titulo, "Separación débil")
        self.assertIn("cautela", explicacion)


if __name__ == "__main__":
    unittest.main()
