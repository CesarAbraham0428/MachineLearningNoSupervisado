"""Regresión: los filtros de Datos deben gobernar estadísticas y entrenamiento."""

import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from services.dataset_service import DIMENSIONES_BIG_FIVE


def _perfiles_con_categoria() -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{
                columna: [1.5, 2.0, 4.0, 4.5]
                for columna in DIMENSIONES_BIG_FIVE
            },
            "Grupo": ["A", "A", "B", "B"],
        }
    )


def _preparar_app(tab: int = 0, indices: list[int] | None = None) -> AppTest:
    datos = _perfiles_con_categoria()
    app = AppTest.from_file("app.py")
    estado = {
        "dataframe_cargado": datos,
        "dataset_original": datos.loc[:, list(DIMENSIONES_BIG_FIVE)],
        "nombre_archivo": "perfiles_prueba.csv",
        "nombre_archivo_original": "perfiles_prueba.csv",
        "fecha_carga": "04/08/2026",
        "fecha_carga_original": "04/08/2026",
    }
    if indices is not None:
        estado.update(
            {
                "indices_filas_filtradas": indices,
                "firma_filtro_activo": (("Grupo",), "Grupo", "B"),
            }
        )
    if tab:
        opciones = {
            1: ":material/bar_chart: Estadística descriptiva",
            2: ":material/model_training: Entrenamiento",
        }
        estado["tab_activa"] = tab
        estado["main_navigation"] = opciones[tab]
    for clave, valor in estado.items():
        app.session_state[clave] = valor
    return app


class PruebasSincronizacionFiltros(unittest.TestCase):
    def test_filtro_categorico_guarda_indices_e_invalida_modelo(self):
        app = _preparar_app().run(timeout=20)
        app.session_state["modelo_entrenado"] = True
        app.session_state["resultado_entrenamiento"] = object()
        app.session_state["evaluaciones_k"] = (object(),)

        app.selectbox[0].set_value("Grupo").run(timeout=20)
        app.selectbox[1].set_value("B").run(timeout=20)

        self.assertEqual(app.session_state["indices_filas_filtradas"], [2, 3])
        self.assertEqual(app.session_state["firma_filtro_activo"][2], "B")
        self.assertFalse(app.session_state["modelo_entrenado"])
        self.assertIsNone(app.session_state["resultado_entrenamiento"])
        self.assertNotIn("evaluaciones_k", app.session_state)

    def test_estadisticas_priorizan_la_limpieza_del_filtro_actual(self):
        datos = _perfiles_con_categoria()
        firma = (("Extraversión", "Estabilidad emocional"), "Grupo", "A")
        estadisticas = _preparar_app(tab=1)
        estadisticas.session_state["dataframe_filtrado"] = datos
        estadisticas.session_state["dataset_limpio"] = datos.iloc[:2].copy()
        estadisticas.session_state["firma_filtro_activo"] = firma
        estadisticas.session_state["filtro_calidad"] = firma
        estadisticas.run(timeout=20)

        metricas = {metrica.label: metrica.value for metrica in estadisticas.metric}
        self.assertEqual(metricas[":material/table_rows: Perfiles analizados"], "2")
    def test_no_permite_activar_menos_de_dos_rasgos_big_five(self):
        app = _preparar_app().run(timeout=20)
        solo_un_rasgo = [next(iter(DIMENSIONES_BIG_FIVE))]

        app.multiselect[0].set_value(solo_un_rasgo).run(timeout=20)

        activas = app.session_state["dataframe_filtrado"].columns.tolist()
        rasgos_activos = [rasgo for rasgo in DIMENSIONES_BIG_FIVE if rasgo in activas]
        self.assertGreaterEqual(len(rasgos_activos), 2)
        self.assertEqual(len(app.warning), 1)
    def test_estadisticas_y_entrenamiento_respetan_las_columnas_filtradas(self):
        datos = _perfiles_con_categoria()
        columnas = list(DIMENSIONES_BIG_FIVE)[:3]
        filtrado = datos.loc[:, columnas]

        estadisticas = _preparar_app(tab=1)
        estadisticas.session_state["dataframe_filtrado"] = filtrado
        estadisticas.run(timeout=20)
        metricas = {metrica.label: metrica.value for metrica in estadisticas.metric}
        self.assertEqual(metricas[":material/psychology: Rasgos analizados"], "3")

        entrenamiento = _preparar_app(tab=2)
        entrenamiento.session_state["dataframe_filtrado"] = filtrado
        entrenamiento.run(timeout=20)
        self.assertEqual(
            entrenamiento.session_state["dataframe_entrenamiento"].columns.tolist(),
            columnas,
        )
        self.assertFalse(entrenamiento.exception)
    def test_estadisticas_y_entrenamiento_consumen_las_filas_filtradas(self):
        estadisticas = _preparar_app(tab=1, indices=[2, 3]).run(timeout=20)
        metricas = {metrica.label: metrica.value for metrica in estadisticas.metric}
        self.assertEqual(metricas[":material/table_rows: Perfiles analizados"], "2")

        entrenamiento = _preparar_app(tab=2, indices=[2, 3]).run(timeout=20)
        self.assertEqual(len(entrenamiento.session_state["dataframe_entrenamiento"]), 2)
        self.assertEqual(
            entrenamiento.session_state["firma_entrenamiento"][0],
            (("Grupo",), "Grupo", "B"),
        )
        self.assertFalse(entrenamiento.exception)


if __name__ == "__main__":
    unittest.main()
