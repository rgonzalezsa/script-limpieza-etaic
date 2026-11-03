#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suite de Pruebas Automatizadas para el Pipeline de Calidad y Validación ETAIC 2026-27.
Cubre los 21 escenarios requeridos por la especificación formal.
"""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from limpiar_dataset_etaic import (
    CleaningConfig,
    DatasetCleaner,
    ExcelReportWriter,
    RecordStatus,
)


class TestPipelineCalidadEtaic(unittest.TestCase):
    """Batería exhaustiva de pruebas unitarias y de integración."""

    def setUp(self) -> None:
        """Fixture base con una fila canónica y completa."""
        warnings.simplefilter("ignore", ResourceWarning)
        self.fila_base = {
            "id": "E0001",
            "fecha_respuesta": pd.Timestamp("2026-10-10"),
            "titulacion": "INF",
            "curso": 2,
            "edad": 20.0,
            "genero": "Hombre",
            "vivienda": "Piso compartido",
            "beca": "Sí",
            "trabaja_horas": 0.0,
            "desplaz_min": 25.0,
            "asistencia_pct": 90.0,
            "nota_media": 7.5,
            "sue_horas_semana": 7.5,
            "sue_horas_finde": 8.0,
            "sue_hora_acostarse": "23:00-00:59",
            "sue_pantalla_antes_min": 30.0,
            "sue_cafeina_tazas": 2.0,
            "sue_siesta": "No",
            "sue_despertares": 1.0,
            "tra_tipo": np.nan,
            "tra_relacion_carrera": np.nan,
            "tra_turno": np.nan,
            "tra_motivo": np.nan,
            "tra_flexibilidad": np.nan,
            "tra_faltas_mes": np.nan,
            "tra_horas_estudio_sem": 20.0,
            "tra_conciliacion": np.nan,
            "ia_herramienta": "ChatGPT",
            "ia_tarea": "Programar",
            "ia_declara_uso": "A veces",
            "ia_conoce_politica": "Sí",
            "ia_frecuencia": 3.0,
            "dig_pantalla_h": 6.0,
            "dig_redes_h": 2.5,
            "dig_movil_clase": "A veces",
            "dig_notificaciones": "Algunas",
            "dig_plataforma": "Instagram",
            "dig_uso_nocturno_min": 45.0,
            "fis_dias_ejercicio": 3.0,
            "fis_tipo_ejercicio": "Gimnasio",
            "fis_desayuna": "Siempre",
            "fis_comidas_fuera_sem": 4.0,
            "fis_comida_rapida_sem": 1.0,
            "fis_bebidas_energeticas_sem": 0.0,
            "fis_hidratacion_l": 2.0,
            "eco_ingresos_mes": 400.0,
            "eco_gasto_mes": 350.0,
            "eco_coste_vivienda": 200.0,
            "eco_gasto_ocio": 100.0,
            "eco_gasto_transporte": 30.0,
            "eco_ahorra": "Sí",
            "eco_ayuda_familiar": 300.0,
            "mov_modo": "A pie",
            "mov_retrasos_mes": 1.0,
            "mov_tiempo_campus_h": 6.0,
            "mov_biblioteca_sem": 2.0,
            "mov_horas_libres_dia": 4.0,
            "mov_horario_preferido": "Mañana",
            "mov_aparcar_facil": np.nan,
            "soc_asociaciones": 1.0,
            "soc_amigos_clase": 5.0,
            "soc_actividades_campus": 2.0,
            "soc_voluntariado": "No",
            "soc_eventos": 3.0,
            "fut_practicas": "Lo he planeado",
            "fut_salario_esperado": 25000.0,
            "fut_ingles": "B2",
            "fut_certificaciones": 1.0,
            "fut_postgrado": "No sé",
            "fut_perfil": "Técnico",
        }
        self.cleaner = DatasetCleaner(config=CleaningConfig(recuperar_nota_x10=False))

    def _crear_df(self, **modificaciones: Any) -> pd.DataFrame:
        """Crea un DataFrame con una sola fila modificando campos específicos."""
        data = dict(self.fila_base)
        data.update(modificaciones)
        return pd.DataFrame([data])

    # 1. Registro válido y completo
    def test_registro_valido_completo(self) -> None:
        df = self._crear_df()
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.VALIDO.value)
        self.assertFalse(clasif["excluido"])
        self.assertEqual(len(res.datos_limpios), 1)
        self.assertEqual(len(res.datos_excluidos), 0)

    # 2. Valores físicamente imposibles (exclusión global)
    def test_valor_fisicamente_imposible_exclusion_global(self) -> None:
        casos = [
            ("sue_horas_semana", 25.0),
            ("edad", 2.0),
            ("edad", 99.0),
            ("dig_pantalla_h", 25.0),
            ("fis_dias_ejercicio", 8.0),
            ("asistencia_pct", 105.0),
            ("nota_media", 12.0),
            ("sue_horas_semana", 0.0),
        ]
        for var, val_imp in casos:
            with self.subTest(variable=var, valor=val_imp):
                df = self._crear_df(**{var: val_imp})
                res = self.cleaner.clean(df)
                clasif = res.clasificacion_registros.iloc[0]
                self.assertEqual(clasif["estado_global"], RecordStatus.NO_FIABLE.value)
                self.assertTrue(clasif["excluido"])
                self.assertIn("imposible", clasif["motivos"].lower())
                self.assertEqual(len(res.datos_limpios), 0)
                self.assertEqual(len(res.datos_excluidos), 1)

    # 3. Valores fuera del rango de aceptación pero no imposibles
    def test_valor_fuera_rango_aceptacion_no_imposible(self) -> None:
        df = self._crear_df(sue_horas_semana=14.0, eco_ingresos_mes=6000.0)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.EN_REVISION.value)
        self.assertFalse(clasif["excluido"])
        self.assertEqual(len(res.datos_limpios), 1)

    # 4. Valores atípicos pero posibles no deben catalogarse como falsos
    def test_valor_atipico_pero_posible_no_falso(self) -> None:
        df = self._crear_df(sue_horas_semana=16.0)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.EN_REVISION.value)
        self.assertFalse(clasif["excluido"])
        self.assertNotIn("FALSO", clasif["motivos"])

    # 5. Campos obligatorios ausentes
    def test_campo_obligatorio_ausente_incompleto(self) -> None:
        df = self._crear_df(edad=np.nan)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.INCOMPLETO.value)
        self.assertFalse(clasif["excluido"])
        self.assertIn("edad", clasif["motivos"])

    # 6. Campos opcionales ausentes no invalidan el registro
    def test_campo_opcional_ausente(self) -> None:
        df = self._crear_df(soc_asociaciones=np.nan)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.VALIDO.value)

    # 7. Preguntas condicionales correctamente vacías
    def test_preguntas_condicionales_correctamente_vacias(self) -> None:
        df = self._crear_df(trabaja_horas=0.0, tra_tipo=np.nan, mov_modo="A pie", mov_aparcar_facil=np.nan)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.VALIDO.value)
        self.assertGreater(clasif["n_no_aplicables"], 0)

    # 8. Preguntas condicionales incompletas
    def test_preguntas_condicionales_incompletas(self) -> None:
        df = self._crear_df(trabaja_horas=20.0, tra_tipo=np.nan)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.INCOMPLETO.value)
        self.assertIn("laboral", clasif["motivos"].lower())

    # 9. Preguntas condicionales incoherentes (responder sin aplicar)
    def test_preguntas_condicionales_incoherentes(self) -> None:
        df = self._crear_df(mov_modo="A pie", mov_aparcar_facil=4.0)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.EN_REVISION.value)

    # 10. Contradicciones críticas entre variables
    def test_contradicciones_criticas(self) -> None:
        df = self._crear_df(dig_pantalla_h=4.0, dig_redes_h=6.0)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.NO_FIABLE.value)
        self.assertTrue(clasif["excluido"])
        self.assertIn("COH_DIG_01", [a["regla_id"] for a in res.avisos_coherencia.to_dict("records")])

    # 11. Un único error crítico provoca la exclusión global
    def test_unico_error_critico_provoca_exclusion(self) -> None:
        df = self._crear_df(fis_dias_ejercicio=8.0)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.NO_FIABLE.value)
        self.assertTrue(clasif["excluido"])

    # 12. Múltiples errores simultáneos en un mismo registro
    def test_multiples_errores_simultaneos(self) -> None:
        df = self._crear_df(
            sue_horas_semana=25.0,  # Crítico
            edad=np.nan,            # Ausente obligatorio
            sue_pantalla_antes_min=400.0,  # Atípico
        )
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.NO_FIABLE.value)
        self.assertTrue(clasif["excluido"])
        self.assertGreater(clasif["n_invalidos"], 0)
        self.assertGreater(clasif["n_ausentes"], 0)
        self.assertGreater(clasif["n_sospechosos"], 0)

    # 13. Registros que solo deben recibir una advertencia
    def test_registros_solo_advertencia(self) -> None:
        df = self._crear_df(eco_ingresos_mes=6000.0)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.EN_REVISION.value)
        self.assertFalse(clasif["excluido"])

    # 14. Duplicados exactos
    def test_duplicados_exactos(self) -> None:
        f1 = dict(self.fila_base)
        f1["id"] = "E0001"
        f2 = dict(self.fila_base)
        f2["id"] = "E0001"
        df = pd.DataFrame([f1, f2])

        res = self.cleaner.clean(df)
        self.assertEqual(len(res.clasificacion_registros), 2)
        c1 = res.clasificacion_registros.iloc[0]
        c2 = res.clasificacion_registros.iloc[1]
        self.assertEqual(c1["estado_global"], RecordStatus.VALIDO.value)
        self.assertEqual(c2["estado_global"], RecordStatus.DUPLICADO.value)
        self.assertTrue(c2["excluido"])
        self.assertEqual(len(res.datos_limpios), 1)
        self.assertEqual(len(res.datos_excluidos), 1)

    # 15. Identificadores repetidos con datos distintos (política determinista)
    def test_identificadores_repetidos_datos_distintos(self) -> None:
        f1 = dict(self.fila_base)
        f1["id"] = "E0001"
        f1["edad"] = 20.0
        f2 = dict(self.fila_base)
        f2["id"] = "E0001"
        f2["edad"] = np.nan
        df = pd.DataFrame([f1, f2])

        res = self.cleaner.clean(df)
        c1 = res.clasificacion_registros.iloc[0]
        c2 = res.clasificacion_registros.iloc[1]
        self.assertEqual(c1["estado_global"], RecordStatus.VALIDO.value)
        self.assertEqual(c2["estado_global"], RecordStatus.DUPLICADO.value)
        self.assertIn("completitud", c2["motivos"].lower())

    # 16. Fechas inválidas o fuera del periodo permitido
    def test_fechas_invalidas_o_fuera_rango(self) -> None:
        df = self._crear_df(fecha_respuesta=pd.Timestamp("2026-09-01"))
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.NO_FIABLE.value)

    # 17. Escalas obtenidas del diccionario
    def test_escalas_diccionario(self) -> None:
        dic = pd.DataFrame([
            {"Variable": "sue_calidad", "Tipo": "Escala 1-5", "Tema": "Sueño", "Descripción": "Calidad"},
        ])
        df = self._crear_df(sue_calidad=8.0)
        res = self.cleaner.clean(df, df_dict=dic)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.NO_FIABLE.value)

    # 18. Dataset vacío
    def test_dataset_vacio(self) -> None:
        df = pd.DataFrame(columns=["id", "titulacion", "edad"])
        res = self.cleaner.clean(df)
        self.assertEqual(len(res.datos_limpios), 0)
        self.assertEqual(len(res.datos_excluidos), 0)

    # 19. Columnas obligatorias ausentes
    def test_columnas_obligatorias_ausentes(self) -> None:
        df = pd.DataFrame([{"edad": 20}])
        with self.assertRaises(KeyError):
            self.cleaner.clean(df)

    # 20. Conservación de los datos originales
    def test_conservacion_datos_originales(self) -> None:
        df = self._crear_df(edad=99.0)
        res = self.cleaner.clean(df)
        self.assertEqual(res.datos_excluidos["edad"].iloc[0], 99.0)
        self.assertEqual(df["edad"].iloc[0], 99.0)

    # 21. Exportación y coherencia de los informes
    def test_exportacion_y_coherencia_informes(self) -> None:
        df = self._crear_df()
        res = self.cleaner.clean(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            ExcelReportWriter.export(res, out_path)
            xlsx = out_path / "ETAIC_Dataset_LIMPIO.xlsx"
            self.assertTrue(xlsx.exists())
            with pd.ExcelFile(xlsx) as xl:
                hojas_esperadas = {
                    "Datos_limpios", "Datos_excluidos", "Clasificacion_registros",
                    "Resumen", "Reglas_aplicadas", "Registro_cambios",
                    "Avisos_coherencia", "Valores_vacios"
                }
                self.assertTrue(hojas_esperadas.issubset(set(xl.sheet_names)))

    # 22. Regresión: Fecha de respuesta ausente debe dar INCOMPLETO, nunca NO_FIABLE
    def test_fecha_ausente_marca_incompleto_no_excluye(self) -> None:
        df = self._crear_df(fecha_respuesta=np.nan)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.INCOMPLETO.value)
        self.assertFalse(clasif["excluido"])
        self.assertIn("fecha_respuesta", clasif["motivos"])

    # 23. Regresión: NaN en mov_modo no debe activar falsamente 'sin usar coche'
    def test_movilidad_nan_no_etiqueta_sin_usar_coche(self) -> None:
        df = self._crear_df(mov_modo=np.nan, mov_aparcar_facil=4.0)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        # Al faltar mov_modo (obligatorio) se marca INCOMPLETO, pero NO debe decir "sin usar coche"
        self.assertNotIn("sin acudir en coche", clasif["motivos"])
        self.assertNotIn("sin usar coche", clasif["motivos"])

    # 24. Regresión: Coherencia debe alimentar contadores de inconsistencias
    def test_coherencia_incrementa_contadores_invalidos(self) -> None:
        df = self._crear_df(dig_pantalla_h=4.0, dig_redes_h=6.0)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["n_invalidos"], 1)

    # 25. Regresión: Valor corrupto no debe etiquetarse como campo ausente
    def test_valor_corrupto_no_etiqueta_falta_campo_obligatorio(self) -> None:
        df = self._crear_df(nota_media=75.0)
        res = self.cleaner.clean(df)
        clasif = res.clasificacion_registros.iloc[0]
        self.assertEqual(clasif["estado_global"], RecordStatus.NO_FIABLE.value)
        self.assertIn("imposible", clasif["motivos"].lower())
        self.assertNotIn("falta campo obligatorio", clasif["motivos"].lower())


if __name__ == "__main__":
    unittest.main()
