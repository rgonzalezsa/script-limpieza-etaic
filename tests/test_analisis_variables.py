#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruebas Unitarias para el Módulo de Análisis de Variables Seleccionadas.
Asignatura: ETAIC 2026-27 (UPSA).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analisis_variables_seleccionadas import (
    COLUMNAS_SELECCIONADAS,
    _self_check,
    build_tabla_1,
    compute_inferential_stats,
    run_analysis,
)


class TestAnalisisVariables(unittest.TestCase):
    def setUp(self) -> None:
        self.clean_csv = Path("salida_limpieza/ETAIC_Dataset_LIMPIO.csv")

    def test_run_analysis_end_to_end(self) -> None:
        if not self.clean_csv.exists():
            self.skipTest("No se encontró el dataset limpio.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            df_sel, df_tab1, stats = run_analysis(self.clean_csv, out_path)

            self.assertEqual(len(df_sel), 1174)
            self.assertEqual(list(df_sel.columns), COLUMNAS_SELECCIONADAS)
            self.assertTrue((out_path / "ETAIC_Dataset_SELECCION_VARIABLES.csv").exists())
            self.assertTrue((out_path / "TABLA_I_RESULTADOS.csv").exists())
            self.assertTrue((out_path / "figura_1_resultados.svg").exists())

            # Self check formal
            _self_check(df_sel, stats)

    def test_inferential_stats_consistency(self) -> None:
        if not self.clean_csv.exists():
            self.skipTest("No se encontró el dataset limpio.")

        df = pd.read_csv(self.clean_csv)
        stats = compute_inferential_stats(df)
        self.assertGreater(stats.chi2_perfil, 300.0)
        self.assertGreater(stats.t_salario, 14.0)
        self.assertGreater(stats.t_confianza_practicas, 8.0)
        self.assertAlmostEqual(stats.diff_salario, 3749.30, delta=1.0)


if __name__ == "__main__":
    unittest.main()
