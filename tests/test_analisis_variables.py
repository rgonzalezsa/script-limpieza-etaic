#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pruebas del módulo de análisis de variables seleccionadas (ETAIC 2026-27 · N = 1.179)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analisis_variables_seleccionadas import (
    COLUMNAS_SELECCIONADAS,
    IDS_CONSERVADOS_ATIPICOS,
    IDS_EXCLUIDOS_SALARIO,
    _self_check,
    compute_inferential_stats,
    fmt_p,
    run_analysis,
)

CLEAN_XLSX = Path("salida_limpieza/ETAIC_Dataset_LIMPIO.xlsx")


@unittest.skipUnless(CLEAN_XLSX.exists(), "No se encontró el Excel limpio.")
class TestAnalisisVariables(unittest.TestCase):
    def test_run_analysis_end_to_end_sobre_excel_limpio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            df, _, contrastes = run_analysis(CLEAN_XLSX, out)
            self.assertEqual(len(df), 1179)
            self.assertEqual(list(df.columns), COLUMNAS_SELECCIONADAS)
            self.assertTrue(df["id"].is_unique)
            self.assertEqual(int((df["titulacion"] == "INF").sum()), 643)
            self.assertEqual(int((df["titulacion"] == "ADET").sum()), 536)

            ids = set(df["id"].astype(str))
            self.assertTrue(set(IDS_CONSERVADOS_ATIPICOS).issubset(ids))
            self.assertTrue(set(IDS_EXCLUIDOS_SALARIO).isdisjoint(ids))

            for nombre in (
                "ETAIC_Dataset_SELECCION_VARIABLES.csv",
                "ETAIC_Dataset_SELECCION_VARIABLES.xlsx",
                "TABLA_I_RESULTADOS.csv",
                "TABLA_I_RESULTADOS.xlsx",
                "TABLA_CORRELACIONES.csv",
                "TABLA_CORRELACIONES.xlsx",
                "figura_1_resultados.svg",
                "figura_1_confianza_titulacion.svg",
                "figura_2_salario_ingles.svg",
                "figura_3_salario_perfil.svg",
                "figura_4_confianza_curso.svg",
            ):
                self.assertTrue((out / nombre).exists(), nombre)

            corr = pd.read_csv(out / "TABLA_CORRELACIONES.csv")
            self.assertEqual(list(corr["n"]), [1106, 1149, 1103, 1097])
            svg = (out / "figura_1_resultados.svg").read_text(encoding="utf-8")
            self.assertIn("Diferencia de medias", svg)
            self.assertIn("(n = 623)", svg)
            self.assertIn("(n = 517)", svg)
            _self_check(df, contrastes)

    def test_valores_de_referencia_n1179(self) -> None:
        c = compute_inferential_stats(pd.read_excel(CLEAN_XLSX))
        self.assertAlmostEqual(c["perfil_titulacion"].valor, 352.03, delta=0.01)
        self.assertAlmostEqual(c["salario_titulacion"].valor, 14.79, delta=0.01)
        self.assertAlmostEqual(c["salario_titulacion"].diferencia, 3742.11, delta=0.05)
        self.assertAlmostEqual(c["confianza_practicas"].valor, 8.19, delta=0.01)
        self.assertAlmostEqual(c["confianza_curso"].valor, 8.36, delta=0.01)
        self.assertAlmostEqual(c["r_nota_confianza"].valor, 0.252, delta=0.001)
        self.assertAlmostEqual(c["r_curso_confianza"].valor, 0.246, delta=0.001)
        self.assertAlmostEqual(c["r_cert_salario"].valor, 0.203, delta=0.001)
        self.assertAlmostEqual(c["r_nota_salario"].valor, -0.042, delta=0.001)
        self.assertLess(c["r_nota_confianza"].p, 1e-15)
        self.assertEqual(fmt_p(c["r_nota_salario"].p), "= 0,165")


if __name__ == "__main__":
    unittest.main()
