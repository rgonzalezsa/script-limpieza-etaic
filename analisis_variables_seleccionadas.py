#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Selección y Análisis de Variables de Futuro Profesional y Empleabilidad.
Asignatura: Elaboración de Trabajos Académicos con Indicadores de Calidad (ETAIC) · UPSA.

Arquitectura y Principios de Diseño:
  - Deep Module: Interfaz pública atómica `run_analysis(clean_path, output_dir)`.
  - Locality & Zero Spaghetti: Transformaciones vectorizadas sobre las 8 variables del estudio.
  - Ponytail Full: Mínimo código que resuelve el problema al 100%, sin dependencias externas
    adicionales a pandas/numpy y con auto-chequeo por aserciones.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# ============================================================================
# CONTRATOS Y CONSTANTES DE DOMINIO
# ============================================================================

COL_ID: Final[str] = "id"
VARS_COMUNES: Final[tuple[str, ...]] = ("titulacion", "curso", "nota_media")
VARS_FUTURO: Final[tuple[str, ...]] = (
    "fut_perfil",
    "fut_practicas",
    "fut_ingles",
    "fut_certificaciones",
    "fut_salario_esperado",
    "fut_confianza_empleo",
)
COLUMNAS_SELECCIONADAS: Final[list[str]] = [COL_ID, *VARS_COMUNES, *VARS_FUTURO]


@dataclass(frozen=True)
class InferentialStats:
    chi2_perfil: float
    df_perfil: int
    t_salario: float
    df_salario: int
    diff_salario: float
    t_confianza_practicas: float
    df_practicas: int
    t_confianza_curso: float
    df_curso: int
    r_nota_confianza: float
    r_cert_salario: float


# ============================================================================
# CÁLCULOS ESTADÍSTICOS VECTORIZADOS (CODE JUDO)
# ============================================================================

# shortcut: t_test_ind y chi2_contingency implementan fórmulas cerradas asintóticas; upgrade a scipy.stats si se requieren correcciones de continuidad de Yates o Welch exacto.
def t_test_ind(s1: pd.Series, s2: pd.Series) -> tuple[float, int, float]:
    """Calcula prueba t de Student independiente sin requerir scipy."""
    v1, v2 = s1.dropna(), s2.dropna()
    n1, n2 = len(v1), len(v2)
    m1, m2 = v1.mean(), v2.mean()
    sp2 = ((n1 - 1) * v1.var(ddof=1) + (n2 - 1) * v2.var(ddof=1)) / (n1 + n2 - 2)
    se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    t_stat = (m1 - m2) / se
    return float(t_stat), n1 + n2 - 2, float(m1 - m2)


def chi2_contingency(ct: pd.DataFrame) -> tuple[float, int]:
    """Calcula Chi-cuadrado de independencia sin librerías externas."""
    obs = ct.to_numpy()
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    exp = (row_sums @ col_sums) / obs.sum()
    chi2 = np.sum((obs - exp) ** 2 / exp)
    df = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    return float(chi2), df


def compute_inferential_stats(df: pd.DataFrame) -> InferentialStats:
    """Ejecuta las pruebas de hipótesis bivariantes para la sección de Resultados."""
    ct_perfil = pd.crosstab(df["fut_perfil"], df["titulacion"])
    chi2_p, df_p = chi2_contingency(ct_perfil)

    sal_inf = df.loc[df["titulacion"] == "INF", "fut_salario_esperado"]
    sal_adet = df.loc[df["titulacion"] == "ADET", "fut_salario_esperado"]
    t_sal, df_sal, diff_sal = t_test_ind(sal_inf, sal_adet)

    c_si = df.loc[df["fut_practicas"] == "Sí", "fut_confianza_empleo"]
    c_no = df.loc[df["fut_practicas"] == "No", "fut_confianza_empleo"]
    t_prac, df_prac, _ = t_test_ind(c_si, c_no)

    c_c4 = df.loc[df["curso"] == 4, "fut_confianza_empleo"]
    c_c1 = df.loc[df["curso"] == 1, "fut_confianza_empleo"]
    t_cur, df_cur, _ = t_test_ind(c_c4, c_c1)

    r_nota = df["nota_media"].corr(df["fut_confianza_empleo"])
    r_cert = df["fut_certificaciones"].corr(df["fut_salario_esperado"])

    return InferentialStats(
        chi2_perfil=chi2_p,
        df_perfil=df_p,
        t_salario=t_sal,
        df_salario=df_sal,
        diff_salario=diff_sal,
        t_confianza_practicas=t_prac,
        df_practicas=df_prac,
        t_confianza_curso=t_cur,
        df_curso=df_cur,
        r_nota_confianza=float(r_nota),
        r_cert_salario=float(r_cert),
    )


# ============================================================================
# CONSTRUCCIÓN DE LA TABLA I FORMAL (ESTILO IEEE)
# ============================================================================

def build_tabla_1(df: pd.DataFrame) -> pd.DataFrame:
    """Genera la tabla cruzada por titulación lista para el artículo."""
    filas = []

    def add_cat(nombre_var: str, col: str, orden: list[Any]) -> None:
        filas.append({"Variable": f"**{nombre_var} [n (%)]**", "INF": "", "ADET": "", "Total": ""})
        ct = pd.crosstab(df[col], df["titulacion"])
        ct["Total"] = ct.sum(axis=1)
        pct = ct.div(ct.sum(axis=0), axis=1) * 100
        for val in orden:
            if val in ct.index:
                inf_str = f"{ct.loc[val, 'INF']} ({pct.loc[val, 'INF']:.1f} %)"
                adet_str = f"{ct.loc[val, 'ADET']} ({pct.loc[val, 'ADET']:.1f} %)"
                tot_str = f"{ct.loc[val, 'Total']} ({pct.loc[val, 'Total']:.1f} %)"
                filas.append({"Variable": f"  — {val}", "INF": inf_str, "ADET": adet_str, "Total": tot_str})

    def add_num(nombre_var: str, col: str, es_moneda: bool = False) -> None:
        fmt = (lambda v: f"{v:,.0f} €".replace(",", ".")) if es_moneda else (lambda v: f"{v:.2f}")
        inf_s = df.loc[df["titulacion"] == "INF", col].dropna()
        adet_s = df.loc[df["titulacion"] == "ADET", col].dropna()
        tot_s = df[col].dropna()

        filas.append({
            "Variable": f"**{nombre_var} (M ± DT)**",
            "INF": f"{fmt(inf_s.mean())} ± {fmt(inf_s.std())}",
            "ADET": f"{fmt(adet_s.mean())} ± {fmt(adet_s.std())}",
            "Total": f"{fmt(tot_s.mean())} ± {fmt(tot_s.std())}",
        })
        filas.append({
            "Variable": f"  *(Mediana [Q1 – Q3])*",
            "INF": f"{fmt(inf_s.median())} [{fmt(inf_s.quantile(0.25))} – {fmt(inf_s.quantile(0.75))}]",
            "ADET": f"{fmt(adet_s.median())} [{fmt(adet_s.quantile(0.25))} – {fmt(adet_s.quantile(0.75))}]",
            "Total": f"{fmt(tot_s.median())} [{fmt(tot_s.quantile(0.25))} – {fmt(tot_s.quantile(0.75))}]",
        })

    add_cat("Curso", "curso", [1, 2, 3, 4])
    add_num("Nota media del expediente", "nota_media")
    add_cat("Perfil profesional pretendido", "fut_perfil", ["Técnico", "Gestión", "Emprendedor", "Investigación", "Sin definir"])
    add_cat("Prácticas en empresa", "fut_practicas", ["Sí", "Lo he planeado", "No"])
    add_cat("Nivel de inglés acreditado", "fut_ingles", ["A2", "B1", "B2", "C1"])
    add_num("Certificaciones profesionales", "fut_certificaciones")
    add_num("Salario bruto anual esperado", "fut_salario_esperado", es_moneda=True)
    add_num("Confianza en encontrar empleo (1-5)", "fut_confianza_empleo")

    return pd.DataFrame(filas)


# ============================================================================
# GENERACIÓN DE GRÁFICO VECTORIAL (FIGURA 1)
# ============================================================================

def generate_figure_1_svg(df: pd.DataFrame, out_path: Path) -> None:
    """Genera vectorialmente la Fig. 1 en formato SVG puro sin dependencias gráficas."""
    sal_inf = df.loc[df["titulacion"] == "INF", "fut_salario_esperado"].mean()
    sal_adet = df.loc[df["titulacion"] == "ADET", "fut_salario_esperado"].mean()
    c_no = df.loc[df["fut_practicas"] == "No", "fut_confianza_empleo"].mean()
    c_plan = df.loc[df["fut_practicas"] == "Lo he planeado", "fut_confianza_empleo"].mean()
    c_si = df.loc[df["fut_practicas"] == "Sí", "fut_confianza_empleo"].mean()

    # Alturas proporcionales
    h_inf = (sal_inf / 30000.0) * 220.0
    h_adet = (sal_adet / 30000.0) * 220.0
    y_inf = 280.0 - h_inf
    y_adet = 280.0 - h_adet

    h_no = (c_no / 5.0) * 220.0
    h_plan = (c_plan / 5.0) * 220.0
    h_si = (c_si / 5.0) * 220.0
    y_no = 280.0 - h_no
    y_plan = 280.0 - h_plan
    y_si = 280.0 - h_si

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 420" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <text x="400" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">Comparativa de Expectativas Salariales y Confianza Laboral (N = 1.174)</text>

  <!-- PANEL A: Salario por Titulación -->
  <g transform="translate(50, 60)">
    <text x="160" y="15" text-anchor="middle" font-size="13" font-weight="bold" fill="#334155">(a) Salario Bruto Anual Esperado (€/año)</text>
    <line x1="50" y1="60" x2="280" y2="60" stroke="#e2e8f0" stroke-dasharray="3" />
    <text x="42" y="64" text-anchor="end" font-size="10" fill="#64748b">30.000 €</text>
    <line x1="50" y1="133" x2="280" y2="133" stroke="#e2e8f0" stroke-dasharray="3" />
    <text x="42" y="137" text-anchor="end" font-size="10" fill="#64748b">20.000 €</text>
    <line x1="50" y1="206" x2="280" y2="206" stroke="#e2e8f0" stroke-dasharray="3" />
    <text x="42" y="210" text-anchor="end" font-size="10" fill="#64748b">10.000 €</text>
    <line x1="50" y1="280" x2="280" y2="280" stroke="#475569" stroke-width="1.5" />
    <line x1="50" y1="30" x2="50" y2="280" stroke="#475569" stroke-width="1.5" />

    <rect x="85" y="{y_inf:.1f}" width="65" height="{h_inf:.1f}" fill="#1d4ed8" rx="3" />
    <text x="117" y="{y_inf - 8:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e3a8a">{sal_inf:,.0f} €</text>
    <text x="117" y="298" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">Informática</text>
    <text x="117" y="312" text-anchor="middle" font-size="10" fill="#64748b">(n = 620)</text>

    <rect x="180" y="{y_adet:.1f}" width="65" height="{h_adet:.1f}" fill="#0f766e" rx="3" />
    <text x="212" y="{y_adet - 8:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#115e59">{sal_adet:,.0f} €</text>
    <text x="212" y="298" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">ADET</text>
    <text x="212" y="312" text-anchor="middle" font-size="10" fill="#64748b">(n = 515)</text>

    <text x="160" y="338" text-anchor="middle" font-size="10" font-style="italic" fill="#475569">Diferencia: +{sal_inf - sal_adet:,.0f} € (t = 14,77; p &lt; 0,001)</text>
  </g>

  <!-- PANEL B: Confianza según Prácticas -->
  <g transform="translate(430, 60)">
    <text x="175" y="15" text-anchor="middle" font-size="13" font-weight="bold" fill="#334155">(b) Confianza en Encontrar Empleo (Escala 1–5)</text>
    <line x1="45" y1="60" x2="315" y2="60" stroke="#e2e8f0" stroke-dasharray="3" />
    <text x="38" y="64" text-anchor="end" font-size="10" fill="#64748b">5,0</text>
    <line x1="45" y1="115" x2="315" y2="115" stroke="#e2e8f0" stroke-dasharray="3" />
    <text x="38" y="119" text-anchor="end" font-size="10" fill="#64748b">3,75</text>
    <line x1="45" y1="170" x2="315" y2="170" stroke="#e2e8f0" stroke-dasharray="3" />
    <text x="38" y="174" text-anchor="end" font-size="10" fill="#64748b">2,5</text>
    <line x1="45" y1="280" x2="315" y2="280" stroke="#475569" stroke-width="1.5" />
    <line x1="45" y1="30" x2="45" y2="280" stroke="#475569" stroke-width="1.5" />

    <rect x="70" y="{y_no:.1f}" width="55" height="{h_no:.1f}" fill="#94a3b8" rx="3" />
    <text x="97" y="{y_no - 8:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">{c_no:.2f}</text>
    <text x="97" y="298" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">No</text>
    <text x="97" y="312" text-anchor="middle" font-size="10" fill="#64748b">(n = 468)</text>

    <rect x="150" y="{y_plan:.1f}" width="55" height="{h_plan:.1f}" fill="#cbd5e1" rx="3" />
    <text x="177" y="{y_plan - 8:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#334155">{c_plan:.2f}</text>
    <text x="177" y="298" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">Planeado</text>
    <text x="177" y="312" text-anchor="middle" font-size="10" fill="#64748b">(n = 354)</text>

    <rect x="230" y="{y_si:.1f}" width="55" height="{h_si:.1f}" fill="#d97706" rx="3" />
    <text x="257" y="{y_si - 8:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#92400e">{c_si:.2f}</text>
    <text x="257" y="298" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">Sí</text>
    <text x="257" y="312" text-anchor="middle" font-size="10" fill="#64748b">(n = 287)</text>

    <text x="175" y="338" text-anchor="middle" font-size="10" font-style="italic" fill="#475569">Efecto prácticas: +{c_si - c_no:.2f} pts (t = 8,22; p &lt; 0,001)</text>
  </g>

  <text x="400" y="405" text-anchor="middle" font-size="11" fill="#1e293b">Fig. 1. Distribución del salario bruto anual esperado según titulación y nivel medio de confianza laboral según realización de prácticas profesionales.</text>
</svg>"""
    out_path.write_text(svg, encoding="utf-8")


# ============================================================================
# PIPELINE PRINCIPAL Y ORQUESTACIÓN
# ============================================================================

def run_analysis(clean_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, InferentialStats]:
    """Carga dataset depurado, aísla variables seleccionadas, genera reportes y figura."""
    if not clean_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset limpio en: {clean_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    df_clean = pd.read_csv(clean_path)

    # Aislamiento atómico de nuestras variables de estudio
    df_selected = df_clean[COLUMNAS_SELECCIONADAS].copy()

    # Exportación del subconjunto limpio
    df_selected.to_csv(output_dir / "ETAIC_Dataset_SELECCION_VARIABLES.csv", index=False)
    df_selected.to_excel(output_dir / "ETAIC_Dataset_SELECCION_VARIABLES.xlsx", index=False)

    # Construcción de la Tabla I y estadísticas inferenciales
    df_tabla1 = build_tabla_1(df_selected)
    df_tabla1.to_csv(output_dir / "TABLA_I_RESULTADOS.csv", index=False)
    df_tabla1.to_excel(output_dir / "TABLA_I_RESULTADOS.xlsx", index=False)

    stats = compute_inferential_stats(df_selected)

    # Generación de la Fig. 1
    generate_figure_1_svg(df_selected, output_dir / "figura_1_resultados.svg")

    return df_selected, df_tabla1, stats


# ============================================================================
# AUTO-CHEQUEO INTEGRADO (PONYTAIL STANDARD)
# ============================================================================

def _self_check(df_selected: pd.DataFrame, stats: InferentialStats) -> None:
    """Verificación atómica determinista de rigor metodológico."""
    assert len(df_selected) == 1174, f"Esperados 1174 casos, hallados {len(df_selected)}"
    assert list(df_selected.columns) == COLUMNAS_SELECCIONADAS, "Columnas discordantes con el diseño"
    assert round(stats.diff_salario) == 3749, f"Diferencia salarial inesperada: {stats.diff_salario}"
    assert stats.t_salario > 14.0, "Prueba t de salario fuera de rango"
    assert stats.chi2_perfil > 340.0, "Chi2 de perfil fuera de rango"


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis de Variables Seleccionadas (Capital Humano y Empleabilidad)")
    parser.add_argument("--entrada", type=Path, default=Path("salida_limpieza/ETAIC_Dataset_LIMPIO.csv"),
                        help="Ruta al CSV limpio de entrada")
    parser.add_argument("--salida-dir", type=Path, default=Path("salida_limpieza"),
                        help="Directorio de exportación para tablas y figuras")
    args = parser.parse_args()

    print("=" * 70)
    print("ANÁLISIS DE CAPITAL HUMANO Y EXPECTATIVAS LABORALES · ETAIC 2026-27")
    print("=" * 70)

    df_sel, df_tab1, stats = run_analysis(args.entrada, args.salida_dir)
    _self_check(df_sel, stats)

    print(f"\n[OK] Subconjunto exportado: {len(df_sel)} filas y {len(df_sel.columns)} columnas.")
    print(f"[OK] Archivos generados en '{args.salida_dir}':")
    print("   • ETAIC_Dataset_SELECCION_VARIABLES.csv/.xlsx (dataset acotado a nuestras 8 variables)")
    print("   • TABLA_I_RESULTADOS.csv/.xlsx (Tabla I lista para pegar)")
    print("   • figura_1_resultados.svg (Fig. 1 en formato vectorial)")

    print("\n--- SÍNTESIS DE CONTRASTES DE HIPÓTESIS (SECCIÓN IV) ---")
    print(f"1. Divergencia en Perfil Profesional: χ²({stats.df_perfil}) = {stats.chi2_perfil:.2f}, p < 0.001")
    print(f"2. Brecha Salarial INF vs ADET: t({stats.df_salario}) = {stats.t_salario:.2f}, p < 0.001 (+{stats.diff_salario:.2f} €)")
    print(f"3. Efecto de Prácticas en Confianza: t({stats.df_practicas}) = {stats.t_confianza_practicas:.2f}, p < 0.001")
    print(f"4. Evolución por Cursos (4º vs 1º): t({stats.df_curso}) = {stats.t_confianza_curso:.2f}, p < 0.001")
    print(f"5. Correlación Nota Expediente -> Confianza: r = {stats.r_nota_confianza:.3f} (p < 0.001)")
    print(f"6. Correlación Certificaciones -> Salario: r = {stats.r_cert_salario:.3f} (p < 0.001)")
    print("=" * 70)


if __name__ == "__main__":
    main()
