#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selección y análisis de las variables de capital humano y empleabilidad (ETAIC · UPSA).

Entrada: dataset limpio generado por `limpiar_dataset_etaic.py`.
Salida: subconjunto con 1 columna identificadora + 9 variables de estudio, Tabla I,
tabla de correlaciones (con n) y Fig. 1. Todos los estadísticos y valores p se calculan
con scipy.stats; nada se escribe a mano.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from scipy import stats

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
class Contraste:
    descripcion: str
    estadistico: str  # "t", "χ²" o "r"
    valor: float
    gl: int
    p: float
    n: int
    diferencia: float | None = None  # diferencia de medias (solo pruebas t)


# ============================================================================
# CONTRASTES (scipy.stats)
# ============================================================================

def _t(df: pd.DataFrame, col: str, grupo: str, a: object, b: object, desc: str) -> Contraste:
    """t de Student para muestras independientes (varianzas iguales, como declara el artículo)."""
    x = df.loc[df[grupo] == a, col].dropna()
    y = df.loc[df[grupo] == b, col].dropna()
    res = stats.ttest_ind(x, y)
    return Contraste(desc, "t", float(res.statistic), len(x) + len(y) - 2, float(res.pvalue),
                     len(x) + len(y), float(x.mean() - y.mean()))


def _chi2(df: pd.DataFrame, fila: str, col: str, desc: str) -> Contraste:
    tabla = pd.crosstab(df[fila], df[col])
    chi2, p, gl, _ = stats.chi2_contingency(tabla)  # sin corrección de Yates (tabla 5×2)
    return Contraste(desc, "χ²", float(chi2), int(gl), float(p), int(tabla.to_numpy().sum()))


def _r(df: pd.DataFrame, x: str, y: str, desc: str) -> Contraste:
    """Pearson con exclusión por pares: cada correlación usa su propio n."""
    sub = df[[x, y]].dropna()
    res = stats.pearsonr(sub[x], sub[y])
    return Contraste(desc, "r", float(res.statistic), len(sub) - 2, float(res.pvalue), len(sub))


def compute_inferential_stats(df: pd.DataFrame) -> dict[str, Contraste]:
    return {
        "perfil_titulacion": _chi2(df, "fut_perfil", "titulacion", "Perfil profesional × titulación"),
        "salario_titulacion": _t(df, "fut_salario_esperado", "titulacion", "INF", "ADET",
                                 "Salario esperado: INF vs ADET"),
        "confianza_practicas": _t(df, "fut_confianza_empleo", "fut_practicas", "Sí", "No",
                                  "Confianza laboral: prácticas Sí vs No"),
        "confianza_curso": _t(df, "fut_confianza_empleo", "curso", 4, 1, "Confianza laboral: 4.º vs 1.º"),
        "r_nota_confianza": _r(df, "nota_media", "fut_confianza_empleo", "Nota media y confianza laboral"),
        "r_curso_confianza": _r(df, "curso", "fut_confianza_empleo", "Curso académico y confianza laboral"),
        "r_cert_salario": _r(df, "fut_certificaciones", "fut_salario_esperado",
                             "Certificaciones y salario esperado"),
        "r_nota_salario": _r(df, "nota_media", "fut_salario_esperado", "Nota media y salario esperado"),
    }


# ============================================================================
# FORMATO (coma decimal, estilo del artículo)
# ============================================================================

def fmt_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else f"= {p:.3f}".replace(".", ",")


def fmt_num(v: float, dec: int = 3) -> str:
    return f"{v:.{dec}f}".replace(".", ",")


def fmt_miles(v: float) -> str:
    return f"{v:,.0f}".replace(",", ".")


# ============================================================================
# TABLAS
# ============================================================================

def build_tabla_1(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla descriptiva cruzada por titulación (Tabla I del artículo)."""
    filas: list[dict[str, str]] = []
    grupos = {"INF": df["titulacion"] == "INF", "ADET": df["titulacion"] == "ADET"}

    def add_cat(nombre: str, col: str, orden: list) -> None:
        filas.append({"Variable": f"{nombre} [n (%)]", "INF": "", "ADET": "", "Total": ""})
        ct = pd.crosstab(df[col], df["titulacion"])
        ct["Total"] = ct.sum(axis=1)
        pct = ct / ct.sum() * 100
        for val in orden:
            filas.append({"Variable": f"  — {val}",
                          **{g: f"{ct.loc[val, g]} ({fmt_num(pct.loc[val, g], 1)} %)" for g in ("INF", "ADET", "Total")}})

    def add_num(nombre: str, col: str, moneda: bool = False) -> None:
        f = (lambda v: f"{fmt_miles(v)} €") if moneda else (lambda v: fmt_num(v, 2))
        series = {g: df.loc[m, col].dropna() for g, m in grupos.items()} | {"Total": df[col].dropna()}
        filas.append({"Variable": f"{nombre} (M ± DT)", **{g: f"{f(s.mean())} ± {f(s.std())}" for g, s in series.items()}})
        filas.append({"Variable": "  (Mediana [Q1 – Q3])",
                      **{g: f"{f(s.median())} [{f(s.quantile(.25))} – {f(s.quantile(.75))}]" for g, s in series.items()}})

    add_cat("Curso", "curso", [1, 2, 3, 4])
    add_num("Nota media del expediente", "nota_media")
    add_cat("Perfil profesional pretendido", "fut_perfil", ["Técnico", "Gestión", "Emprendedor", "Investigación", "Sin definir"])
    add_cat("Prácticas en empresa", "fut_practicas", ["Sí", "Lo he planeado", "No"])
    add_cat("Nivel de inglés declarado", "fut_ingles", ["A2", "B1", "B2", "C1"])
    add_num("Certificaciones profesionales", "fut_certificaciones")
    add_num("Salario bruto anual esperado", "fut_salario_esperado", moneda=True)
    add_num("Confianza en encontrar empleo (1-5)", "fut_confianza_empleo")
    return pd.DataFrame(filas)


def build_tabla_correlaciones(contrastes: dict[str, Contraste]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Variables relacionadas": c.descripcion, "n": c.n, "r de Pearson": fmt_num(c.valor), "p": fmt_p(c.p).removeprefix("= ")}
        for c in contrastes.values() if c.estadistico == "r"
    ])


# ============================================================================
# FIGURA 1 (SVG sin dependencias gráficas)
# ============================================================================

def _panel(x0: int, titulo: str, barras: list[tuple[str, float, int, str]], escala_max: float,
           ticks: list[tuple[float, str]], valor_fmt, pie: str) -> str:
    """Un panel de barras: cada barra es (etiqueta, media, n, color). Alto útil: 220 px."""
    alto = lambda v: v / escala_max * 220.0
    partes = [f'<g transform="translate({x0}, 60)">',
              f'<text x="175" y="15" text-anchor="middle" font-size="13" font-weight="bold" fill="#334155">{titulo}</text>']
    for v, etiqueta in ticks:
        y = 280 - alto(v)
        partes.append(f'<line x1="50" y1="{y:.1f}" x2="310" y2="{y:.1f}" stroke="#e2e8f0" stroke-dasharray="3"/>'
                      f'<text x="42" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#64748b">{etiqueta}</text>')
    partes.append('<line x1="50" y1="280" x2="310" y2="280" stroke="#475569" stroke-width="1.5"/>'
                  '<line x1="50" y1="30" x2="50" y2="280" stroke="#475569" stroke-width="1.5"/>')
    ancho = 240 / len(barras)
    for i, (etiqueta, media, n, color) in enumerate(barras):
        cx, h = 60 + ancho * i + ancho / 2, alto(media)
        partes.append(
            f'<rect x="{cx - 28:.1f}" y="{280 - h:.1f}" width="56" height="{h:.1f}" fill="{color}" rx="3"/>'
            f'<text x="{cx:.1f}" y="{272 - h:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">{valor_fmt(media)}</text>'
            f'<text x="{cx:.1f}" y="298" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">{etiqueta}</text>'
            f'<text x="{cx:.1f}" y="312" text-anchor="middle" font-size="10" fill="#64748b">(n = {fmt_miles(n)})</text>')
    partes.append(f'<text x="175" y="338" text-anchor="middle" font-size="10" font-style="italic" fill="#475569">{pie}</text></g>')
    return "\n".join(partes)


def _barras(df: pd.DataFrame, col: str, grupo: str, niveles: list[tuple[object, str, str]]) -> list[tuple[str, float, int, str]]:
    out = []
    for nivel, etiqueta, color in niveles:
        s = df.loc[df[grupo] == nivel, col].dropna()
        out.append((etiqueta, float(s.mean()), len(s), color))
    return out


def generate_figure_1_svg(df: pd.DataFrame, contrastes: dict[str, Contraste], out_path: Path) -> None:
    sal, conf = contrastes["salario_titulacion"], contrastes["confianza_practicas"]
    panel_a = _panel(
        40, "(a) Salario bruto anual esperado (€/año)",
        _barras(df, "fut_salario_esperado", "titulacion",
                [("INF", "Informática", "#1d4ed8"), ("ADET", "ADET", "#0f766e")]),
        30000.0, [(10000, "10.000 €"), (20000, "20.000 €"), (30000, "30.000 €")],
        lambda v: f"{fmt_miles(v)} €",
        f"Diferencia de medias: {fmt_miles(sal.diferencia)} € (t({sal.gl}) = {fmt_num(sal.valor, 2)}; p {fmt_p(sal.p)})".replace("&", "&amp;").replace("<", "&lt;"))
    panel_b = _panel(
        420, "(b) Confianza en encontrar empleo (1–5)",
        _barras(df, "fut_confianza_empleo", "fut_practicas",
                [("No", "No", "#94a3b8"), ("Lo he planeado", "Planeado", "#cbd5e1"), ("Sí", "Sí", "#d97706")]),
        5.0, [(2.5, "2,5"), (3.75, "3,75"), (5.0, "5,0")],
        lambda v: fmt_num(v, 2),
        f"Diferencia de medias Sí − No: {fmt_num(conf.diferencia, 2)} (t({conf.gl}) = {fmt_num(conf.valor, 2)}; p {fmt_p(conf.p)})".replace("<", "&lt;"))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 420" width="100%" height="100%" '
           f'style="background:#ffffff; font-family:Helvetica, Arial, sans-serif;">\n'
           f'<text x="400" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">'
           f'Expectativa salarial y confianza laboral (N = {fmt_miles(len(df))})</text>\n'
           f'{panel_a}\n{panel_b}\n</svg>')
    out_path.write_text(svg, encoding="utf-8")


def generate_paper_figures_svg(df: pd.DataFrame, out_dir: Path) -> None:
    """Genera además las 4 figuras individuales del artículo a partir de la muestra N = 1.179."""
    wrap = lambda body: (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 360" width="100%" height="100%" '
        'style="background:#ffffff; font-family:Helvetica, Arial, sans-serif;">\n' + body + "\n</svg>"
    )
    # Fig 1: Confianza media por titulación
    f1 = _panel(
        40, "Confianza media en encontrar empleo por titulación (1–5)",
        _barras(df, "fut_confianza_empleo", "titulacion",
                [("ADET", "ADET", "#4f7df3"), ("INF", "INF", "#4f7df3")]),
        5.0, [(1.0, "1"), (2.0, "2"), (3.0, "3"), (4.0, "4"), (5.0, "5")],
        lambda v: fmt_num(v, 2), "Titulación",
    ).replace('transform="translate(40, 60)"', 'transform="translate(40, 20)"')
    (out_dir / "figura_1_confianza_titulacion.svg").write_text(wrap(f1), encoding="utf-8")

    # Fig 2: Salario esperado según nivel de inglés
    f2 = _panel(
        40, "Salario bruto anual esperado según nivel de inglés (€)",
        _barras(df, "fut_salario_esperado", "fut_ingles",
                [(k, k, "#4f7df3") for k in ("A2", "B1", "B2", "C1")]),
        30000.0, [(10000, "10.000 €"), (20000, "20.000 €"), (30000, "30.000 €")],
        lambda v: f"{fmt_miles(v)} €", "Nivel de inglés declarado",
    ).replace('transform="translate(40, 60)"', 'transform="translate(40, 20)"')
    (out_dir / "figura_2_salario_ingles.svg").write_text(wrap(f2), encoding="utf-8")

    # Fig 3: Salario esperado según perfil profesional deseado
    f3 = _panel(
        40, "Salario esperado medio según perfil profesional (€)",
        _barras(df, "fut_salario_esperado", "fut_perfil",
                [(k, k, "#4f7df3") for k in ("Emprendedor", "Gestión", "Investigación", "Sin definir", "Técnico")]),
        30000.0, [(10000, "10.000 €"), (20000, "20.000 €"), (30000, "30.000 €")],
        lambda v: f"{fmt_miles(v)} €", "Perfil profesional deseado",
    ).replace('transform="translate(40, 60)"', 'transform="translate(40, 20)"')
    (out_dir / "figura_3_salario_perfil.svg").write_text(wrap(f3), encoding="utf-8")

    # Fig 4: Evolución de la confianza media según curso académico (línea)
    medias_c = [(c, float(df.loc[df["curso"] == c, "fut_confianza_empleo"].dropna().mean()),
                 int(df.loc[df["curso"] == c, "fut_confianza_empleo"].dropna().count())) for c in (1, 2, 3, 4)]
    pts = [(90 + i * 75, 280 - (m - 2.6) / 1.2 * 220) for i, (_, m, _) in enumerate(medias_c)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    grid = "\n".join(
        f'<line x1="60" y1="{280 - (v - 2.6) / 1.2 * 220:.1f}" x2="340" y2="{280 - (v - 2.6) / 1.2 * 220:.1f}" stroke="#e2e8f0" stroke-dasharray="3"/>'
        f'<text x="52" y="{284 - (v - 2.6) / 1.2 * 220:.1f}" text-anchor="end" font-size="10" fill="#64748b">{fmt_num(v, 1)}</text>'
        for v in (2.8, 3.0, 3.2, 3.4, 3.6)
    )
    nodos = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1d4ed8"/>'
        f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">{fmt_num(m, 2)}</text>'
        f'<text x="{x:.1f}" y="298" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">{c}.º</text>'
        f'<text x="{x:.1f}" y="312" text-anchor="middle" font-size="10" fill="#64748b">(n = {n})</text>'
        for (x, y), (c, m, n) in zip(pts, medias_c)
    )
    f4 = (
        '<g transform="translate(30, 20)">'
        '<text x="200" y="15" text-anchor="middle" font-size="13" font-weight="bold" fill="#334155">'
        'Evolución de la confianza media según curso académico (1–5)</text>'
        + grid
        + '<line x1="60" y1="280" x2="340" y2="280" stroke="#475569" stroke-width="1.5"/>'
          '<line x1="60" y1="40" x2="60" y2="280" stroke="#475569" stroke-width="1.5"/>'
        + f'<polyline fill="none" stroke="#1d4ed8" stroke-width="2.5" points="{poly}"/>'
        + nodos
        + '<text x="200" y="335" text-anchor="middle" font-size="10" font-style="italic" fill="#475569">Curso académico</text></g>'
    )
    (out_dir / "figura_4_confianza_curso.svg").write_text(wrap(f4), encoding="utf-8")


# ============================================================================
# ORQUESTACIÓN
# ============================================================================

def resolve_clean_dataset(clean_path: Path | None = None) -> Path:
    """Busca el dataset limpio (CSV o Excel) en la raíz o en salida_limpieza/."""
    candidatos = [clean_path] if clean_path else [
        Path("salida_limpieza/ETAIC_Dataset_LIMPIO.csv"), Path("ETAIC_Dataset_LIMPIO.csv"),
        Path("salida_limpieza/ETAIC_Dataset_LIMPIO.xlsx"), Path("ETAIC_Dataset_LIMPIO.xlsx"),
    ]
    for c in candidatos:
        if c.exists():
            return c
    raise FileNotFoundError("No se encontró el dataset limpio en:\n - " + "\n - ".join(map(str, candidatos)))


def run_analysis(clean_path: Path | None = None, output_dir: Path = Path("salida_limpieza")
                 ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Contraste]]:
    ruta = resolve_clean_dataset(clean_path)
    df = pd.read_excel(ruta) if ruta.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(ruta)
    df = df[COLUMNAS_SELECCIONADAS].copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    contrastes = compute_inferential_stats(df)
    tabla1 = build_tabla_1(df)
    salidas = {
        "ETAIC_Dataset_SELECCION_VARIABLES": df,
        "TABLA_I_RESULTADOS": tabla1,
        "TABLA_CORRELACIONES": build_tabla_correlaciones(contrastes),
    }
    for nombre, tabla in salidas.items():
        tabla.to_csv(output_dir / f"{nombre}.csv", index=False)
        tabla.to_excel(output_dir / f"{nombre}.xlsx", index=False)
    generate_figure_1_svg(df, contrastes, output_dir / "figura_1_resultados.svg")
    generate_paper_figures_svg(df, output_dir)
    return df, tabla1, contrastes


IDS_CONSERVADOS_ATIPICOS: Final[tuple[str, ...]] = ("E0566", "E0802", "E0933", "E1022", "E1130")
IDS_EXCLUIDOS_SALARIO: Final[tuple[str, ...]] = ("E0529", "E0642", "E0687", "E0874", "E0907")


def _self_check(df: pd.DataFrame, contrastes: dict[str, Contraste]) -> None:
    """Valores de referencia del artículo actualizado (N = 1.179: 643 INF y 536 ADET)."""
    assert len(df) == 1179, f"Esperados 1179 casos, hallados {len(df)}"
    assert list(df.columns) == COLUMNAS_SELECCIONADAS
    assert df["id"].is_unique, "Existen identificadores duplicados en la muestra limpia"
    assert int((df["titulacion"] == "INF").sum()) == 643
    assert int((df["titulacion"] == "ADET").sum()) == 536
    ids_presentes = set(df["id"].astype(str))
    assert set(IDS_CONSERVADOS_ATIPICOS).issubset(ids_presentes), "Faltan registros atípicos en sueño/ingresos que debían conservarse"
    assert set(IDS_EXCLUIDOS_SALARIO).isdisjoint(ids_presentes), "Aparecen registros con expectativas salariales extremas que debían excluirse"
    assert round(contrastes["salario_titulacion"].diferencia) == 3742
    assert abs(contrastes["perfil_titulacion"].valor - 352.03) < 0.01
    assert abs(contrastes["r_nota_salario"].p - 0.165) < 0.002
    assert [c.n for k, c in contrastes.items() if k.startswith("r_")] == [1106, 1149, 1103, 1097]


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis de variables seleccionadas (capital humano y empleabilidad)")
    parser.add_argument("--entrada", type=Path, default=None, help="Dataset limpio (.csv o .xlsx). Por defecto se busca solo.")
    parser.add_argument("--salida-dir", type=Path, default=Path("salida_limpieza"))
    args = parser.parse_args()

    df, _, contrastes = run_analysis(args.entrada, args.salida_dir)
    _self_check(df, contrastes)

    print(f"Subconjunto: {len(df)} filas × {len(df.columns)} columnas (id + 9 variables de estudio)")
    print(f"Archivos en '{args.salida_dir}': ETAIC_Dataset_SELECCION_VARIABLES, TABLA_I_RESULTADOS, "
          "TABLA_CORRELACIONES (.csv/.xlsx) y figura_1_resultados.svg\n")
    for c in contrastes.values():
        extra = f"; dif. medias = {c.diferencia:.2f}" if c.diferencia is not None else ""
        print(f"{c.descripcion}: {c.estadistico}({c.gl}) = {c.valor:.3f}, p = {c.p:.3g}, n = {c.n}{extra}")


if __name__ == "__main__":
    main()
