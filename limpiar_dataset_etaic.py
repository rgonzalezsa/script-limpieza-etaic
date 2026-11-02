#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Limpieza, Validación Estricta, Detección de Datos No Fiables y Auditoría.
Dataset: «Encuesta sobre vida universitaria 2026» (ETAIC 2026-27).

Arquitectura (Deep Modules & Thermo-Nuclear Standards):
  - Deep Modules: Interfaz compacta `clean(df_raw, df_dict) -> CleaningResult`.
  - Code Judo: Vectorización pura en C/NumPy (`np.select`), sin bucles celda a celda.
  - File Size Constraint: Mantenido bajo el límite estricto de 1.000 líneas.
  - Clasificación en dos niveles: FieldStatus y RecordStatus con jerarquía determinista.
  - Preservación íntegra de registros originales y auditoría atómica.
"""

from __future__ import annotations

import argparse
from enum import Enum
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np
import pandas as pd


# ============================================================================
# ENUMERACIONES Y CONTRATOS DE DATOS INMUTABLES
# ============================================================================

class FieldStatus(str, Enum):
    VALIDO = "VALIDO"
    AUSENTE = "AUSENTE"
    INVALIDO = "INVALIDO"
    SOSPECHOSO = "SOSPECHOSO"
    NO_APLICABLE = "NO_APLICABLE"


class RecordStatus(str, Enum):
    VALIDO = "VALIDO"
    INCOMPLETO = "INCOMPLETO"
    EN_REVISION = "EN_REVISION"
    NO_FIABLE = "NO_FIABLE"
    DUPLICADO = "DUPLICADO"


class Severity(str, Enum):
    CRITICA = "CRITICA"          # Imposibilidad física/lógica -> Exclusión (NO_FIABLE)
    ADVERTENCIA = "ADVERTENCIA"  # Anomalía o incoherencia sospechosa -> EN_REVISION
    INFORMATIVA = "INFORMATIVA"  # Inconsistencia plausible o ausencia de campo


@dataclass(frozen=True)
class ValidationRule:
    regla_id: str
    nombre: str
    variables: tuple[str, ...]
    severidad: Severity
    condicion: str
    descripcion: str
    accion: str
    excluye_registro: bool = False


@dataclass(frozen=True)
class AuditEntry:
    id: str
    variable: str
    valor_original: Any
    valor_nuevo: Any
    regla: str
    severidad: str = "INFORMATIVA"


@dataclass(frozen=True)
class InconsistencyAlert:
    regla_id: str
    comprobacion: str
    severidad: str
    n_casos: int
    ids: str
    afecta_estado_global: bool
    excluye_analisis_principal: bool


@dataclass(frozen=True)
class RecordClassification:
    id: str
    estado_global: RecordStatus
    motivos: str
    n_invalidos: int
    n_sospechosos: int
    n_ausentes: int
    n_no_aplicables: int
    excluido: bool
    recomendacion: str


@dataclass(frozen=True)
class CleaningConfig:
    recuperar_nota_x10: bool = False
    excluir_en_revision: bool = False
    fecha_min: pd.Timestamp = pd.Timestamp("2026-10-05")
    fecha_max: pd.Timestamp = pd.Timestamp("2026-10-20")

    columnas_obligatorias: tuple[str, ...] = (
        "id", "fecha_respuesta", "titulacion", "curso", "edad",
        "genero", "vivienda", "beca", "trabaja_horas", "desplaz_min",
        "asistencia_pct", "nota_media", "mov_modo"
    )

    validos: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "titulacion": ("INF", "ADET"),
            "genero": ("Hombre", "Mujer", "Otro / No responde"),
            "vivienda": ("Casa familiar", "Piso compartido", "Residencia", "Otra"),
            "beca": ("Sí", "No"),
            "sue_hora_acostarse": ("Antes de las 23:00", "23:00-00:59", "01:00-02:59", "A partir de las 03:00"),
            "sue_siesta": ("Sí", "No"),
            "tra_tipo": ("Hostelería", "Comercio", "Clases particulares", "Informática / tecnología", "Otro"),
            "tra_relacion_carrera": ("Sí", "Parcial", "No"),
            "tra_turno": ("Mañana", "Tarde", "Noche", "Fines de semana", "Variable"),
            "tra_motivo": ("Económico", "Experiencia", "Ambos"),
            "ia_herramienta": ("ChatGPT", "Copilot", "Gemini", "Claude", "Otra", "Ninguna"),
            "ia_tarea": ("Programar", "Redactar", "Resumir", "Estudiar / explicar", "Traducir", "Ninguna"),
            "ia_declara_uso": ("Siempre", "A veces", "Nunca", "No aplica"),
            "ia_conoce_politica": ("Sí", "No"),
            "dig_movil_clase": ("Nunca", "A veces", "Casi siempre"),
            "dig_notificaciones": ("Ninguna", "Algunas", "Todas"),
            "dig_plataforma": ("Instagram", "TikTok", "YouTube", "WhatsApp", "X (Twitter)", "Otra"),
            "fis_tipo_ejercicio": ("Gimnasio", "Deporte de equipo", "Correr / caminar", "Otro", "Ninguno"),
            "fis_desayuna": ("Siempre", "A veces", "Nunca"),
            "eco_ahorra": ("Sí", "No"),
            "mov_modo": ("A pie", "Bici / patinete", "Bus urbano", "Coche", "Tren / autobús interurbano"),
            "mov_horario_preferido": ("Mañana", "Tarde", "Indiferente"),
            "soc_voluntariado": ("Sí", "No"),
            "fut_practicas": ("Sí", "Lo he planeado", "No"),
            "fut_ingles": ("A2", "B1", "B2", "C1"),
            "fut_postgrado": ("Sí", "No", "No sé"),
            "fut_perfil": ("Técnico", "Gestión", "Investigación", "Emprendedor", "Sin definir"),
        }
    )

    limites_fisicos: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "edad": (16.0, 90.0), "curso": (1.0, 4.0), "asistencia_pct": (0.0, 100.0),
            "nota_media": (0.0, 10.0), "sue_horas_semana": (0.01, 24.0), "sue_horas_finde": (0.01, 24.0),
            "sue_pantalla_antes_min": (0.0, 1440.0), "dig_pantalla_h": (0.0, 24.0),
            "dig_redes_h": (0.0, 24.0), "dig_uso_nocturno_min": (0.0, 1440.0),
            "fis_dias_ejercicio": (0.0, 7.0), "mov_tiempo_campus_h": (0.0, 24.0),
            "mov_horas_libres_dia": (0.0, 24.0), "tra_faltas_mes": (0.0, 31.0),
            "mov_retrasos_mes": (0.0, 31.0), "trabaja_horas": (0.0, 112.0),
            "tra_horas_estudio_sem": (0.0, 112.0), "fis_comidas_fuera_sem": (0.0, 35.0),
            "fis_comida_rapida_sem": (0.0, 35.0),
        }
    )

    rangos_operativos: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "edad": (17.0, 50.0), "trabaja_horas": (0.0, 60.0), "desplaz_min": (0.0, 240.0),
            "sue_horas_semana": (3.0, 12.0), "sue_horas_finde": (3.0, 14.0),
            "sue_pantalla_antes_min": (0.0, 300.0), "sue_cafeina_tazas": (0.0, 15.0),
            "sue_despertares": (0.0, 10.0), "tra_horas_estudio_sem": (0.0, 80.0),
            "est_entregas_simult": (0.0, 20.0), "est_semanas_pico": (0.0, 20.0),
            "est_horas_estudio_examen": (0.0, 100.0), "fis_bebidas_energeticas_sem": (0.0, 30.0),
            "fis_hidratacion_l": (0.0, 10.0), "eco_ingresos_mes": (0.0, 5000.0),
            "eco_gasto_mes": (0.0, 5000.0), "eco_coste_vivienda": (0.0, 3000.0),
            "eco_gasto_ocio": (0.0, 3000.0), "eco_ayuda_familiar": (0.0, 3000.0),
            "eco_gasto_transporte": (0.0, 1000.0), "mov_biblioteca_sem": (0.0, 21.0),
            "soc_asociaciones": (0.0, 20.0), "soc_amigos_clase": (0.0, 100.0),
            "soc_actividades_campus": (0.0, 50.0), "soc_eventos": (0.0, 50.0),
            "fut_salario_esperado": (10000.0, 100000.0), "fut_certificaciones": (0.0, 20.0),
        }
    )


@dataclass(frozen=True)
class CleaningResult:
    datos_limpios: pd.DataFrame
    datos_excluidos: pd.DataFrame
    clasificacion_registros: pd.DataFrame
    registro_cambios: pd.DataFrame
    avisos_coherencia: pd.DataFrame
    valores_vacios: pd.DataFrame
    resumen: pd.DataFrame
    reglas_aplicadas: pd.DataFrame


# ============================================================================
# AUDITORÍA Y HELPERS CANÓNICOS
# ============================================================================

class AuditLedger:
    """Acumulador atómico vectorizado de eventos de limpieza."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(self, id_: str, variable: str, before: Any, after: Any, rule: str, severity: str = "INFORMATIVA") -> None:
        self._entries.append(AuditEntry(str(id_), variable, before, after, rule, severity))

    def record_batch(
        self, ids: Sequence[str] | pd.Series, variable: str,
        before: Sequence[Any] | pd.Series, after: Sequence[Any] | pd.Series | Any,
        rule: str, severity: str = "INFORMATIVA",
    ) -> None:
        id_list = list(ids)
        before_list = list(before)
        after_list = list(after) if isinstance(after, (pd.Series, list, tuple)) else [after] * len(id_list)
        for i_v, b_v, a_v in zip(id_list, before_list, after_list):
            self._entries.append(AuditEntry(str(i_v), variable, b_v, a_v, rule, severity))

    def to_dataframe(self) -> pd.DataFrame:
        cols = ["id", "variable", "valor_original", "valor_nuevo", "regla", "severidad"]
        return pd.DataFrame([e.__dict__ for e in self._entries]) if self._entries else pd.DataFrame(columns=cols)


def normalizar_clave(texto: Any) -> str:
    if pd.isna(texto):
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip().casefold()


def extraer_rangos_diccionario(df_dic: pd.DataFrame) -> dict[str, tuple[float, float]]:
    escalas: dict[str, tuple[float, float]] = {}
    if "Tipo" not in df_dic.columns or "Variable" not in df_dic.columns:
        return escalas
    patron = re.compile(r"Escala\s+(\d+)-(\d+)", re.IGNORECASE)
    for _, fila in df_dic.iterrows():
        m = patron.search(str(fila.get("Tipo", "")))
        if m:
            var = str(fila.get("Variable", "")).strip()
            if var:
                escalas[var] = (float(m.group(1)), float(m.group(2)))
    return escalas


def parsear_fechas_hibridas(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    res = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    num = pd.to_numeric(series, errors="coerce")
    m_excel = num.notna() & (num >= 1000) & (num <= 100000)
    if m_excel.any():
        res.loc[m_excel] = pd.to_datetime(num.loc[m_excel], unit="D", origin="1899-12-30")
    m_txt = (~m_excel) & series.notna()
    if m_txt.any():
        # shortcut: format="mixed" para pandas 2.0+ con compatibilidad hacia atrás
        try:
            res.loc[m_txt] = pd.to_datetime(series.loc[m_txt], errors="coerce", dayfirst=True, format="mixed")
        except (TypeError, ValueError):
            res.loc[m_txt] = pd.to_datetime(series.loc[m_txt], errors="coerce", dayfirst=True)
    return res


# ============================================================================
# ACUMULADOR UNIFICADO DE HALLAZGOS Y AUDITORÍA (CODE JUDO)
# ============================================================================

class Findings:
    """Acumulador atómico de hallazgos, severidades, contadores y motivos por registro."""

    def __init__(self, index: pd.Index) -> None:
        self.index = index
        self._motivos: list[set[str]] = [set() for _ in range(len(index))]
        self.mask_dup = pd.Series(False, index=index)
        self.mask_crit = pd.Series(False, index=index)
        self.mask_susp = pd.Series(False, index=index)
        self.mask_incomp = pd.Series(False, index=index)
        self.n_invalidos = pd.Series(0, index=index)
        self.n_sospechosos = pd.Series(0, index=index)
        self.n_ausentes = pd.Series(0, index=index)
        self.n_no_aplicables = pd.Series(0, index=index)

    def flag_critico(self, mask: pd.Series, motivo: str) -> None:
        m = mask.fillna(False).astype(bool)
        if m.any():
            active_idx = m.index[m]
            self.mask_crit.loc[active_idx] = True
            self.n_invalidos.loc[active_idx] += 1
            for i in active_idx:
                self._motivos[i].add(motivo)

    def flag_sospechoso(self, mask: pd.Series, motivo: str) -> None:
        m = mask.fillna(False).astype(bool)
        if m.any():
            active_idx = m.index[m]
            self.mask_susp.loc[active_idx] = True
            self.n_sospechosos.loc[active_idx] += 1
            for i in active_idx:
                self._motivos[i].add(motivo)

    def flag_incompleto(self, mask: pd.Series, motivo: str) -> None:
        m = mask.fillna(False).astype(bool)
        if m.any():
            active_idx = m.index[m]
            self.mask_incomp.loc[active_idx] = True
            self.n_ausentes.loc[active_idx] += 1
            for i in active_idx:
                self._motivos[i].add(motivo)

    def flag_duplicado(self, mask: pd.Series, motivo: str) -> None:
        m = mask.fillna(False).astype(bool)
        if m.any():
            active_idx = m.index[m]
            self.mask_dup.loc[active_idx] = True
            for i in active_idx:
                self._motivos[i].add(motivo)

    def flag_aviso(self, mask: pd.Series, motivo: str) -> None:
        m = mask.fillna(False).astype(bool)
        if m.any():
            for i in m.index[m]:
                self._motivos[i].add(motivo)

    def add_no_aplicable(self, mask: pd.Series) -> None:
        m = mask.fillna(False).astype(bool)
        if m.any():
            self.n_no_aplicables.loc[m.index[m]] += 1

    def motivos_series(self) -> list[str]:
        return ["; ".join(sorted(m)) if m else "Conforme a reglas" for m in self._motivos]


# ============================================================================
# MOTOR CENTRAL DE CALIDAD, VALIDACIÓN Y EXCLUSIÓN (DEEP MODULE)
# ============================================================================

class DatasetCleaner:
    """Módulo profundo de validación estricta y limpieza del dataset ETAIC."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config: Final[CleaningConfig] = config or CleaningConfig()

    def clean(self, df_raw: pd.DataFrame, df_dict: pd.DataFrame | None = None) -> CleaningResult:
        if "id" not in df_raw.columns:
            raise KeyError("La columna obligatoria 'id' no está presente en el dataset.")
        if df_raw.empty:
            return self._resultado_vacio(df_raw)

        df = df_raw.copy().reset_index(drop=True)
        df["id"] = df["id"].astype(str).str.strip()
        df_original = df.copy()

        ledger = AuditLedger()
        findings = Findings(index=df.index)

        # 1. Normalización de categorías canónicas
        df = self._limpiar_texto(df, ledger, findings)

        # 2. Detección y gestión de duplicados
        df = self._gestionar_duplicados(df, ledger, findings)

        # 3. Validación de límites físicos, rangos operativos y fechas
        df = self._validar_limites_y_fechas(df, df_dict, ledger, findings)

        # 4. Evaluación de obligatoriedad y preguntas condicionales
        self._evaluar_condicionales(df, findings)

        # 5. Auditoría de coherencia inter-variable
        df_alertas, df_catalogo = self._evaluar_coherencia(df, findings)

        # 6. Clasificación determinista con np.select (Code Judo)
        condiciones = [findings.mask_dup, findings.mask_crit, findings.mask_susp, findings.mask_incomp]
        elecciones = [
            RecordStatus.DUPLICADO.value,
            RecordStatus.NO_FIABLE.value,
            RecordStatus.EN_REVISION.value,
            RecordStatus.INCOMPLETO.value,
        ]
        estado_global = np.select(condiciones, elecciones, default=RecordStatus.VALIDO.value)

        excluido = np.isin(estado_global, [RecordStatus.DUPLICADO.value, RecordStatus.NO_FIABLE.value])
        if self.config.excluir_en_revision:
            excluido |= (estado_global == RecordStatus.EN_REVISION.value)

        recom_map = {
            RecordStatus.DUPLICADO.value: "Descartado: fila duplicada",
            RecordStatus.NO_FIABLE.value: "Excluido: contiene anomalías físicas o lógicas críticas",
            RecordStatus.EN_REVISION.value: "Revisión manual requerida (anomalías atípicas no críticas)",
            RecordStatus.INCOMPLETO.value: "Admitido con reserva: faltan datos obligatorios para ciertos análisis",
            RecordStatus.VALIDO.value: "Admitido plenamente para análisis estadístico principal",
        }

        df_clasif = pd.DataFrame({
            "id": df["id"],
            "estado_global": estado_global,
            "motivos": findings.motivos_series(),
            "n_invalidos": findings.n_invalidos,
            "n_sospechosos": findings.n_sospechosos,
            "n_ausentes": findings.n_ausentes,
            "n_no_aplicables": findings.n_no_aplicables,
            "excluido": excluido,
            "recomendacion": [recom_map[s] for s in estado_global],
        })

        # Partición en admitidos y excluidos
        df_limpios = df.loc[~excluido].copy().reset_index(drop=True)
        df_excluidos = df_original.loc[excluido].copy()
        df_excluidos.insert(0, "estado_global", df_clasif.loc[excluido, "estado_global"].values)
        df_excluidos.insert(1, "motivo_exclusion", df_clasif.loc[excluido, "motivos"].values)
        df_excluidos = df_excluidos.reset_index(drop=True)

        resumen, valores_vacios = self._construir_reportes(df_original, df_limpios, df_excluidos, df_clasif, ledger)

        return CleaningResult(
            datos_limpios=df_limpios, datos_excluidos=df_excluidos,
            clasificacion_registros=df_clasif, registro_cambios=ledger.to_dataframe(),
            avisos_coherencia=df_alertas, valores_vacios=valores_vacios,
            resumen=resumen, reglas_aplicadas=df_catalogo,
        )

    def _limpiar_texto(
        self, df: pd.DataFrame, ledger: AuditLedger, findings: Findings
    ) -> pd.DataFrame:
        for col, validos in self.config.validos.items():
            if col not in df.columns:
                continue
            s_orig = df[col]
            mapa = {normalizar_clave(v): v for v in validos}
            canon = s_orig.map(normalizar_clave).map(mapa)

            m_val = s_orig.notna()
            m_inv = m_val & canon.isna()
            m_mod = m_val & canon.notna() & (s_orig != canon)

            if m_inv.any():
                ledger.record_batch(df.loc[m_inv, "id"], col, s_orig.loc[m_inv], np.nan, "Categoría no válida -> vacío", "ADVERTENCIA")
                findings.flag_sospechoso(m_inv, f"Categoría no válida en '{col}'")
                df.loc[m_inv, col] = np.nan

            if m_mod.any():
                ledger.record_batch(df.loc[m_mod, "id"], col, s_orig.loc[m_mod], canon.loc[m_mod], "Normalizar mayúsculas/acentos/espacios", "INFORMATIVA")
                df.loc[m_mod, col] = canon.loc[m_mod]
        return df

    def _gestionar_duplicados(
        self, df: pd.DataFrame, ledger: AuditLedger, findings: Findings
    ) -> pd.DataFrame:
        m_exactos = df.duplicated(keep="first")
        if m_exactos.any():
            findings.flag_duplicado(m_exactos, "Duplicado exacto de fila completa")
            ledger.record_batch(df.loc[m_exactos, "id"], "(fila completa)", "fila duplicada idéntica", "marcada como duplicada", "Duplicado exacto", "CRITICA")

        df_sin = df.loc[~m_exactos]
        m_id_dup = df_sin["id"].duplicated(keep=False)
        if m_id_dup.any():
            for r_id in df_sin.loc[m_id_dup, "id"].unique():
                sub = df_sin[df_sin["id"] == r_id]
                comp = sub.notna().sum(axis=1)
                best_idx = comp.idxmax()
                for idx in sub.index:
                    if idx != best_idx:
                        findings.flag_duplicado(pd.Series([True], index=[idx]), f"Colisión de ID con datos discrepantes (descartada por menor completitud: {comp[idx]} vs {comp[best_idx]})")
                        ledger.record(str(r_id), "id", "id repetido con datos discrepantes", "descartado por menor completitud", "Id duplicado discrepante", "CRITICA")
        return df

    def _validar_limites_y_fechas(
        self, df: pd.DataFrame, df_dict: pd.DataFrame | None, ledger: AuditLedger,
        findings: Findings
    ) -> pd.DataFrame:
        escalas = extraer_rangos_diccionario(df_dict) if df_dict is not None else {}
        for col, (min_e, max_e) in escalas.items():
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                bad = s.notna() & ((s < min_e) | (s > max_e))
                if bad.any():
                    ledger.record_batch(df.loc[bad, "id"], col, s.loc[bad], np.nan, f"Valor fuera de escala psicométrica [{min_e}, {max_e}]", "CRITICA")
                    findings.flag_critico(bad, f"Escala psicométrica imposible en '{col}'")
                    df.loc[bad, col] = np.nan

        for col, (min_f, max_f) in self.config.limites_fisicos.items():
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            bad_txt = df[col].notna() & s.isna()
            if bad_txt.any():
                ledger.record_batch(df.loc[bad_txt, "id"], col, df.loc[bad_txt, col], np.nan, "Texto no numérico en campo cuantitativo", "CRITICA")
                findings.flag_critico(bad_txt, f"Error de tipo/formato en '{col}'")
                df.loc[bad_txt, col] = np.nan

            if col == "nota_media" and self.config.recuperar_nota_x10:
                m_div = s.notna() & (s > 10.0) & (s <= 100.0)
                if m_div.any():
                    rec = (s.loc[m_div] / 10.0).round(2)
                    ledger.record_batch(df.loc[m_div, "id"], col, s.loc[m_div], rec, "Nota >10 recuperada por error de coma decimal (÷10)", "INFORMATIVA")
                    df.loc[m_div, col] = rec
                    s.loc[m_div] = rec

            bad_phys = s.notna() & ((s < min_f) | (s > max_f))
            if bad_phys.any():
                tag = f"[{int(min_f) if min_f.is_integer() else min_f}, {int(max_f) if max_f.is_integer() else max_f}]"
                ledger.record_batch(df.loc[bad_phys, "id"], col, s.loc[bad_phys], np.nan, f"Valor físicamente imposible fuera de {tag}", "CRITICA")
                findings.flag_critico(bad_phys, f"Valor físicamente imposible en '{col}'")
                s.loc[bad_phys] = np.nan

            df[col] = s

        for col, (min_o, max_o) in self.config.rangos_operativos.items():
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                atyp = s.notna() & ((s < min_o) | (s > max_o))
                if atyp.any():
                    tag = f"[{int(min_o) if min_o.is_integer() else min_o}, {int(max_o) if max_o.is_integer() else max_o}]"
                    ledger.record_batch(df.loc[atyp, "id"], col, s.loc[atyp], s.loc[atyp], f"Valor atípico fuera de rango operativo {tag}", "ADVERTENCIA")
                    findings.flag_sospechoso(atyp, f"Valor atípico en '{col}'")

        if "fecha_respuesta" in df.columns:
            f = parsear_fechas_hibridas(df["fecha_respuesta"])
            presentes = df["fecha_respuesta"].notna()
            bad_f = presentes & (f.isna() | (f < self.config.fecha_min) | (f > self.config.fecha_max))
            if bad_f.any():
                periodo_str = f"{self.config.fecha_min.strftime('%d/%m')}–{self.config.fecha_max.strftime('%d/%m/%Y')}"
                ledger.record_batch(df.loc[bad_f, "id"], "fecha_respuesta", df.loc[bad_f, "fecha_respuesta"], pd.NaT, f"Fecha fuera del periodo lectivo oficial ({periodo_str})", "CRITICA")
                findings.flag_critico(bad_f, "Fecha fuera de periodo oficial")
                f.loc[bad_f] = pd.NaT
            df["fecha_respuesta"] = f

        return df

    def _evaluar_condicionales(
        self, df: pd.DataFrame, findings: Findings
    ) -> None:
        for col in self.config.columnas_obligatorias:
            if col in df.columns:
                nulls = df[col].isna()
                if nulls.any():
                    findings.flag_incompleto(nulls, f"Falta campo obligatorio '{col}'")

        # Registro de preguntas legítimamente no aplicables (sin duplicar reglas de coherencia)
        if "trabaja_horas" in df.columns:
            th = pd.to_numeric(df["trabaja_horas"], errors="coerce")
            cols_tra = ("tra_tipo", "tra_relacion_carrera", "tra_turno", "tra_motivo", "tra_flexibilidad", "tra_faltas_mes", "tra_conciliacion")
            for c in cols_tra:
                if c in df.columns:
                    findings.add_no_aplicable((th == 0.0) & df[c].isna())

        if "mov_modo" in df.columns and "mov_aparcar_facil" in df.columns:
            mm = df["mov_modo"]
            ma = pd.to_numeric(df["mov_aparcar_facil"], errors="coerce")
            findings.add_no_aplicable(mm.notna() & (mm != "Coche") & ma.isna())

        if "fis_dias_ejercicio" in df.columns and "fis_tipo_ejercicio" in df.columns:
            fe = pd.to_numeric(df["fis_dias_ejercicio"], errors="coerce")
            te = df["fis_tipo_ejercicio"]
            findings.add_no_aplicable((fe == 0.0) & ((te == "Ninguno") | te.isna()))

    def _evaluar_coherencia(
        self, df: pd.DataFrame, findings: Findings
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Catálogo declarativo y evaluación vectorizada uniforme (Code Judo)."""
        num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
        txt = lambda c: df[c] if c in df.columns else pd.Series(np.nan, index=df.index)

        redes, pant, sue_s, sue_f = num("dig_redes_h"), num("dig_pantalla_h"), num("sue_horas_semana"), num("sue_horas_finde")
        g_o, g_v, g_t, g_m = num("eco_gasto_ocio"), num("eco_coste_vivienda"), num("eco_gasto_transporte"), num("eco_gasto_mes")
        ing, ayu, th, te = num("eco_ingresos_mes"), num("eco_ayuda_familiar"), num("trabaja_horas"), num("tra_horas_estudio_sem")
        tt, mm, ma = txt("tra_tipo"), txt("mov_modo"), num("mov_aparcar_facil")
        ih, ifreq, idcl = txt("ia_herramienta"), num("ia_frecuencia"), txt("ia_declara_uso")

        specs: list[tuple[ValidationRule, pd.Series]] = [
            (ValidationRule("COH_DIG_01", "Redes superan pantalla total", ("dig_redes_h", "dig_pantalla_h"), Severity.CRITICA,
                            "dig_redes_h > dig_pantalla_h", "Horas en redes sociales exceden el tiempo total de pantalla diaria",
                            "Exclusión global del registro (NO_FIABLE)", True),
             redes.notna() & pant.notna() & (redes > pant)),
            (ValidationRule("COH_DIG_02", "Sueño + pantalla excede día", ("dig_pantalla_h", "sue_horas_semana"), Severity.CRITICA,
                            "(dig_pantalla_h + sue_horas_semana) > 24", "La suma diaria de sueño y pantalla supera las 24 horas del día",
                            "Exclusión global del registro (NO_FIABLE)", True),
             pant.notna() & sue_s.notna() & ((pant + sue_s) > 24.0)),
            (ValidationRule("COH_ECO_01", "Desglose excede gasto total", ("eco_gasto_ocio", "eco_coste_vivienda", "eco_gasto_transporte", "eco_gasto_mes"), Severity.CRITICA,
                            "(eco_gasto_ocio + eco_coste_vivienda + eco_gasto_transporte) > eco_gasto_mes", "La suma de vivienda, ocio y transporte supera el gasto mensual total",
                            "Exclusión global del registro (NO_FIABLE)", True),
             g_o.notna() & g_v.notna() & g_t.notna() & g_m.notna() & ((g_o + g_v + g_t) > g_m)),
            (ValidationRule("COH_TRA_01", "Trabaja con tipo de empleo ausente", ("trabaja_horas", "tra_tipo"), Severity.INFORMATIVA,
                            "(trabaja_horas > 0) & tra_tipo.isna()", "Declara horas de trabajo pero no especifica el sector laboral o tipo de empleo",
                            "Marcar como INCOMPLETO", False),
             (th > 0.0) & tt.isna()),
            (ValidationRule("COH_TRA_02", "No trabaja pero declara tipo de empleo", ("trabaja_horas", "tra_tipo"), Severity.ADVERTENCIA,
                            "(trabaja_horas == 0) & tra_tipo.notna()", "Declara no trabajar (0 h) pero cumplimenta detalles del sector laboral",
                            "Marcar como SOSPECHOSO (EN_REVISION)", False),
             (th == 0.0) & tt.notna()),
            (ValidationRule("COH_TRA_03", "Datos laborales sin horas de trabajo", ("trabaja_horas", "tra_tipo"), Severity.INFORMATIVA,
                            "trabaja_horas.isna() & tra_tipo.notna()", "Cumplimenta detalles laborales pero omite las horas semanales trabajadas",
                            "Marcar como INCOMPLETO", False),
             th.isna() & tt.notna()),
            (ValidationRule("COH_MOV_01", "Conductor sin valorar aparcamiento", ("mov_modo", "mov_aparcar_facil"), Severity.INFORMATIVA,
                            "(mov_modo == 'Coche') & mov_aparcar_facil.isna()", "Se desplaza en coche pero omite la valoración de facilidad para aparcar",
                            "Marcar como INCOMPLETO", False),
             (mm == "Coche") & ma.isna()),
            (ValidationRule("COH_MOV_02", "Valora aparcamiento sin usar coche", ("mov_modo", "mov_aparcar_facil"), Severity.ADVERTENCIA,
                            "mov_modo.notna() & (mov_modo != 'Coche') & mov_aparcar_facil.notna()", "Responde sobre la facilidad de aparcar sin acudir en coche al campus",
                            "Marcar como SOSPECHOSO (EN_REVISION)", False),
             mm.notna() & (mm != "Coche") & ma.notna()),
            (ValidationRule("COH_IA_01", "IA Ninguna con uso activo", ("ia_herramienta", "ia_frecuencia"), Severity.ADVERTENCIA,
                            "(ia_herramienta == 'Ninguna') & (ia_frecuencia > 1)", "Declara no usar herramientas de IA pero reporta frecuencia de uso activa",
                            "Marcar como SOSPECHOSO (EN_REVISION)", False),
             (ih == "Ninguna") & (ifreq > 1.0)),
            (ValidationRule("COH_IA_02", "Contradicción frecuencia y declaración IA", ("ia_frecuencia", "ia_declara_uso"), Severity.ADVERTENCIA,
                            "(ia_frecuencia == 1) & (ia_declara_uso == 'Siempre')", "Declara no usar nunca IA pero afirma declarar su uso siempre en trabajos",
                            "Marcar como SOSPECHOSO (EN_REVISION)", False),
             (ifreq == 1.0) & (idcl == "Siempre")),
            (ValidationRule("COH_TRA_04", "Carga extrema trabajo + estudio", ("trabaja_horas", "tra_horas_estudio_sem"), Severity.ADVERTENCIA,
                            "(trabaja_horas + tra_horas_estudio_sem) > 112", "Carga semanal combinada supera 16 horas diarias de dedicación",
                            "Marcar como SOSPECHOSO (EN_REVISION)", False),
             th.notna() & te.notna() & ((th + te) > 112.0)),
            (ValidationRule("COH_ECO_02", "Gasto supera ingresos declarados", ("eco_gasto_mes", "eco_ingresos_mes", "eco_ayuda_familiar"), Severity.INFORMATIVA,
                            "eco_gasto_mes > (eco_ingresos_mes + eco_ayuda_familiar + 1)", "Gasto mensual superior a ingresos y ayudas familiares (financiamiento por ahorro o crédito)",
                            "Aviso analítico (No excluye registro)", False),
             g_m.notna() & ing.notna() & ayu.notna() & (g_m > (ing + ayu + 1.0))),
            (ValidationRule("COH_SUE_01", "Menor descanso en fin de semana", ("sue_horas_finde", "sue_horas_semana"), Severity.INFORMATIVA,
                            "sue_horas_finde < (sue_horas_semana - 1)", "Duerme más de 1 hora menos en fin de semana (plausible por empleo nocturno u ocio)",
                            "Aviso analítico (No excluye registro)", False),
             sue_s.notna() & sue_f.notna() & (sue_f < (sue_s - 1.0))),
        ]

        alertas: list[InconsistencyAlert] = []
        reglas_cat: list[ValidationRule] = []

        for rule, mask in specs:
            reglas_cat.append(rule)
            casos_idx = df.index[mask].tolist()
            if casos_idx:
                ids_trunc = ", ".join([str(df.at[i, "id"]) for i in casos_idx[:15]]) + (" ..." if len(casos_idx) > 15 else "")
                afecta_global = (rule.severidad in (Severity.CRITICA, Severity.ADVERTENCIA)) or (rule.accion == "Marcar como INCOMPLETO")
                alertas.append(InconsistencyAlert(rule.regla_id, rule.descripcion, rule.severidad.value, len(casos_idx), ids_trunc, afecta_global, rule.excluye_registro))
                if rule.severidad == Severity.CRITICA:
                    findings.flag_critico(mask, rule.descripcion)
                elif rule.severidad == Severity.ADVERTENCIA:
                    findings.flag_sospechoso(mask, rule.descripcion)
                elif rule.accion == "Marcar como INCOMPLETO":
                    findings.flag_incompleto(mask, rule.descripcion)
                else:
                    findings.flag_aviso(mask, rule.descripcion)

        df_alertas = pd.DataFrame([a.__dict__ for a in alertas]) if alertas else pd.DataFrame(
            columns=["regla_id", "comprobacion", "severidad", "n_casos", "ids", "afecta_estado_global", "excluye_analisis_principal"]
        )
        return df_alertas, pd.DataFrame([r.__dict__ for r in reglas_cat])

    def _construir_reportes(
        self, df_orig: pd.DataFrame, df_limp: pd.DataFrame, df_excl: pd.DataFrame,
        df_clas: pd.DataFrame, ledger: AuditLedger
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        n_tot, n_adm, n_exc = len(df_orig), len(df_limp), len(df_excl)
        counts = df_clas["estado_global"].value_counts().to_dict()
        tasa = round((n_adm / n_tot) * 100.0, 2) if n_tot else 0.0

        resumen = pd.DataFrame({
            "concepto": [
                "Total registros evaluados", "Registros VÁLIDOS (completos y conformes)",
                "Registros INCOMPLETOS (ausencias no críticas)", "Registros EN REVISIÓN (atípicos / advertencias)",
                "Registros NO FIABLES (excluidos por anomalía crítica)", "Registros DUPLICADOS (excluidos por redundancia)",
                "Total registros admitidos (Datos limpios)", "Total registros excluidos (Datos excluidos)",
                "Tasa de fiabilidad global del dataset (%)", "Celdas vacías en dataset de entrada",
                "Celdas vacías en conjunto de datos limpios", "Total eventos registrados en auditoría",
                "Política de recuperación de nota >10", "Política de exclusión de registros en revisión",
            ],
            "valor": [
                str(n_tot), str(counts.get(RecordStatus.VALIDO.value, 0)), str(counts.get(RecordStatus.INCOMPLETO.value, 0)),
                str(counts.get(RecordStatus.EN_REVISION.value, 0)), str(counts.get(RecordStatus.NO_FIABLE.value, 0)),
                str(counts.get(RecordStatus.DUPLICADO.value, 0)), str(n_adm), str(n_exc), f"{tasa}%",
                str(int(df_orig.isna().sum().sum())), str(int(df_limp.isna().sum().sum())), str(len(ledger.to_dataframe())),
                "Activa (÷10)" if self.config.recuperar_nota_x10 else "Inactiva (anulada a NaN)",
                "Excluidos" if self.config.excluir_en_revision else "Admitidos con marca",
            ]
        })

        vacios = (df_limp.isna().mean() * 100.0).round(1).rename("pct_vacios").reset_index().rename(columns={"index": "variable"})
        vacios["n_vacios"] = df_limp.isna().sum().values
        vacios["es_obligatoria"] = vacios["variable"].isin(self.config.columnas_obligatorias)
        return resumen, vacios.sort_values("pct_vacios", ascending=False).reset_index(drop=True)

    def _resultado_vacio(self, df_raw: pd.DataFrame) -> CleaningResult:
        empty = pd.DataFrame(columns=df_raw.columns)
        return CleaningResult(
            datos_limpios=empty.copy(), datos_excluidos=empty.copy(),
            clasificacion_registros=pd.DataFrame(columns=["id", "estado_global", "motivos", "n_invalidos", "n_sospechosos", "n_ausentes", "n_no_aplicables", "excluido", "recomendacion"]),
            registro_cambios=AuditLedger().to_dataframe(),
            avisos_coherencia=pd.DataFrame(columns=["regla_id", "comprobacion", "severidad", "n_casos", "ids", "afecta_estado_global", "excluye_analisis_principal"]),
            valores_vacios=pd.DataFrame(columns=["variable", "n_vacios", "pct_vacios", "es_obligatoria"]),
            resumen=pd.DataFrame(columns=["concepto", "valor"]),
            reglas_aplicadas=pd.DataFrame(columns=["regla_id", "nombre", "variables", "severidad", "condicion", "descripcion", "accion", "excluye_registro"]),
        )


# ============================================================================
# EXPORTADOR Y CAPA DE INFRAESTRUCTURA (EXCEL / CSV)
# ============================================================================

class ExcelReportWriter:
    """Exportador con formato profesional openpyxl y soporte CSV."""

    @staticmethod
    def export(result: CleaningResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = output_dir / "ETAIC_Dataset_LIMPIO.xlsx"

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            result.datos_limpios.to_excel(writer, sheet_name="Datos_limpios", index=False)
            result.datos_excluidos.to_excel(writer, sheet_name="Datos_excluidos", index=False)
            result.clasificacion_registros.to_excel(writer, sheet_name="Clasificacion_registros", index=False)
            result.resumen.to_excel(writer, sheet_name="Resumen", index=False)
            result.reglas_aplicadas.to_excel(writer, sheet_name="Reglas_aplicadas", index=False)
            result.registro_cambios.to_excel(writer, sheet_name="Registro_cambios", index=False)
            result.avisos_coherencia.to_excel(writer, sheet_name="Avisos_coherencia", index=False)
            result.valores_vacios.to_excel(writer, sheet_name="Valores_vacios", index=False)

            for ws in writer.book.worksheets:
                ws.freeze_panes = "B2"
                for col_cells in ws.columns:
                    max_len = max(len(str(c.value)) if c.value is not None else 0 for c in list(col_cells)[:60])
                    ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 3, 11), 60)

        result.datos_limpios.to_csv(output_dir / "ETAIC_Dataset_LIMPIO.csv", index=False, encoding="utf-8-sig")
        result.datos_excluidos.to_csv(output_dir / "ETAIC_Dataset_EXCLUIDOS.csv", index=False, encoding="utf-8-sig")
        result.clasificacion_registros.to_csv(output_dir / "clasificacion_registros.csv", index=False, encoding="utf-8-sig")
        result.registro_cambios.to_csv(output_dir / "registro_cambios.csv", index=False, encoding="utf-8-sig")


# ============================================================================
# INTERFAZ DE LÍNEA DE COMANDOS (CLI)
# ============================================================================

def resolver_ruta_entrada(ruta: Path) -> Path | None:
    if ruta.exists():
        return ruta
    for c in [Path("excel") / ruta.name, Path("excel") / "ETAIC_Dataset_VidaUniversitaria2026.xlsx", Path("ETAIC_Dataset_VidaUniversitaria2026.xlsx")]:
        if c.exists():
            return c
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pipeline de validación estricta y limpieza ETAIC 2026.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--entrada", type=Path, default=Path("ETAIC_Dataset_VidaUniversitaria2026.xlsx"), help="Ruta al archivo Excel de origen.")
    p.add_argument("--salida-dir", type=Path, default=Path("salida_limpieza"), help="Directorio destino para exportaciones.")
    p.add_argument("--recuperar-nota-x10", action="store_true", default=False, help="Recupera notas entre 10 y 100 dividiendo por 10.")
    p.add_argument("--excluir-en-revision", action="store_true", default=False, help="Excluye también registros en revisión.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    input_path = resolver_ruta_entrada(args.entrada)
    if input_path is None:
        sys.stderr.write(f"Error crítico: Archivo no encontrado en '{args.entrada}'.\n")
        return 1

    print(f"Iniciando pipeline de calidad sobre: {input_path.resolve()}")
    try:
        df_raw = pd.read_excel(input_path, sheet_name="Datos")
        try:
            df_dict = pd.read_excel(input_path, sheet_name="Diccionario")
        except Exception as exc_dict:
            sys.stderr.write(f"Aviso: No se pudo cargar la hoja 'Diccionario' ({exc_dict}); se omitirá la validación de escalas dinámicas.\n")
            df_dict = None
    except Exception as exc:
        sys.stderr.write(f"Error crítico de lectura Excel: {exc}\n")
        return 1

    cleaner = DatasetCleaner(config=CleaningConfig(recuperar_nota_x10=args.recuperar_nota_x10, excluir_en_revision=args.excluir_en_revision))
    resultado = cleaner.clean(df_raw=df_raw, df_dict=df_dict)

    print(f"Exportando resultados a: {args.salida_dir.resolve()} ...")
    ExcelReportWriter.export(resultado, output_dir=args.salida_dir)

    print("\n" + "=" * 70)
    print("RESUMEN DE EJECUCIÓN Y CALIDAD DE DATOS")
    print("=" * 70)
    print(resultado.resumen.to_string(index=False))
    print("\nAVISOS DE INCOHERENCIA DETECTADOS:")
    if not resultado.avisos_coherencia.empty:
        print(resultado.avisos_coherencia[["regla_id", "comprobacion", "severidad", "n_casos"]].to_string(index=False))
    print(f"\n[✓] Proceso completado exitosamente en: {args.salida_dir.resolve()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
