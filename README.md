# Pipeline de Limpieza, Validación y Auditoría de Datos · ETAIC 2026-27

> **Trabajo Final:** Artículo científico breve a partir de datos comunes, revisión por pares y Jornadas ETAIC  
> **Asignatura:** Elaboración de Trabajos Académicos con Indicadores de Calidad (ETAIC)  
> **Institución:** Facultad de Informática · Universidad Pontificia de Salamanca (UPSA)  
> **Formato de Publicación:** Plantilla IEEE (Ingeniería Informática)
> **Dataset Base:** *«Encuesta sobre vida universitaria 2026»* (Datos simulados, S. Alonso Rollán)

---

## 1. Descripción del Proyecto

Este repositorio contiene la limpieza automatizada, validación estricta y auditoría del dataset utilizado para la redacción del artículo científico breve en la asignatura **ETAIC (curso 2026-27)**.

---

## 2. Síntesis de Calidad y Muestra Final (Criterio 4)

El pipeline ejecuta una auditoría atómica determinista sobre las 1.200 respuestas originales:

| Métrica de Muestra | Casos | Impacto Metodológico |
| :--- | :---: | :--- |
| **Muestra de partida ($N_{\text{total}}$)** | **1.200** | Respuestas brutas en `Datos`. |
| **Duplicados exactos eliminados** | **6** | Descartados por redundancia idéntica de fila. |
| **Registros no fiables excluidos** | **10** | Anomalías críticas (incoherencia lógica o límites físicos imposibles). |
| **Muestra analítica final ($N_{\text{limpio}}$)** | **1.184** | **98,67% de fiabilidad global** admitida para el análisis. |
| *— Registros válidos (completos)* | *865* | Datos 100% conformes y sin valores nulos obligatorios. |
| *— Registros incompletos (admitidos)* | *309* | Válidos para análisis por bloques (omisiones no críticas). |
| *— Registros en revisión (advertencias)* | *10* | Valores atípicos pero físicamente plausibles (auditados). |
| **Categorías textuales normalizadas** | **91** | Corrección de erratas sintácticas (`inf` $\rightarrow$ `INF`, etc.). |
| **Eventos de auditoría registrados** | **116** | Trazabilidad celda por celda exportada en bitácora. |

---

## 3. Instalación y Requisitos

El proyecto requiere **Python 3.9 o superior** y tres dependencias estándar de análisis de datos:

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO

# 2. (Opcional) Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate

# 3. Instalar librerías
pip install -r requirements.txt
```

---

## 4. Guía de Ejecución

### A. Ejecutar el Pipeline de Limpieza
Para procesar el archivo Excel original y generar los reportes de auditoría:

```bash
python3 limpiar_dataset_etaic.py --entrada excel/ETAIC_Dataset_VidaUniversitaria2026.xlsx --salida-dir salida_limpieza
```

#### Opciones avanzadas del CLI:
* `--recuperar-nota-x10`: Recupera notas entre 10 y 100 dividiendo por 10 (asumiendo error en coma decimal).
* `--excluir-en-revision`: Excluye de la muestra limpia también los registros con advertencias no críticas.

### B. Artefactos Generados en `salida_limpieza/`:
* `ETAIC_Dataset_LIMPIO.xlsx`: Libro de trabajo profesional con 8 hojas formateadas:
  1. **`Datos_limpios`**: Muestra purificada ($N=1184$) lista para tablas dinámicas y análisis estadístico.
  2. **`Datos_excluidos`**: 16 registros descartados con el motivo exacto de exclusión.
  3. **`Clasificacion_registros`**: Diagnóstico de fiabilidad registro por registro.
  4. **`Resumen`**: Estadísticas globales del proceso de limpieza.
  5. **`Reglas_aplicadas`**: Catálogo formal de las 15 reglas de validación y coherencia.
  6. **`Registro_cambios`**: Bitácora atómica de los 116 cambios aplicados.
  7. **`Avisos_coherencia`**: Detección de patrones discordantes (transporte, IA, sueño).
  8. **`Valores_vacios`**: Porcentaje de ausencia por variable para identificar saltos condicionales legítimos.
* `ETAIC_Dataset_LIMPIO.csv` y `ETAIC_Dataset_EXCLUIDOS.csv`: Datos en formato tabular estándar.

### C. Ejecutar la Batería de Pruebas Unitarias
El motor cuenta con **25 pruebas automatizadas** que cubren valores físicamente imposibles, límites psicométricos, deduplicación vectorizada y prevención de falsas omisiones:

```bash
python3 -m unittest discover tests
```

---

## 5. Declaración sobre el Uso de Inteligencia Artificial

> De conformidad con la normativa docente de ETAIC (curso 2026-27):  
* **Declaración de uso:** Se ha utilizado Inteligencia Artificial como herramienta de apoyo puntual y asistencia técnica para la **generación, optimización vectorizada y refactorización del script de Python (`limpiar_dataset_etaic.py`)**, así como para la elaboración de la batería de pruebas unitarias (`tests/test_limpieza_etaic.py`).
* **Responsabilidad metodológica:** El diseño de la pregunta de investigación, los objetivos, la formulación de hipótesis, la toma de decisiones sobre qué criterios de exclusión aplicar al dataset y la redacción íntegra de todas las secciones del artículo científico corresponden exclusivamente a los autores humanos del grupo.

---

## 6. Autores y Licencia

* **Autores:** Rubén González Sánchez, Julen San Román Delarraya y Eusebio Gómez
* **Titulación:** Grado en Ingeniería Informática / ADET · Universidad Pontificia de Salamanca (UPSA)
* **Licencia:** Material académico desarrollado para la asignatura ETAIC (2026-27). Prohibida su difusión externa sin autorización expresa de los autores y la coordinación docente.
