# Pipeline de Limpieza, Validación y Auditoría de Datos · ETAIC 2026-27

> **Trabajo Final:** Artículo científico breve a partir de datos comunes, revisión por pares y Jornadas ETAIC  
> **Asignatura:** Elaboración de Trabajos Académicos con Indicadores de Calidad (ETAIC)  
> **Institución:** Facultad de Informática · Universidad Pontificia de Salamanca (UPSA)  
> **Formato de Publicación:** Plantilla IEEE (Ingeniería Informática) / APA 7 (ADET)  
> **Dataset Base:** *«Encuesta sobre vida universitaria 2026»* (Datos simulados, S. Alonso Rollán)

---

## 1. Descripción del Proyecto

Este repositorio contiene la infraestructura computacional de reproducibilidad, limpieza automatizada, validación estricta y auditoría del dataset utilizado para la redacción del artículo científico breve en la asignatura **ETAIC (curso 2026-27)**.

El objetivo principal es dar cobertura técnica y justificación cuantitativa al **Criterio 4 de la rúbrica de evaluación («Datos y método»)**, garantizando que cada registro admitido o descartado cuente con trazabilidad matemática reproducible y cumplimiento con los estándares de rigor metodológico.

---

## 2. Declaración de Datos y Reproducibilidad Académica

### Origen de los Datos
Los datos proceden del archivo docente oficial `ETAIC_Dataset_VidaUniversitaria2026.xlsx` compuesto por **1.200 respuestas** y **92 variables** (12 comunes sociodemográficas y 10 bloques temáticos de 8 variables).

> **Aviso Obligatorio de Integridad (Normativa ETAIC):**  
> *«Los datos utilizados en este trabajo son datos SIMULADOS por ordenador con fines docentes para la asignatura ETAIC (UPSA, 2026-27). No proceden de personas reales ni describen a estudiantes concretos.»*

### Cómo Citar el Conjunto de Datos
* **Estilo IEEE:**  
  `[1] S. Alonso Rollán, «Encuesta sobre vida universitaria 2026 (datos simulados)», conjunto de datos, Universidad Pontificia de Salamanca, Salamanca, España, 2026.`
* **Estilo APA 7:**  
  `Alonso Rollán, S. (2026). Encuesta sobre vida universitaria 2026 (datos simulados) [Conjunto de datos]. Universidad Pontificia de Salamanca. Material docente de ETAIC.`

---

## 3. Síntesis de Calidad y Muestra Final (Criterio 4)

El pipeline ejecuta una auditoría atómica determinista sobre las 1.200 respuestas originales:

| Métrica de Muestra | Casos | Impacto Metodológico |
| :--- | :---: | :--- |
| **Muestra de partida ($N_{\text{total}}$)** | **1.200** | Respuestas brutas en `Datos`. |
| **Duplicados exactos eliminados** | **6** | Descartados por redundancia idéntica de fila. |
| **Registros no fiables excluidos** | **10** | Anomalías críticas (incoherencia lógica o límites físicos imposibles). |
| **Registros atípicos excluidos (variables de estudio)** | **5** | Expectativas salariales extremas fuera de $[12.000\ €, 90.000\ €]$ (>200k€). |
| **Muestra analítica final ($N_{\text{limpio}}$)** | **1.179** | **98,25% de fiabilidad global** admitida para el análisis (643 INF, 536 ADET). |
| *— Registros válidos (completos)* | *865* | Datos 100% conformes y sin valores nulos obligatorios. |
| *— Registros incompletos (admitidos)* | *309* | Válidos para análisis por bloques (omisiones no críticas). |
| *— Registros en revisión conservados* | *5* | Atípicos en variables ajenas al estudio (4 de sueño y 1 de ingresos). |
| **Categorías textuales normalizadas** | **91** | Corrección de erratas sintácticas (`inf` $\rightarrow$ `INF`, etc.). |
| **Eventos de auditoría registrados** | **116** | Trazabilidad celda por celda exportada en bitácora. |

---

## 4. Estructura del Repositorio

```text
.
├── excel/
│   └── ETAIC_Dataset_VidaUniversitaria2026.xlsx   # Dataset original docente crudo
├── tests/
│   ├── test_limpieza_etaic.py                     # Batería de 26 pruebas unitarias de limpieza
│   └── test_analisis_variables.py                 # Pruebas del módulo de análisis de variables
├── limpiar_dataset_etaic.py                       # Motor de limpieza y validación vectorizado
├── analisis_variables_seleccionadas.py            # Aislamiento de variables, Tabla I y contrastes
├── requirements.txt                               # Dependencias de Python necesarias
├── .gitignore                                     # Exclusiones de caché y temporales
└── README.md                                      # Documentación y guía de reproducibilidad
```

---

## 5. Instalación y Requisitos

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

## 6. Guía de Ejecución

### A. Ejecutar el Pipeline de Limpieza
Para procesar el archivo Excel original y generar los reportes de auditoría:

```bash
python3 limpiar_dataset_etaic.py --entrada excel/ETAIC_Dataset_VidaUniversitaria2026.xlsx --salida-dir salida_limpieza
```

#### Opciones avanzadas del CLI:
* `--recuperar-nota-x10`: Recupera notas entre 10 y 100 dividiendo por 10 (asumiendo error en coma decimal).
* `--conservar-en-revision`: Permite admitir los registros atípicos (en revisión) en lugar de excluirlos por criterio estricto.

### B. Artefactos Generados en `salida_limpieza/`:
* `ETAIC_Dataset_LIMPIO.xlsx`: Libro de trabajo profesional con 8 hojas formateadas:
  1. **`Datos_limpios`**: Muestra purificada ($N=1179$) lista para tablas dinámicas y análisis estadístico.
  2. **`Datos_excluidos`**: 21 registros descartados (6 duplicados, 10 anomalías críticas y 5 valores atípicos salariales).
  3. **`Clasificacion_registros`**: Diagnóstico de fiabilidad registro por registro.
  4. **`Resumen`**: Estadísticas globales del proceso de limpieza.
  5. **`Reglas_aplicadas`**: Catálogo formal de las 15 reglas de validación y coherencia.
  6. **`Registro_cambios`**: Bitácora atómica de los 116 cambios aplicados.
  7. **`Avisos_coherencia`**: Detección de patrones discordantes (transporte, IA, sueño).
  8. **`Valores_vacios`**: Porcentaje de ausencia por variable para identificar saltos condicionales legítimos.
* `ETAIC_Dataset_LIMPIO.csv` y `ETAIC_Dataset_EXCLUIDOS.csv`: Datos en formato tabular estándar.

### C. Ejecutar la Batería de Pruebas Unitarias
El motor cuenta con **26 pruebas automatizadas** que cubren valores físicamente imposibles, límites psicométricos, deduplicación vectorizada, prevención de falsas omisiones y exclusión tajante de atípicos:

```bash
python3 -m unittest discover tests
```

---

## 7. Cronograma y Talleres ETAIC (Hitos del Trabajo)

| Fecha | Hito / Taller | Entregable / Producto |
| :--- | :--- | :--- |
| **M 27 oct** | Resumen de la práctica y Mini hito | Elección de tema, enfoque y variables. |
| **M 3 nov** | **Taller 1:** Introducción y Estado del Arte | Sello 1 · Objetivos y $\ge 3$ fuentes con DOI en gestor. |
| **L 9 nov** | **Taller 2:** Datos, Método y Resultados | Sello 2 · Ejecución de limpieza, $N$ final, figuras y tablas. |
| **M 10 nov** | **Taller 3:** Discusión, Conclusiones y Resumen | Sello 3 · Borrador anónimo completo (hasta 23:59). |
| **L 16 nov** | **Taller 4:** Revisión por pares | Sello 4 · Informe de revisión emitido sobre otro grupo. |
| **V 20 nov** | Versión Final | Carta de respuesta a revisores + Artículo definitivo en Moodle. |
| **23 nov - 1 dic** | **Jornadas ETAIC** | Exposición oral (12 min exposición + 4 min preguntas). |

---

## 8. Declaración sobre el Uso de Inteligencia Artificial

> De conformidad con la normativa docente de ETAIC (curso 2026-27):  
> *«La IA es un apoyo puntual o un mecanismo de consulta. Sin embargo, la IA no realiza el trabajo; si se detecta un trabajo realizado con IA se da por suspensa toda la evaluación.»*

* **Declaración explícita de uso:** Se ha utilizado Inteligencia Artificial como herramienta de apoyo puntual y asistencia técnica para la **generación, optimización vectorizada y refactorización del script de Python (`limpiar_dataset_etaic.py`)**, así como para la elaboración de la batería de pruebas unitarias (`tests/test_limpieza_etaic.py`).
* **Responsabilidad metodológica:** El diseño de la pregunta de investigación, los objetivos, la formulación de hipótesis, la toma de decisiones sobre qué criterios de exclusión aplicar al dataset y la redacción íntegra de todas las secciones del artículo científico corresponden exclusivamente a los autores humanos del grupo.
* **Trazabilidad docente:** El uso de la IA queda registrado en el **Diario de a bordo** del grupo y en el apartado de *«Declaración de uso de IA»* del artículo final conforme a las directrices de la plantilla.

---

## 9. Autores y Licencia

* **Autores:** [Nombres y Apellidos del Grupo]
* **Titulación:** Grado en Ingeniería Informática / ADET · Universidad Pontificia de Salamanca (UPSA)
* **Licencia:** Material académico desarrollado para la asignatura ETAIC (2026-27). Prohibida su difusión externa sin autorización expresa de los autores y la coordinación docente.

