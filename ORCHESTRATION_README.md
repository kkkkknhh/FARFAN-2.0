# Sistema de Orquestación FARFAN 2.0

## Visión General

Este sistema implementa un **flujo canónico, determinista e inmutable** para evaluar Planes de Desarrollo Municipal mediante la respuesta sistemática a **300 preguntas de evaluación causal**.

## Componentes Principales

### 1. `orchestrator.py` - Orquestador Principal
Coordina el flujo completo de procesamiento en 9 etapas canónicas:
1. Carga y extracción de documento (PDF)
2. Extracción de texto, tablas y secciones
3. Análisis semántico
4. Extracción de jerarquía causal
5. Inferencia de mecanismos
6. Auditoría financiera
7. Validación DNP
8. **Respuesta a 300 preguntas**
9. **Generación de reportes (3 niveles)**

### 2. `question_answering_engine.py` - Motor de Preguntas
Responde 300 preguntas (30 base × 10 áreas de política):
- Coordina TODOS los módulos del framework
- Genera respuestas estructuradas (texto + argumento doctoral + nota cuantitativa)
- Acumula evidencia de múltiples fuentes

### 3. `report_generator.py` - Generador de Reportes
Crea reportes a tres niveles:

#### Nivel MICRO
- 300 respuestas individuales
- Cada respuesta incluye: texto, argumento (2+ párrafos), nota (0-1), evidencia
- Formato: JSON

#### Nivel MESO
- Agrupación en 4 clústeres temáticos
- Análisis por 6 dimensiones del Marco Lógico
- Evaluación cualitativa por clúster
- Formato: JSON

#### Nivel MACRO
- Evaluación global de alineación con el decálogo
- Análisis retrospectivo (¿qué tan cerca está?)
- Análisis prospectivo (¿qué debe mejorar?)
- Recomendaciones prioritarias
- Formato: JSON + Markdown

### 4. `module_choreographer.py` - Coreógrafo de Módulos
- Registra ejecución de cada módulo
- Acumula contribuciones de múltiples módulos
- Genera trazabilidad completa
- Verifica que TODOS los módulos se usen

## Sistema de 300 Preguntas

### Estructura

**30 Preguntas Base** organizadas en **6 Dimensiones** (D1-D6):
- D1: Insumos (Diagnóstico y Líneas Base) - 5 preguntas
- D2: Actividades (Formalizadas) - 5 preguntas  
- D3: Productos (Verificables) - 5 preguntas
- D4: Resultados (Medibles) - 5 preguntas
- D5: Impactos (Largo Plazo) - 5 preguntas
- D6: Causalidad (Teoría de Cambio) - 5 preguntas

**10 Áreas de Política** (Decálogo):
- P1: Derechos de las mujeres e igualdad de género
- P2: Prevención de la violencia y protección frente al conflicto
- P3: Ambiente sano, cambio climático, prevención y atención a desastres
- P4: Derechos económicos, sociales y culturales
- P5: Derechos de las víctimas y construcción de paz
- P6: Derecho al buen futuro de la niñez, adolescencia, juventud
- P7: Tierras y territorios
- P8: Líderes y defensores de derechos humanos
- P9: Crisis de derechos de personas privadas de la libertad
- P10: Migración transfronteriza

**Total: 30 × 10 = 300 preguntas**

### ID de Preguntas

Formato: `P{punto}-D{dimensión}-Q{número}`

Ejemplos:
- `P1-D1-Q1`: Seguridad, Insumos, Pregunta 1
- `P5-D4-Q16`: Víctimas, Resultados, Pregunta 16
- `P10-D6-Q30`: Migración, Causalidad, Pregunta 30

## Uso

### Instalación de Dependencias

```bash
pip install pymupdf networkx pandas spacy pyyaml fuzzywuzzy python-Levenshtein pydot scipy numpy
python -m spacy download es_core_news_lg
```

### Desde Línea de Comandos

```bash
python orchestrator.py plan_desarrollo.pdf \
    --policy-code PDM2024-ANT-MED \
    --output-dir ./resultados \
    --pdet \
    --log-level INFO
```

### Desde Python

```python
from orchestrator import FARFANOrchestrator
from pathlib import Path

# Crear orquestador
orchestrator = FARFANOrchestrator(
    output_dir=Path("./resultados"),
    log_level="INFO"
)

# Procesar plan
context = orchestrator.process_plan(
    pdf_path=Path("plan_desarrollo.pdf"),
    policy_code="PDM2024-ANT-MED",
    es_municipio_pdet=True
)

# Acceder a resultados
print(f"Score global: {context.macro_report['evaluacion_global']['score_global']:.2f}")
print(f"Preguntas: {len(context.question_responses)}")
```

### Procesar Múltiples Planes (170 municipios)

```python
from pathlib import Path
from orchestrator import FARFANOrchestrator

orchestrator = FARFANOrchestrator(output_dir=Path("./resultados_170"))

planes_dir = Path("./planes_desarrollo")
for pdf_file in planes_dir.glob("*.pdf"):
    policy_code = pdf_file.stem  # Usa nombre de archivo como código
    
    try:
        context = orchestrator.process_plan(
            pdf_path=pdf_file,
            policy_code=policy_code,
            es_municipio_pdet=True  # Ajustar según sea PDET
        )
        print(f"✓ {policy_code}: Score {context.macro_report['evaluacion_global']['score_global']:.2f}")
    except Exception as e:
        print(f"✗ {policy_code}: Error - {e}")
```

## Archivos Generados

Para cada plan procesado con código `PDM2024-ANT-MED`, se generan:

1. `micro_report_PDM2024-ANT-MED.json` - 300 respuestas detalladas
2. `meso_report_PDM2024-ANT-MED.json` - 4 clústeres × 6 dimensiones
3. `macro_report_PDM2024-ANT-MED.json` - Evaluación global (JSON)
4. `macro_report_PDM2024-ANT-MED.md` - Evaluación global (Markdown)

## Módulos Utilizados

El orquestador integra **TODOS** los módulos del framework:

| Módulo | Clases Principales | Funciones Utilizadas |
|--------|-------------------|---------------------|
| `dereck_beach` | CDAFFramework, PDFProcessor, CausalExtractor | load_document, extract_text, extract_causal_hierarchy |
| `dnp_integration` | ValidadorDNP | validar_proyecto_integral, generar_reporte_cumplimiento |
| `competencias_municipales` | CatalogoCompetenciasMunicipales | validar_competencia_municipal, get_pdet_prioritarias |
| `mga_indicadores` | CatalogoIndicadoresMGA | buscar_por_sector, generar_reporte_alineacion |
| `pdet_lineamientos` | LineamientosPDET | recomendar_lineamientos, validar_requisitos |
| `initial_processor_causal_policy` | PolicyDocumentAnalyzer | analyze_document (cuando esté disponible) |

## Sistema de Scoring

### Escalas

```
1.00 - 0.85: Excelente
0.84 - 0.70: Bueno
0.69 - 0.55: Aceptable
0.54 - 0.40: Insuficiente
0.39 - 0.00: No Cumplimiento
```

### Agregación

1. **Por Pregunta**: Basado en evidencia y criterios
2. **Por Dimensión**: Promedio de 5 preguntas
3. **Por Punto**: Promedio ponderado de 6 dimensiones
4. **Por Clúster**: Promedio de puntos incluidos
5. **Global**: Promedio de 300 preguntas

## Clústeres Meso

**C1: Derechos de las Mujeres, Prevención de Violencia y Protección de Líderes**
- P1: Derechos de las mujeres e igualdad de género
- P2: Prevención de la violencia y protección frente al conflicto
- P8: Líderes y defensores de derechos humanos

**C2: Derechos Económicos, Sociales, Culturales y Poblaciones Vulnerables**
- P4: Derechos económicos, sociales y culturales
- P5: Derechos de las víctimas y construcción de paz
- P6: Derecho al buen futuro de la niñez, adolescencia, juventud

**C3: Ambiente, Cambio Climático, Tierras y Territorios**
- P3: Ambiente sano, cambio climático, prevención y atención a desastres
- P7: Tierras y territorios

**C4: Personas Privadas de Libertad y Migración**
- P9: Crisis de derechos de personas privadas de la libertad
- P10: Migración transfronteriza

## Garantías del Sistema

### ✓ Determinismo
- Mismo input → mismo output
- No aleatoriedad en scoring
- Orden de ejecución fijo

### ✓ Inmutabilidad
- Datos originales nunca se modifican
- Cada etapa crea nuevos objetos
- Trazabilidad completa

### ✓ Completitud
- TODAS las clases y funciones se utilizan
- Ninguna pregunta queda sin responder
- Todos los nodos se validan con DNP

### ✓ Trazabilidad
- Cada respuesta indica módulos utilizados
- Evidencia vinculada al documento
- Metadata completa en reportes

## Ejemplos de Salida

### Micro Report (extracto)
```json
{
  "P1-D1-Q1": {
    "respuesta": "Sí, el plan cumple excelentemente...",
    "argumento": "La evaluación de Derechos de las mujeres e igualdad de género...\n\nLa evidencia encontrada...",
    "nota_cuantitativa": 0.85,
    "evidencia": ["Nodo X: ...", "Score DNP: 82/100"],
    "modulos_utilizados": ["dereck_beach", "dnp_integration"]
  }
}
```

### Meso Report (extracto)
```json
{
  "C1": {
    "nombre": "Seguridad, Paz y Protección",
    "dimensiones": {
      "D1": {"score": 0.78, "nivel_cumplimiento": "Bueno"},
      "D2": {"score": 0.65, "nivel_cumplimiento": "Aceptable"}
    },
    "evaluacion_general": "El clúster presenta un score promedio..."
  }
}
```

### Macro Report (Markdown)
```markdown
# Reporte Macro - Evaluación de Plan de Desarrollo
## PDM2024-ANT-MED

**Score Global:** 0.75 (75%)
**Nivel de Alineación:** Alineado
**Score DNP:** 78/100

## Análisis Retrospectivo
El plan se encuentra aproximadamente a 25% de distancia...

## Recomendaciones Prioritarias
1. Fortalecer Dimensión D2 (Actividades): score de 0.58
2. Incorporar líneas base cuantitativas...
```

## Documentación Adicional

- Ver `ORCHESTRATOR_DOCUMENTATION.md` para detalles técnicos completos
- Ver código fuente para comentarios detallados
- Ver ejemplos en `ejemplo_dnp_completo.py`

## Soporte

Para preguntas o problemas:
1. Revisar logs generados durante el procesamiento
2. Verificar que todos los módulos estén instalados correctamente
3. Consultar documentación de módulos individuales

## Autores

FARFAN 2.0 Development Team - 2024
