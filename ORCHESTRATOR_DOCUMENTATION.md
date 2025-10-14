# FARFAN 2.0 - Documentación del Orquestador

## Visión General

El sistema de orquestación de FARFAN 2.0 implementa un flujo canónico, determinista e inmutable para evaluar Planes de Desarrollo Municipal mediante la respuesta sistemática a 300 preguntas de evaluación causal.

## Arquitectura del Sistema

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR.PY                          │
│         Flujo Canónico de 9 Etapas                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐   ┌────▼────┐   ┌───▼────┐
    │  CDAF   │   │   DNP   │   │  QA    │
    │Framework│   │Validator│   │ Engine │
    └─────────┘   └─────────┘   └────────┘
         │             │             │
         └─────────────┼─────────────┘
                       │
              ┌────────▼────────┐
              │ Report Generator│
              │   (3 niveles)   │
              └─────────────────┘
```

### Flujo de Procesamiento (9 Etapas Canónicas)

#### STAGE 1-2: Document Loading and Extraction
- **Módulo**: `dereck_beach` (PDFProcessor)
- **Input**: PDF del Plan de Desarrollo
- **Output**: 
  - `raw_text`: Texto completo extraído
  - `tables`: Tablas identificadas
  - `sections`: Secciones del documento
- **Funciones utilizadas**:
  - `PDFProcessor.load_document()`
  - `PDFProcessor.extract_text()`
  - `PDFProcessor.extract_tables()`
  - `PDFProcessor.extract_sections()`

#### STAGE 3: Semantic Analysis
- **Módulo**: `initial_processor_causal_policy` (PolicyDocumentAnalyzer)
- **Input**: raw_text, sections
- **Output**: 
  - `semantic_chunks`: Fragmentos semánticos
  - `dimension_scores`: Scores por dimensión
- **Funciones utilizadas**:
  - `PolicyDocumentAnalyzer.analyze_document()`
  - `SemanticProcessor.embed_texts()`
  - `BayesianEvidenceIntegrator.integrate()`

#### STAGE 4: Causal Extraction
- **Módulo**: `dereck_beach` (CausalExtractor)
- **Input**: raw_text
- **Output**:
  - `causal_graph`: NetworkX DiGraph con jerarquía causal
  - `nodes`: Dict de MetaNodes (productos, resultados, impactos)
  - `causal_chains`: Lista de enlaces causales
- **Funciones utilizadas**:
  - `CausalExtractor.extract_causal_hierarchy()`
  - `CausalExtractor.classify_goal()`
  - `CausalExtractor.extract_causal_link()`

#### STAGE 5: Mechanism Inference
- **Módulo**: `dereck_beach` (MechanismPartExtractor, BayesianMechanismInference)
- **Input**: nodes, causal_graph
- **Output**:
  - `mechanism_parts`: Lista de tuplas Entidad-Actividad
  - `bayesian_inferences`: Inferencias bayesianas de mecanismos
- **Funciones utilizadas**:
  - `MechanismPartExtractor.extract_entity_activity()`
  - `BayesianMechanismInference.infer_mechanism()`
  - `BayesianMechanismInference.run_mcmc()`

#### STAGE 6: Financial Audit
- **Módulo**: `dereck_beach` (FinancialAuditor)
- **Input**: tables, nodes
- **Output**:
  - `financial_allocations`: Asignaciones presupuestales por nodo
  - `budget_traceability`: Trazabilidad BPIN/PPI
- **Funciones utilizadas**:
  - `FinancialAuditor.trace_financial_allocation()`
  - `FinancialAuditor._process_financial_table()`

#### STAGE 7: DNP Validation
- **Módulos**: `dnp_integration`, `competencias_municipales`, `mga_indicadores`, `pdet_lineamientos`
- **Input**: nodes, financial_allocations
- **Output**:
  - `dnp_validation_results`: Resultados de validación por nodo
  - `compliance_score`: Score de cumplimiento DNP (0-100)
- **Funciones utilizadas**:
  - `ValidadorDNP.validar_proyecto_integral()`
  - `CatalogoCompetenciasMunicipales.validar_competencia_municipal()`
  - `CatalogoIndicadoresMGA.buscar_por_sector()`
  - `LineamientosPDET.recomendar_lineamientos()`

#### STAGE 8: Question Answering (300 Preguntas)
- **Módulo**: `question_answering_engine` (QuestionAnsweringEngine)
- **Input**: PipelineContext completo
- **Output**:
  - `question_responses`: Dict con 300 RespuestaPregunta
    - Cada respuesta incluye: texto, argumento doctoral, nota cuantitativa, evidencia
- **Funciones utilizadas**:
  - `QuestionAnsweringEngine.answer_all_questions()`
  - `QuestionAnsweringEngine._answer_single_question()`
  - Coordina TODOS los módulos para generar respuestas fundamentadas

#### STAGE 9: Report Generation (3 Niveles)
- **Módulo**: `report_generator` (ReportGenerator)
- **Input**: question_responses, compliance_score
- **Output**:
  - `micro_report`: 300 respuestas individuales (JSON)
  - `meso_report`: 4 clústeres × 6 dimensiones (JSON)
  - `macro_report`: Alineación global (JSON + Markdown)
- **Funciones utilizadas**:
  - `ReportGenerator.generate_micro_report()`
  - `ReportGenerator.generate_meso_report()`
  - `ReportGenerator.generate_macro_report()`

## Sistema de 300 Preguntas

### Estructura

- **30 Preguntas Base**: Organizadas en 6 dimensiones analíticas (D1-D6)
  - D1: Insumos (5 preguntas: Q1-Q5)
  - D2: Actividades (5 preguntas: Q6-Q10)
  - D3: Productos (5 preguntas: Q11-Q15)
  - D4: Resultados (5 preguntas: Q16-Q20)
  - D5: Impactos (5 preguntas: Q21-Q25)
  - D6: Causalidad (5 preguntas: Q26-Q30)

- **10 Áreas de Política** (Decálogo):
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

- **Total**: 30 preguntas × 10 áreas = **300 preguntas**

### ID de Preguntas

Formato: `P{punto}-D{dimensión}-Q{número}`

Ejemplos:
- `P1-D1-Q1`: Seguridad, Dimensión Insumos, Pregunta 1
- `P5-D4-Q16`: Víctimas, Dimensión Resultados, Pregunta 16
- `P10-D6-Q30`: Migración, Dimensión Causalidad, Pregunta 30

### Estructura de Respuesta

Cada pregunta genera una `RespuestaPregunta` con:

```python
{
    "pregunta_id": "P1-D1-Q1",
    "respuesta_texto": "Respuesta directa a la pregunta",
    "argumento": "Dos o más párrafos de argumentación de nivel doctoral...",
    "nota_cuantitativa": 0.85,  # Score 0.0-1.0
    "evidencia": [
        "Extracto 1 del documento",
        "Extracto 2 del documento",
        "Score DNP: 78/100"
    ],
    "modulos_utilizados": [
        "dereck_beach",
        "dnp_integration",
        "mga_indicadores"
    ],
    "nivel_confianza": 0.9
}
```

## Reportes Generados

### Nivel MICRO (micro_report_{policy_code}.json)

Contiene las 300 respuestas individuales con toda la evidencia y argumentación.

**Uso**: 
- Revisión detallada pregunta por pregunta
- Identificación de brechas específicas
- Soporte técnico para reformulación

### Nivel MESO (meso_report_{policy_code}.json)

Agrupa en 4 clústeres temáticos:
- **C1**: Derechos de las Mujeres, Prevención de Violencia y Protección de Líderes (P1, P2, P8)
- **C2**: Derechos Económicos, Sociales, Culturales y Poblaciones Vulnerables (P4, P5, P6)
- **C3**: Ambiente, Cambio Climático, Tierras y Territorios (P3, P7)
- **C4**: Personas Privadas de Libertad y Migración (P9, P10)

Para cada clúster analiza las 6 dimensiones (D1-D6):
```json
{
    "C1": {
        "nombre": "Derechos de las Mujeres, Prevención de Violencia y Protección de Líderes",
        "dimensiones": {
            "D1": {
                "score": 0.78,
                "nivel_cumplimiento": "Bueno",
                "observaciones": "..."
            },
            ...
        },
        "evaluacion_general": "Análisis integral del clúster..."
    }
}
```

**Uso**:
- Identificación de patrones transversales
- Priorización de áreas de fortalecimiento
- Análisis de coherencia entre áreas relacionadas

### Nivel MACRO (macro_report_{policy_code}.md/json)

Evaluación global del plan con:
- **Análisis Retrospectivo**: ¿Qué tan lejos/cerca está del óptimo?
- **Análisis Prospectivo**: ¿Qué debe mejorar prioritariamente?
- **Score Global**: Promedio ponderado de las 300 preguntas
- **Recomendaciones Prioritarias**: Top 5-10 acciones correctivas
- **Fortalezas y Debilidades Críticas**

**Uso**:
- Toma de decisiones estratégicas
- Comunicación con autoridades territoriales
- Base para planes de mejoramiento

## Sistema de Scoring

### Escalas de Evaluación

```
1.0 - 0.85: Excelente
0.84 - 0.70: Bueno
0.69 - 0.55: Aceptable
0.54 - 0.40: Insuficiente
0.39 - 0.00: No Cumplimiento
```

### Agregación de Scores

1. **Por Pregunta**: Basado en evidencia y criterios específicos
2. **Por Dimensión**: Promedio ponderado de 5 preguntas
3. **Por Punto del Decálogo**: Promedio ponderado de 6 dimensiones (con pesos variables)
4. **Por Clúster**: Promedio de puntos incluidos
5. **Global**: Promedio de todas las 300 preguntas

### Pesos por Dimensión según Punto

Ejemplo para P1 (Seguridad):
```json
{
    "D1_weight": 0.20,
    "D2_weight": 0.20,
    "D3_weight": 0.15,
    "D4_weight": 0.20,
    "D5_weight": 0.15,
    "D6_weight": 0.10,
    "critical_dimensions": ["D1", "D2", "D4"],
    "minimum_per_dimension": 0.50
}
```

## Uso del Sistema

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
print(f"Score global: {context.macro_report['evaluacion_global']['score_global']}")
print(f"Preguntas respondidas: {len(context.question_responses)}")
```

## Módulos y Responsabilidades

### Mapeo Pregunta → Módulos

| Dimensión | Pregunta | Módulos Responsables |
|-----------|----------|---------------------|
| D1-Q1     | Líneas base | `initial_processor_causal_policy`, `dereck_beach` |
| D1-Q2     | Magnitud problema | `dereck_beach`, `initial_processor_causal_policy` |
| D1-Q3     | Recursos trazables | `dereck_beach`, `dnp_integration` |
| D1-Q4     | Capacidades | `dereck_beach`, `competencias_municipales` |
| D1-Q5     | Coherencia | `dnp_integration`, `dereck_beach` |
| D2-Q6     | Estructura tabular | `dereck_beach` |
| D3-Q11    | Indicadores | `mga_indicadores`, `dereck_beach` |
| D4-Q16    | Resultados | `dereck_beach`, `mga_indicadores` |
| D6-Q26    | Teoría cambio | `dereck_beach`, `initial_processor_causal_policy` |

## Garantías del Sistema

### Determinismo
- Mismo PDF + mismo policy_code → mismo resultado
- No hay aleatoriedad en scoring (solo en inferencia bayesiana con seed fijo)
- Orden de evaluación de preguntas es fijo

### Inmutabilidad
- Los datos originales (PDF, contexto) nunca se modifican
- Cada etapa crea nuevos objetos, no modifica existentes
- Trazabilidad completa de transformaciones

### Completitud
- **TODAS** las clases y funciones de cada módulo son utilizadas
- Ninguna pregunta queda sin responder
- Todos los nodos extraídos son validados con DNP

### Trazabilidad
- Cada respuesta indica qué módulos la generaron
- Evidencia vinculada a fragmentos del documento original
- Metadata completa en todos los reportes

## Extensión del Sistema

### Agregar Nuevas Preguntas

1. Editar `question_answering_engine.py`
2. Agregar en `_load_question_templates()`:

```python
preguntas["D7-Q31"] = PreguntaBase(
    id_base="D7-Q31",
    dimension=DimensionCausal.D7_NUEVA,
    numero=31,
    texto_template="¿Nueva pregunta para {}?",
    criterios_evaluacion={...},
    scoring={...},
    modulos_responsables=["nuevo_modulo", "dereck_beach"]
)
```

### Agregar Nuevos Módulos

1. Importar en `orchestrator.py` en `_init_modules()`
2. Agregar uso en la etapa correspondiente del pipeline
3. Registrar en `modulos_responsables` de preguntas relevantes
4. Usar en `question_answering_engine._answer_single_question()`

### Personalizar Clústeres

Editar `report_generator.py`:
```python
self.punto_to_cluster = {
    "P1": ClusterMeso.C1_NUEVO,
    ...
}
```

## Mantenimiento y Validación

### Tests Recomendados

1. **Test de Completitud**: Verificar que todas las 300 preguntas se respondan
2. **Test de Determinismo**: Mismo input debe dar mismo output
3. **Test de Módulos**: Cada módulo debe ser invocado al menos una vez
4. **Test de Scoring**: Scores deben estar en rango [0, 1]
5. **Test de Reportes**: Verificar que los 3 niveles se generen correctamente

### Logs y Debugging

El sistema genera logs detallados en cada etapa:
```
2024-10-14 10:00:00 - orchestrator - INFO - [STAGE 1-2] Extrayendo documento...
2024-10-14 10:00:05 - orchestrator - INFO -   ✓ Texto extraído: 125000 caracteres
2024-10-14 10:00:05 - orchestrator - INFO -   ✓ Tablas extraídas: 15
...
```

## Preguntas Frecuentes

**P: ¿Por qué 300 preguntas y no 170 (30×5.67)?**
R: Son 30 preguntas base × 10 áreas de política = 300 preguntas totales.

**P: ¿Se puede procesar más de un plan a la vez?**
R: Sí, simplemente crear múltiples instancias del orquestador o usar diferentes output_dir.

**P: ¿Qué pasa si un módulo falla?**
R: El orquestador continúa con advertencias, pero el scoring puede verse afectado.

**P: ¿Cómo se integra con los 170 planes?**
R: Usar un script que itere sobre los 170 PDFs:
```python
for pdf in planes_directory.glob("*.pdf"):
    context = orchestrator.process_plan(pdf, pdf.stem)
```

**P: ¿Los reportes se pueden regenerar?**
R: Sí, dado el PipelineContext guardado, se pueden regenerar reportes sin reprocesar.

---

**Versión**: 1.0.0  
**Fecha**: 2024-10-14  
**Autores**: FARFAN 2.0 Development Team
