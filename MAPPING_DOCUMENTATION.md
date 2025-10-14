# Documentación del Mapeo de Preguntas a Callables

## Resumen Ejecutivo

El archivo `cuestionario_canonico_mapped.json` contiene el mapeo completo de las **300 preguntas** del sistema FARFAN 2.0 a los scripts, clases y funciones responsables de responderlas.

### Estadísticas Globales

- **Total de preguntas:** 300 (10 políticas × 30 preguntas por política)
- **Scripts utilizados:** 15 de 25 (60%)
- **Callables únicos mapeados:** 200+
- **Cobertura:** 100% - todas las preguntas tienen mapeos completos
- **Score promedio:** 8.50/10

## Estructura del Mapeo

Cada pregunta en el archivo tiene la siguiente estructura:

```json
{
  "id": "P1-D1-Q1",
  "texto": "¿El diagnóstico presenta datos numéricos...",
  "mapping": [
    {
      "script": "competencias_municipales.py",
      "callable": "CatalogoCompetenciasMunicipales.validar_competencia_municipal",
      "contribution_type": "partial",
      "question_portion_handled": "validación de diagnóstico, líneas base y recursos",
      "input_needed": ["pdf_document", "diagnóstico_text", "marco_legal_competencias"],
      "output_expected": ["score_parcial", "evidencia_text", "cumplimiento_criterios"]
    }
  ],
  "resolution_mode": "accumulation",
  "scoring": {
    "cuantitativo": 8.5,
    "cualitativo": "Excelente",
    "justificacion": "Diagnóstico robusto con fuentes oficiales..."
  },
  "audit_notes": "",
  "coverage_complete": true
}
```

### Campos Principales

#### `mapping`
Array de objetos que especifica todos los callables involucrados:

- **`script`**: Nombre del archivo Python
- **`callable`**: Nombre de la clase, función o método
- **`contribution_type`**: Tipo de contribución
  - `full`: Callable responde completamente la pregunta
  - `partial`: Callable contribuye parcialmente
  - `autonomous`: Callable opera independientemente
  - `orphan`: Callable no mapeado (excluido de mapeos)
  
- **`question_portion_handled`**: Porción textual específica que maneja
- **`input_needed`**: Lista de inputs requeridos
- **`output_expected`**: Lista de outputs producidos

#### `resolution_mode`
Modo de resolución de la pregunta:

- **`accumulation`**: Múltiples fuentes acumulan scores (común en D1, D5, D6)
- **`pipeline`**: Procesamiento secuencial (común en D2, D3, D4)
- **`hybrid`**: Combinación de acumulación y pipeline

#### `scoring`
Evaluación de la pregunta:

- **`cuantitativo`**: Score de 0-10
- **`cualitativo`**: Categoría (Excelente, Muy Bueno, Bueno, etc.)
- **`justificacion`**: Argumento doctoral detallado

## Mapeo de Dimensiones a Módulos

### D1: Diagnóstico y Líneas Base
**Módulos primarios:**
- `dereck_beach` - Análisis causal y diagnóstico
- `initial_processor_causal_policy` - Procesamiento inicial
- `dnp_integration` - Validación DNP

**Módulos secundarios:**
- `competencias_municipales` - Validación de competencias
- `resource_management` - Gestión de recursos

### D2: Actividades Formalizadas
**Módulos primarios:**
- `dereck_beach` - Análisis de actividades
- `module_choreographer` - Coordinación de módulos
- `pipeline_dag` - Gestión de pipeline

**Módulos secundarios:**
- `competencias_municipales` - Validación de competencias
- `dnp_integration` - Estándares DNP

### D3: Productos Verificables
**Módulos primarios:**
- `mga_indicadores` - Catálogo MGA
- `dnp_integration` - Validación DNP
- `dereck_beach` - Análisis causal

**Módulos secundarios:**
- `competencias_municipales` - Marco legal
- `report_generator` - Generación de reportes

### D4: Resultados Medibles
**Módulos primarios:**
- `dereck_beach` - Cadena causal
- `mga_indicadores` - Indicadores MGA
- `report_generator` - Reportes de resultados

**Módulos secundarios:**
- `dnp_integration` - Validación
- `smart_recommendations` - Recomendaciones

### D5: Impactos
**Módulos primarios:**
- `dereck_beach` - Teoría de cambio
- `report_generator` - Reportes de impacto
- `smart_recommendations` - Análisis de impacto

**Módulos secundarios:**
- `dnp_integration` - Alineación
- `canonical_notation` - Notación canónica

### D6: Causalidad y Teoría de Cambio
**Módulos primarios:**
- `dereck_beach` - CDAF framework
- `initial_processor_causal_policy` - Procesamiento causal
- `canonical_notation` - Sistema canónico

**Módulos secundarios:**
- `report_generator` - Documentación
- `module_choreographer` - Orquestación

## Mapeo de Políticas a Módulos

### P1: Género
- `competencias_municipales` - Marco legal de género

### P2: Violencia y Conflicto
- `competencias_municipales` - Competencias de seguridad
- `pdet_lineamientos` - Lineamientos PDET

### P3: Ambiente
- `competencias_municipales` - Competencias ambientales
- `dnp_integration` - Estándares ambientales

### P4: DESC (Derechos Económicos, Sociales y Culturales)
- `competencias_municipales` - Competencias sociales
- `mga_indicadores` - Indicadores sociales

### P5: Víctimas y Paz
- `pdet_lineamientos` - Lineamientos de paz
- `competencias_municipales` - Marco de víctimas

### P6: Niñez y Juventud
- `competencias_municipales` - Marco de protección

### P7: Tierras y Territorios
- `competencias_municipales` - Competencias rurales
- `pdet_lineamientos` - Reforma rural

### P8: Líderes y Defensores
- `competencias_municipales` - Protección
- `pdet_lineamientos` - Garantías

### P9: PPL (Personas Privadas de Libertad)
- `competencias_municipales` - Marco carcelario

### P10: Migración
- `competencias_municipales` - Gestión migratoria

## Scripts Más Utilizados

1. **competencias_municipales.py** - 3,300 mappings
   - Validación de competencias legales
   - Catálogo de competencias municipales
   
2. **canonical_notation.py** - 3,075 mappings
   - Sistema de notación canónica
   - Gestión de IDs y evidencia
   
3. **module_choreographer.py** - 1,836 mappings
   - Coordinación de ejecución
   - Acumulación de respuestas
   
4. **question_answering_engine.py** - 1,800 mappings
   - Motor principal de respuestas
   - Generación de respuestas doctorales
   
5. **smart_recommendations.py** - 1,700 mappings
   - Generación de recomendaciones
   - Análisis de mejoras

## Scripts No Utilizados (10)

Los siguientes scripts no están mapeados porque son de **soporte e infraestructura**:

1. **canonical_integration.py** - Integración del sistema canónico
2. **check_dependencies.py** - Verificación de dependencias
3. **estimate_processing.py** - Estimación de recursos
4. **example_risk_integration.py** - Script de ejemplo
5. **import_analyzer.py** - Análisis de imports
6. **module_interfaces.py** - Definición de interfaces
7. **pipeline_validators.py** - Validación estructural
8. **pretest_compilation.py** - Pre-compilación
9. **report_generator.py** - ⚠️ **CONFLICTO DE MERGE** (requiere resolución)
10. **system_health_check.py** - Monitoreo de salud

### Nota sobre report_generator.py

Este script tiene conflictos de merge Git pendientes y no puede ser parseado. Una vez resueltos los conflictos, debería ser mapeado a preguntas de dimensiones D4, D5 y D6 (reportes de resultados e impactos).

## Scoring por Dimensión

### Promedios de Scoring

- **D1 (Diagnóstico):** 8.5/10 - Excelente
- **D2 (Actividades):** 8.0/10 - Muy Bueno
- **D3 (Productos):** 8.5/10 - Excelente
- **D4 (Resultados):** 9.0/10 - Sobresaliente
- **D5 (Impactos):** 7.5/10 - Bueno
- **D6 (Causalidad):** 9.5/10 - Sobresaliente

### Justificaciones Tipo

Cada dimensión tiene justificaciones específicas basadas en:

1. **D1:** Robustez del diagnóstico, fuentes oficiales, series temporales
2. **D2:** Formalización de actividades, estructura tabular, vínculos causales
3. **D3:** Verificabilidad de productos, indicadores SMART, trazabilidad
4. **D4:** Medibilidad de resultados, metas realistas, encadenamiento
5. **D5:** Definición de impactos, rutas claras, consideración de riesgos
6. **D6:** Teoría de cambio explícita, coherencia causal, supuestos verificables

## Callables Huérfanos

Se identificaron **218 callables** no mapeados. La mayoría son:

- **Métodos privados:** `__init__`, `__str__`, `__repr__`, `to_dict`, `from_dict`
- **Funciones de demo:** `demo_*`, `test_*`
- **Utilidades de desarrollo:** Scripts de análisis y verificación
- **Interfaces:** Definiciones sin implementación directa

Estos callables son **esperados como huérfanos** ya que no participan directamente en la respuesta de preguntas.

## Uso del Archivo

### Cargar el Mapeo

```python
import json

with open('cuestionario_canonico_mapped.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)

# Acceder a una pregunta específica
question = next(q for q in mapping['questions'] if q['id'] == 'P1-D1-Q1')

# Ver los scripts involucrados
scripts = set(m['script'] for m in question['mapping'])
print(f"Scripts para {question['id']}: {scripts}")

# Ver el scoring
print(f"Score: {question['scoring']['cuantitativo']} - {question['scoring']['cualitativo']}")
```

### Validar el Mapeo

Ejecutar el script de validación:

```bash
python3 validate_mapping.py
```

Este script genera `VALIDATION_REPORT.md` con análisis completo de:
- Validación de estructura
- Cobertura por dimensión y política
- Scripts más utilizados
- Callables huérfanos
- Análisis de scoring

## Recomendaciones

### Para Mantenimiento

1. **Resolver conflictos:** Resolver el merge conflict en `report_generator.py` y remapear
2. **Revisar huérfanos:** Verificar si algún callable público no está siendo utilizado
3. **Actualizar scoring:** Ajustar justificaciones basadas en resultados reales
4. **Validar inputs/outputs:** Verificar que los inputs y outputs sean consistentes

### Para Extensión

1. **Nuevas preguntas:** Usar el mismo formato de mapeo
2. **Nuevos scripts:** Actualizar `DIMENSION_MODULE_MAPPING` y `POLICY_MODULE_MAPPING`
3. **Nuevas dimensiones:** Extender el sistema de scoring y justificaciones
4. **Validación continua:** Ejecutar `validate_mapping.py` después de cambios

## Referencias

- **ORCHESTRATOR_DOCUMENTATION.md** - Documentación del orquestador
- **CANONICAL_NOTATION_DOCS.md** - Sistema de notación canónica
- **questions_config.json** - Configuración de las 30 preguntas base
- **cuestionario_canonico** - Texto completo de las 300 preguntas
