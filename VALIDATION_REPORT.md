# Reporte de Validación: cuestionario_canonico_mapped.json

```
================================================================================
AUDITORÍA DE cuestionario_canonico_mapped.json
================================================================================

1. VALIDACIÓN DE ESTRUCTURA
----------------------------------------
✅ Estructura válida - todos los campos requeridos presentes

2. ANÁLISIS DE COBERTURA
----------------------------------------
Total preguntas: 300
Scripts únicos utilizados: 15
Callables únicos mapeados: 200

Cobertura por dimensión:
  D1: 50/50 (100.0%)
  D2: 50/50 (100.0%)
  D3: 50/50 (100.0%)
  D4: 50/50 (100.0%)
  D5: 50/50 (100.0%)
  D6: 50/50 (100.0%)

Cobertura por política:
  P1: 30/30 (100.0%)
  P10: 30/30 (100.0%)
  P2: 30/30 (100.0%)
  P3: 30/30 (100.0%)
  P4: 30/30 (100.0%)
  P5: 30/30 (100.0%)
  P6: 30/30 (100.0%)
  P7: 30/30 (100.0%)
  P8: 30/30 (100.0%)
  P9: 30/30 (100.0%)

Scripts más utilizados:
  competencias_municipales.py: 3300 mappings
  canonical_notation.py: 3075 mappings
  module_choreographer.py: 1836 mappings
  question_answering_engine.py: 1800 mappings
  smart_recommendations.py: 1700 mappings
  mga_indicadores.py: 1656 mappings
  dnp_integration.py: 1530 mappings
  orchestrator.py: 1500 mappings
  pdet_lineamientos.py: 1440 mappings
  pipeline_dag.py: 700 mappings

3. CALLABLES HUÉRFANOS
----------------------------------------
⚠️  218 callables no mapeados:
  - canonical_integration.py::GuiaCuestionarioIntegration (class)
  - canonical_integration.py::GuiaCuestionarioIntegration.__init__
  - canonical_integration.py::GuiaCuestionarioIntegration._load_config
  - canonical_integration.py::GuiaCuestionarioIntegration.create_canonical_evidence
  - canonical_integration.py::GuiaCuestionarioIntegration.get_all_canonical_ids
  - canonical_integration.py::GuiaCuestionarioIntegration.get_critical_dimensions
  - canonical_integration.py::GuiaCuestionarioIntegration.get_dimension_mapping
  - canonical_integration.py::GuiaCuestionarioIntegration.get_dimension_template
  - canonical_integration.py::GuiaCuestionarioIntegration.get_dimension_weight
  - canonical_integration.py::GuiaCuestionarioIntegration.get_required_elements
  - canonical_integration.py::GuiaCuestionarioIntegration.get_validation_patterns
  - canonical_integration.py::GuiaCuestionarioIntegration.is_critical_dimension
  - canonical_integration.py::GuiaCuestionarioIntegration.validate_questionnaire_structure
  - canonical_integration.py::demo_integration
  - canonical_notation.py::CanonicalID.__post_init__
  - canonical_notation.py::CanonicalID.__str__
  - canonical_notation.py::CanonicalID.to_dict
  - canonical_notation.py::EvidenceEntry.__post_init__
  - canonical_notation.py::EvidenceEntry.to_dict
  - canonical_notation.py::RubricKey.__post_init__
  ... y 198 más

NOTA: Callables privados (__init__, __str__, etc.) y de testing/demo
son esperados como huérfanos. Verificar solo callables públicos.

4. ANÁLISIS DE SCORING
----------------------------------------
Score promedio global: 8.50
Score mínimo: 7.50
Score máximo: 9.50

Score promedio por dimensión:
  D1: 8.50 (50 preguntas)
  D2: 8.00 (50 preguntas)
  D3: 8.50 (50 preguntas)
  D4: 9.00 (50 preguntas)
  D5: 7.50 (50 preguntas)
  D6: 9.50 (50 preguntas)

5. SCRIPTS NO UTILIZADOS
----------------------------------------

================================================================================
RESUMEN EJECUTIVO
================================================================================
✅ Completitud: 300/300 (100.0%)
✅ Score promedio: 8.50/10
✅ Scripts utilizados: 15
✅ Callables mapeados: 200
ℹ️  Callables huérfanos: 218

================================================================================
```
