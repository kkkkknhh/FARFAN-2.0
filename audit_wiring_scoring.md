# Auditoría de Wiring, Flujo y Scoring - FARFAN 2.0

**Fecha de auditoría**: 1.0
**Repositorio**: /home/runner/work/FARFAN-2.0/FARFAN-2.0
**Módulos analizados**: 14
**Pasos en flujo canónico**: 18

---

## 1. Resumen Ejecutivo

### Cobertura de Funciones
- **Total de funciones públicas**: 209
- **Funciones en flujo canónico**: 18
- **Cobertura**: 8.61%
- **Todas las funciones utilizadas**: ⚠️ NO

### Scoring
- **Componentes de scoring**: 8
- **Scoring verificado**: ✅ SÍ

### Problemas Detectados: ⚠️ 1

- 🟡 172 funciones públicas no utilizadas en el flujo canónico

---

## 2. Flujo de Datos entre Módulos

| Stage | Script | Función | Inputs | Transmite a | Formato |
|-------|--------|---------|--------|-------------|---------|
| STAGE_1_2 | dereck_beach | PDFProcessor.load_document | pdf_path | next_stage | PipelineContext |
| STAGE_1_2 | dereck_beach | PDFProcessor.extract_text | — | next_stage | PipelineContext |
| STAGE_1_2 | dereck_beach | PDFProcessor.extract_tables | — | next_stage | PipelineContext |
| STAGE_1_2 | dereck_beach | PDFProcessor.extract_sections | — | next_stage | PipelineContext |
| STAGE_3 | initial_processor_causal_policy | PolicyDocumentAnalyzer.analyze_document | text | next_stage | PipelineContext |
| STAGE_4 | dereck_beach | CausalExtractor.extract_causal_hierarchy | text | next_stage | PipelineContext |
| STAGE_4 | dereck_beach | CausalExtractor.classify_goal | text | next_stage | PipelineContext |
| STAGE_5 | dereck_beach | MechanismPartExtractor.extract_entity_activity | text | next_stage | PipelineContext |
| STAGE_5 | dereck_beach | BayesianMechanismInference.infer_mechanism | node, observations | next_stage | PipelineContext |
| STAGE_6 | dereck_beach | FinancialAuditor.trace_financial_allocation | tables, nodes | next_stage | PipelineContext |
| STAGE_7 | dnp_integration | ValidadorDNP.validar_proyecto_integral | sector, descripcion, indicadores_propuestos | next_stage | PipelineContext |
| STAGE_7 | competencias_municipales | CatalogoCompetenciasMunicipales.validar_competencia_municipal | sector | next_stage | PipelineContext |
| STAGE_7 | mga_indicadores | CatalogoIndicadoresMGA.buscar_por_sector | sector | next_stage | PipelineContext |
| STAGE_7 | pdet_lineamientos | LineamientosPDET.recomendar_lineamientos | sector | next_stage | PipelineContext |
| STAGE_8 | question_answering_engine | QuestionAnsweringEngine.answer_all_questions | pipeline_context | next_stage | PipelineContext |
| STAGE_9 | report_generator | ReportGenerator.generate_micro_report | question_responses, policy_code | next_stage | PipelineContext |
| STAGE_9 | report_generator | ReportGenerator.generate_meso_report | question_responses, policy_code | next_stage | PipelineContext |
| STAGE_9 | report_generator | ReportGenerator.generate_macro_report | question_responses, compliance_score, policy_code | next_stage | PipelineContext |

---

## 3. Componentes de Scoring

### dereck_beach.CausalExtractor.extract_causal_hierarchy
- **Contribuye a**: quantitative, qualitative, justification
- **Peso**: 0.3

### dereck_beach.FinancialAuditor.trace_financial_allocation
- **Contribuye a**: quantitative, qualitative, justification
- **Peso**: 0.3

### dereck_beach.OperationalizationAuditor.audit_node_completeness
- **Contribuye a**: quantitative, qualitative, justification
- **Peso**: 0.3

### dnp_integration.ValidadorDNP.validar_proyecto_integral
- **Contribuye a**: quantitative, compliance
- **Peso**: 0.25

### dnp_integration.ValidadorDNP.validar_indicador_mga
- **Contribuye a**: quantitative, compliance
- **Peso**: 0.25

### competencias_municipales.CatalogoCompetenciasMunicipales.validar_competencia_municipal
- **Contribuye a**: qualitative, compliance
- **Peso**: 0.15

### mga_indicadores.CatalogoIndicadoresMGA.buscar_por_sector
- **Contribuye a**: quantitative, justification
- **Peso**: 0.15

### pdet_lineamientos.LineamientosPDET.recomendar_lineamientos
- **Contribuye a**: qualitative, justification
- **Peso**: 0.15


---

## 4. Funciones No Utilizadas

⚠️ **172 funciones públicas no utilizadas:**

- `orchestrator.FARFANOrchestrator.process_plan` (línea 317)
- `orchestrator.process_plan` (línea 317)
- `module_choreographer.ResponseAccumulator.add_contribution` (línea 42)
- `module_choreographer.ResponseAccumulator.synthesize` (línea 56)
- `module_choreographer.ModuleChoreographer.register_module` (línea 94)
- `module_choreographer.ModuleChoreographer.execute_module_stage` (línea 99)
- `module_choreographer.ModuleChoreographer.accumulate_for_question` (línea 163)
- `module_choreographer.ModuleChoreographer.synthesize_responses` (línea 180)
- `module_choreographer.ModuleChoreographer.get_data_transfer_log` (línea 195)
- `module_choreographer.ModuleChoreographer.get_module_usage_report` (línea 221)
- `module_choreographer.ModuleChoreographer.verify_all_modules_used` (línea 256)
- `module_choreographer.ModuleChoreographer.verify_all_functions_used` (línea 278)
- `module_choreographer.ModuleChoreographer.generate_flow_diagram` (línea 302)
- `module_choreographer.ModuleChoreographer.generate_mermaid_diagram` (línea 332)
- `module_choreographer.ModuleChoreographer.compare_execution_trace` (línea 368)
- `module_choreographer.ModuleChoreographer.export_execution_trace` (línea 407)
- `module_choreographer.create_canonical_flow` (línea 436)
- `module_choreographer.add_contribution` (línea 42)
- `module_choreographer.synthesize` (línea 56)
- `module_choreographer.register_module` (línea 94)
- `module_choreographer.execute_module_stage` (línea 99)
- `module_choreographer.accumulate_for_question` (línea 163)
- `module_choreographer.synthesize_responses` (línea 180)
- `module_choreographer.get_data_transfer_log` (línea 195)
- `module_choreographer.get_module_usage_report` (línea 221)
- `module_choreographer.verify_all_modules_used` (línea 256)
- `module_choreographer.verify_all_functions_used` (línea 278)
- `module_choreographer.generate_flow_diagram` (línea 302)
- `module_choreographer.generate_mermaid_diagram` (línea 332)
- `module_choreographer.compare_execution_trace` (línea 368)
- `module_choreographer.export_execution_trace` (línea 407)
- `dereck_beach.MechanismTypeConfig.check_sum_to_one` (línea 162)
- `dereck_beach.ConfigLoader.get` (línea 455)
- `dereck_beach.ConfigLoader.get_bayesian_threshold` (línea 466)
- `dereck_beach.ConfigLoader.get_mechanism_prior` (línea 472)
- `dereck_beach.ConfigLoader.get_performance_setting` (línea 478)
- `dereck_beach.ConfigLoader.update_priors_from_feedback` (línea 484)
- `dereck_beach.OperationalizationAuditor.audit_evidence_traceability` (línea 1250)
- `dereck_beach.OperationalizationAuditor.audit_sequence_logic` (línea 1295)
- `dereck_beach.OperationalizationAuditor.bayesian_counterfactual_audit` (línea 1335)
- `dereck_beach.BayesianMechanismInference.infer_mechanisms` (línea 1681)
- `dereck_beach.CausalInferenceSetup.classify_goal_dynamics` (línea 2029)
- `dereck_beach.CausalInferenceSetup.assign_probative_value` (línea 2040)
- `dereck_beach.CausalInferenceSetup.identify_failure_points` (línea 2063)
- `dereck_beach.ReportingEngine.generate_causal_diagram` (línea 2109)
- `dereck_beach.ReportingEngine.generate_accountability_matrix` (línea 2196)
- `dereck_beach.ReportingEngine.generate_confidence_report` (línea 2255)
- `dereck_beach.ReportingEngine.generate_causal_model_json` (línea 2355)
- `dereck_beach.CDAFFramework.process_document` (línea 2444)
- `dereck_beach.check_sum_to_one` (línea 162)

*...y 122 más*

---

## 5. Estadísticas por Módulo

| Módulo | Funciones Totales | Funciones Públicas | Clases |
|--------|-------------------|-------------------|---------|
| orchestrator | 29 | 3 | 3 |
| module_choreographer | 31 | 29 | 3 |
| question_answering_engine | 24 | 2 | 5 |
| report_generator | 54 | 6 | 2 |
| dereck_beach | 159 | 53 | 26 |
| dnp_integration | 15 | 5 | 3 |
| competencias_municipales | 18 | 14 | 4 |
| mga_indicadores | 20 | 16 | 4 |
| pdet_lineamientos | 16 | 12 | 6 |
| initial_processor_causal_policy | 37 | 11 | 6 |
| pipeline_validators | 13 | 13 | 9 |
| resource_management | 12 | 10 | 1 |
| risk_mitigation_layer | 31 | 19 | 8 |
| circuit_breaker | 34 | 16 | 6 |

---

## 6. Grafo de Flujo de Datos

**Nodos totales**: 17
**Conexiones**: 35

### Principales conexiones:

- `STAGE_1_2` → `PipelineContext` (STAGE_1_2): raw_text
- `STAGE_1_2` → `PipelineContext` (STAGE_1_2): sections
- `STAGE_1_2` → `PipelineContext` (STAGE_1_2): tables
- `STAGE_3` → `PipelineContext` (STAGE_3): semantic_chunks
- `STAGE_3` → `PipelineContext` (STAGE_3): dimension_scores
- `STAGE_4` → `PipelineContext` (STAGE_4): causal_graph
- `STAGE_4` → `PipelineContext` (STAGE_4): nodes
- `STAGE_4` → `PipelineContext` (STAGE_4): causal_chains
- `STAGE_5` → `PipelineContext` (STAGE_5): mechanism_parts
- `STAGE_5` → `PipelineContext` (STAGE_5): bayesian_inferences
- `STAGE_6` → `PipelineContext` (STAGE_6): financial_allocations
- `STAGE_6` → `PipelineContext` (STAGE_6): budget_traceability
- `STAGE_7` → `PipelineContext` (STAGE_7): dnp_validation_results
- `STAGE_7` → `PipelineContext` (STAGE_7): compliance_score
- `STAGE_8` → `PipelineContext` (STAGE_8): question_responses
- `STAGE_9` → `PipelineContext` (STAGE_9): micro_report
- `STAGE_9` → `PipelineContext` (STAGE_9): meso_report
- `STAGE_9` → `PipelineContext` (STAGE_9): macro_report
- `dereck_beach` → `dereck_beach` (STAGE_1_2): PDFProcessor.load_document → PDFProcessor.extract_text
- `dereck_beach` → `dereck_beach` (STAGE_1_2): PDFProcessor.extract_text → PDFProcessor.extract_tables

---

## 7. Conclusiones

⚠️ **Se detectaron algunas áreas de mejora:**
- Hay 172 funciones sin utilizar