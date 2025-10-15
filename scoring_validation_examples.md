# Validación de Scoring - Ejemplos Input/Output

## Resumen

**Total de ejemplos**: 8
**Ejemplos validados**: 4
**Tasa de validación**: 50.0%

---

## Ejemplos Detallados

### Ejemplo 1: Proyecto con teoría de cambio clara y completa

**Status**: ⚠️ DESVIACIÓN

#### Inputs:

- **dereck_beach.CausalExtractor.extract_causal_hierarchy**
  - `text`: Mejorar la seguridad alimentaria mediante la implementación de huertas comunitarias que incrementen el acceso a alimentos nutritivos
  - `nodes_extracted`: 5
  - `causal_links`: 4
  - `hierarchy_levels`: 4

#### Output Esperado:

- **Score**: 0.85
- **Cualitativo**: Excelente - Teoría de cambio clara con cadena causal bien definida
- **Justificación**: El proyecto presenta una jerarquía causal completa desde insumos hasta impactos, con 5 nodos conectados lógicamente y 4 vínculos causales explícitos.
- **Evidencia**: 5 nodos causales identificados, 4 vínculos causales explícitos, Jerarquía de 4 niveles presente

#### Output Actual:

- **Score**: 1.00
- **Diferencia**: 0.150
- **Cualitativo**: Excelente

---

### Ejemplo 2: Proyecto con teoría de cambio incompleta

**Status**: ⚠️ DESVIACIÓN

#### Inputs:

- **dereck_beach.CausalExtractor.extract_causal_hierarchy**
  - `text`: Construir escuelas para mejorar la educación
  - `nodes_extracted`: 2
  - `causal_links`: 1
  - `hierarchy_levels`: 2

#### Output Esperado:

- **Score**: 0.45
- **Cualitativo**: Insuficiente - Cadena causal incompleta, falta operacionalización
- **Justificación**: El proyecto solo identifica 2 nodos (producto e impacto) sin especificar mecanismos intermedios ni resultados medibles.
- **Evidencia**: Solo 2 nodos causales, 1 vínculo causal, Faltan niveles intermedios

#### Output Actual:

- **Score**: 0.85
- **Diferencia**: 0.400
- **Cualitativo**: Excelente

---

### Ejemplo 3: Proyecto alineado con estándares DNP

**Status**: ✅ VALIDADO

#### Inputs:

- **dnp_integration.ValidadorDNP.validar_proyecto_integral**
  - `sector`: Educación
  - `descripcion`: Programa de mejoramiento de infraestructura educativa
  - `indicadores_propuestos`: ['Tasa de deserción', 'Puntaje SABER']
  - `mga_aligned`: True
  - `competencia_valida`: True

#### Output Esperado:

- **Score**: 0.9
- **Cualitativo**: Excelente - Cumple estándares DNP y MGA
- **Justificación**: El proyecto está alineado con indicadores MGA del sector Educación, es competencia municipal válida y usa indicadores estándar del DNP.
- **Evidencia**: Alineado con MGA, Competencia municipal válida, Indicadores DNP presentes

#### Output Actual:

- **Score**: 1.00
- **Diferencia**: 0.100
- **Cualitativo**: Excelente

---

### Ejemplo 4: Proyecto con presupuesto trazable

**Status**: ✅ VALIDADO

#### Inputs:

- **dereck_beach.FinancialAuditor.trace_financial_allocation**
  - `total_budget`: 1000000000
  - `allocations_by_node`: {'P1': 300000000, 'P2': 400000000, 'P3': 300000000}
  - `traceability_score`: 0.95

#### Output Esperado:

- **Score**: 0.88
- **Cualitativo**: Bueno - Presupuesto completamente trazable a nodos causales
- **Justificación**: El 100% del presupuesto está asignado a nodos específicos con trazabilidad clara. La distribución es coherente con la jerarquía causal.
- **Evidencia**: 100% del presupuesto asignado, Trazabilidad: 0.95, 3 nodos con asignación financiera

#### Output Actual:

- **Score**: 0.90
- **Diferencia**: 0.022
- **Cualitativo**: Excelente

---

### Ejemplo 5: Proyecto dentro de competencias municipales

**Status**: ✅ VALIDADO

#### Inputs:

- **competencias_municipales.CatalogoCompetenciasMunicipales.validar_competencia_municipal**
  - `sector`: Salud
  - `subsector`: Atención Primaria
  - `nivel_gobierno`: Municipal
  - `es_competencia_valida`: True

#### Output Esperado:

- **Score**: 1.0
- **Cualitativo**: Excelente - Competencia municipal válida
- **Justificación**: La atención primaria en salud es competencia directa del nivel municipal según la Ley 715 de 2001.
- **Evidencia**: Competencia municipal validada, Subsector: Atención Primaria, Base legal: Ley 715/2001

#### Output Actual:

- **Score**: 1.00
- **Diferencia**: 0.000
- **Cualitativo**: Excelente

---

### Ejemplo 6: Proyecto con indicadores MGA alineados

**Status**: ⚠️ DESVIACIÓN

#### Inputs:

- **mga_indicadores.CatalogoIndicadoresMGA.buscar_por_sector**
  - `sector`: Agua potable y saneamiento básico
  - `indicadores_encontrados`: 5
  - `indicadores_proyecto`: 3
  - `match_percentage`: 60

#### Output Esperado:

- **Score**: 0.75
- **Cualitativo**: Bueno - 60% de alineación con indicadores MGA
- **Justificación**: El proyecto usa 3 de 5 indicadores MGA recomendados para el sector, mostrando buena alineación con estándares nacionales.
- **Evidencia**: 5 indicadores MGA disponibles, 3 indicadores usados, 60% de alineación

#### Output Actual:

- **Score**: 0.60
- **Diferencia**: 0.150
- **Cualitativo**: Aceptable

---

### Ejemplo 7: Proyecto en municipio PDET alineado con lineamientos

**Status**: ⚠️ DESVIACIÓN

#### Inputs:

- **pdet_lineamientos.LineamientosPDET.recomendar_lineamientos**
  - `es_municipio_pdet`: True
  - `sector`: Rural
  - `lineamientos_aplicables`: 4
  - `lineamientos_cumplidos`: 3

#### Output Esperado:

- **Score**: 0.82
- **Cualitativo**: Bueno - 75% de alineación con lineamientos PDET
- **Justificación**: El proyecto cumple 3 de 4 lineamientos PDET aplicables al sector rural, demostrando compromiso con los Programas de Desarrollo con Enfoque Territorial.
- **Evidencia**: 4 lineamientos PDET aplicables, 3 lineamientos cumplidos, Municipio PDET confirmado

#### Output Actual:

- **Score**: 0.50
- **Diferencia**: 0.320
- **Cualitativo**: Insuficiente

---

### Ejemplo 8: Score final amalgamado de múltiples módulos

**Status**: ✅ VALIDADO

#### Inputs:

- **dereck_beach.CausalExtractor.extract_causal_hierarchy**
  - `score`: 0.85

- **dnp_integration.ValidadorDNP.validar_proyecto_integral**
  - `score`: 0.9

- **competencias_municipales.CatalogoCompetenciasMunicipales.validar_competencia_municipal**
  - `score`: 1.0

- **mga_indicadores.CatalogoIndicadoresMGA.buscar_por_sector**
  - `score`: 0.75

#### Output Esperado:

- **Score**: 0.86
- **Cualitativo**: Excelente - Proyecto bien fundamentado en múltiples dimensiones
- **Justificación**: Score amalgamado: Causal (0.85×0.30) + DNP (0.90×0.25) + Competencias (1.0×0.15) + MGA (0.75×0.15) + PDET (0.80×0.15) = 0.86. El proyecto demuestra solidez técnica, cumplimiento normativo y alineación con estándares nacionales.
- **Evidencia**: Teoría de cambio clara (0.85), Cumplimiento DNP alto (0.90), Competencia válida (1.0), Alineación MGA buena (0.75)

#### Output Actual:

- **Score**: 0.87
- **Diferencia**: 0.014
- **Cualitativo**: Excelente

---
