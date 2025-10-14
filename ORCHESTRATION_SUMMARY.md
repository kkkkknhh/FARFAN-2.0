# FARFAN 2.0 - Sistema de Orquestación: Resumen Ejecutivo

## Visión General

Se ha implementado un **sistema completo de orquestación** que integra todos los módulos del framework FARFAN 2.0 en un flujo canónico, determinista e inmutable para evaluar Planes de Desarrollo Municipal mediante **300 preguntas de evaluación causal**.

## Componentes Implementados

### 1. Archivos Principales

| Archivo | Líneas | Descripción |
|---------|--------|-------------|
| `orchestrator.py` | 523 | Orquestador principal con flujo de 9 etapas canónicas |
| `question_answering_engine.py` | 590 | Motor de respuesta a 300 preguntas con coordinación de módulos |
| `report_generator.py` | 747 | Generador de reportes a 3 niveles (micro, meso, macro) |
| `module_choreographer.py` | 370 | Coreógrafo que registra ejecución y acumula respuestas |
| `questions_config.json` | 905 | Configuración completa de 30 preguntas base con criterios |
| `demo_orchestrator.py` | 303 | Script de demostración del sistema |

### 2. Documentación

| Archivo | Descripción |
|---------|-------------|
| `ORCHESTRATION_README.md` | Guía de uso del sistema de orquestación |
| `ORCHESTRATOR_DOCUMENTATION.md` | Documentación técnica detallada del flujo |

## Sistema de 300 Preguntas

**Estructura**: 30 Preguntas Base × 10 Áreas de Política = **300 Preguntas Totales**

Ver `questions_config.json` para especificación completa de:
- 30 preguntas base organizadas en 6 dimensiones (D1-D6)
- 10 áreas de política (P1-P10) del decálogo
- Criterios de evaluación detallados
- Patrones de verificación
- Sistema de scoring multinivel

## Reportes a 3 Niveles

### MICRO: 300 Respuestas Individuales
- Respuesta directa + argumento doctoral (2+ párrafos)
- Nota cuantitativa (0.0-1.0)
- Evidencia del documento
- Trazabilidad de módulos

### MESO: 4 Clústeres × 6 Dimensiones
- C1: Seguridad, Paz y Protección
- C2: Derechos Sociales
- C3: Territorio y Ambiente
- C4: Poblaciones Especiales

### MACRO: Alineación Global
- Score global y DNP
- Análisis retrospectivo/prospectivo
- Recomendaciones prioritarias
- Fortalezas y debilidades

## Garantías del Sistema

✓ **Determinismo**: Mismo input → mismo output  
✓ **Inmutabilidad**: Datos originales nunca modificados  
✓ **Completitud**: TODOS los módulos y funciones utilizados  
✓ **Trazabilidad**: Evidencia vinculada, metadata completa  

## Uso

```bash
# Procesar un plan
python orchestrator.py plan.pdf --policy-code PDM2024-ANT-MED --output-dir ./resultados

# Demo
python demo_orchestrator.py --simple
```

## Estado

**✅ COMPLETADO** - Sistema listo para procesar 170 planes de desarrollo

Ver documentación completa en:
- `ORCHESTRATION_README.md` - Guía de usuario
- `ORCHESTRATOR_DOCUMENTATION.md` - Documentación técnica
- `questions_config.json` - Configuración de preguntas

---
**Versión**: 1.0.0 | **Fecha**: 14 de octubre de 2024
