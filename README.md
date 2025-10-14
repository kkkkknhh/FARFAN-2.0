# FARFAN-2.0
Framework Avanzado de Reconstrucción y Análisis de Formulaciones de Acción Nacional 2.0

## Descripción

FARFAN-2.0 es un framework de grado industrial para la deconstrucción y auditoría causal de Planes de Desarrollo Territorial en Colombia, con énfasis en cumplimiento riguroso de estándares del DNP (Departamento Nacional de Planeación).

## Características Principales

### 1. **NUEVO: Sistema de Evaluación de 300 Preguntas**

El orquestador implementa un sistema completo de evaluación mediante **300 preguntas causales**:

- **30 Preguntas Base**: Organizadas en 6 dimensiones del Marco Lógico
  - D1: Insumos (Diagnóstico y Líneas Base)
  - D2: Actividades (Formalizadas)
  - D3: Productos (Verificables)
  - D4: Resultados (Medibles)
  - D5: Impactos (Largo Plazo)
  - D6: Causalidad (Teoría de Cambio)

- **10 Áreas de Política** (Decálogo):
  - P1: Seguridad y Convivencia
  - P2: Alertas Tempranas
  - P3: Ambiente y Recursos Naturales
  - P4: Derechos Básicos
  - P5: Víctimas
  - P6: Niñez y Juventud
  - P7: Desarrollo Rural
  - P8: Líderes Sociales
  - P9: Sistema Carcelario
  - P10: Migración

**Cada respuesta incluye**:
- Texto de respuesta directa
- Argumento de nivel doctoral (2+ párrafos)
- Nota cuantitativa (0.0-1.0)
- Evidencia del documento
- Módulos que contribuyeron

**Reportes a 3 Niveles**:
1. **MICRO**: 300 respuestas individuales
2. **MESO**: 4 clústeres × 6 dimensiones
3. **MACRO**: Alineación global + análisis retrospectivo/prospectivo

### 2. Framework CDAF (Causal Deconstruction and Audit Framework)
- Extracción automática de jerarquías causales desde PDFs
- Análisis de mecanismos causales (Entidad-Actividad)
- Trazabilidad financiera
- Auditoría de operacionalización
- Generación de diagramas causales y matrices de responsabilidad

### 3. **NUEVO: Cumplimiento Integral de Estándares DNP**

#### Competencias Municipales
- **17 competencias** catalogadas según normativa colombiana
- Validación automática de competencias propias y concurrentes
- Base legal completa (Ley 136/1994, Ley 715/2001, Ley 1551/2012)
- 14 sectores de intervención cubiertos

#### Indicadores MGA
- **51 indicadores** del catálogo oficial MGA
  - 28 indicadores de producto
  - 23 indicadores de resultado
- Fórmulas de cálculo oficiales
- Fuentes de información verificadas
- Alineación con ODS (Objetivos de Desarrollo Sostenible)

#### Lineamientos PDET
- **17 lineamientos** para los 170 municipios PDET
- **8 pilares** del Acuerdo de Paz implementados
- Validación especial de participación comunitaria
- Requisitos de inversión rural (>60%)
- Alineación con PATR subregionales

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/kkkkknhh/FARFAN-2.0.git
cd FARFAN-2.0

# Instalar dependencias (opcional, para framework completo)
pip install pymupdf networkx pandas spacy pyyaml fuzzywuzzy python-Levenshtein pydot

# Descargar modelo spaCy español
python -m spacy download es_core_news_lg
```

## Uso Rápido

### Sistema de Orquestación Completo (NUEVO)

El orquestador integra **todos los módulos** para evaluar planes mediante **300 preguntas**:

```bash
# Procesar un plan de desarrollo
python orchestrator.py plan_desarrollo.pdf \
    --policy-code PDM2024-ANT-MED \
    --output-dir ./resultados \
    --pdet

# Demostración del sistema
python demo_orchestrator.py --simple
```

**Salida generada**:
- `micro_report_{code}.json` - 300 respuestas individuales
- `meso_report_{code}.json` - 4 clústeres × 6 dimensiones
- `macro_report_{code}.json/md` - Evaluación global

Ver [ORCHESTRATION_README.md](ORCHESTRATION_README.md) para documentación completa.

### Validación DNP Standalone

```python
from dnp_integration import ValidadorDNP

validador = ValidadorDNP(es_municipio_pdet=True)

resultado = validador.validar_proyecto_integral(
    sector="educacion",
    descripcion="Construcción de 5 sedes educativas en zona rural",
    indicadores_propuestos=["EDU-020", "EDU-021", "EDU-002"],
    presupuesto=2_000_000_000,
    es_rural=True,
    poblacion_victimas=True
)

print(validador.generar_reporte_cumplimiento(resultado))
```

### Framework Completo CDAF

```bash
# Procesamiento estándar
python dereck_beach documento.pdf --output-dir resultados/ --policy-code PDM2024

# Procesamiento para municipio PDET
python dereck_beach documento.pdf --output-dir resultados/ --policy-code PDM2024 --pdet
```

### Ejemplos Interactivos

```bash
# Ejecutar ejemplos completos
python ejemplo_dnp_completo.py
```

## Módulos

### **NUEVO: Sistema de Orquestación Integral**
- `orchestrator.py` - Orquestador principal con flujo canónico de 9 etapas
- `question_answering_engine.py` - Motor de respuesta a 300 preguntas
- `report_generator.py` - Generador de reportes micro, meso y macro
- `module_choreographer.py` - Coreógrafo de módulos y acumulador de respuestas

### Módulos DNP (Nuevos)
- `competencias_municipales.py` - Catálogo de competencias municipales
- `mga_indicadores.py` - Catálogo de indicadores MGA
- `pdet_lineamientos.py` - Lineamientos PDET
- `dnp_integration.py` - Integración y validación DNP
- `ejemplo_dnp_completo.py` - Ejemplos de uso

### Módulos Framework Principal
- `dereck_beach` - Framework CDAF principal
- `initial_processor_causal_policy` - Procesador de políticas causales
- `teoria_cambio_validacion_monte_carlo` - Validación de teoría de cambio
- `guia_cuestionario` - Cuestionario de validación causal

## Salidas Generadas

El framework genera automáticamente:

1. **{policy_code}_causal_diagram.png** - Diagrama causal visual
2. **{policy_code}_accountability_matrix.md** - Matriz de responsabilidades
3. **{policy_code}_confidence_report.json** - Reporte de confianza
4. **{policy_code}_causal_model.json** - Modelo causal estructurado
5. **{policy_code}_dnp_compliance_report.txt** - **NUEVO:** Reporte de cumplimiento DNP

## Documentación

- [DNP Integration Documentation](DNP_INTEGRATION_DOCS.md) - Guía completa de validación DNP
- Ver ejemplos en `ejemplo_dnp_completo.py`

## Estándares y Normativa

### Competencias Municipales
- Constitución Política de Colombia (1991)
- Ley 136 de 1994 - Organización Municipal
- Ley 715 de 2001 - Sistema General de Participaciones
- Ley 1551 de 2012 - Modernización Municipal

### Indicadores MGA
- DNP - Metodología General Ajustada (MGA)
- Sistema de Seguimiento a Proyectos de Inversión (SPI)

### PDET
- Decreto 893 de 2017 - Creación de PDET
- Acuerdo Final para la Terminación del Conflicto (2016)
- Agencia de Renovación del Territorio (ART)

## Niveles de Cumplimiento DNP

- **EXCELENTE**: >90% - Cumplimiento sobresaliente
- **BUENO**: 75-90% - Cumplimiento adecuado
- **ACEPTABLE**: 60-75% - Cumplimiento mínimo
- **INSUFICIENTE**: <60% - Requiere mejoras

## Contribuciones

Este proyecto implementa estándares oficiales del DNP y el Acuerdo de Paz de Colombia. Las contribuciones deben mantener estricta adherencia a la normativa colombiana vigente.

## Licencia

Ver archivo LICENSE

## Contacto

Para soporte sobre estándares DNP:
- DNP: https://www.dnp.gov.co
- ART: https://www.renovacionterritorio.gov.co

