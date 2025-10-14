# Integración de Estándares DNP - Documentación

## Visión General

Este módulo integra los estándares del Departamento Nacional de Planeación (DNP) de Colombia al framework FARFAN-2.0, garantizando:

1. **Competencias Municipales**: Validación estricta de competencias propias y concurrentes
2. **Indicadores MGA**: Alineación con el catálogo oficial de indicadores de producto y resultado
3. **Lineamientos PDET**: Cumplimiento de requisitos especiales para municipios PDET

## Módulos Implementados

### 1. competencias_municipales.py

Catálogo completo de competencias municipales colombianas basado en:
- Constitución Política de Colombia (1991)
- Ley 136 de 1994 (Organización Municipal)
- Ley 715 de 2001 (Recursos y Competencias)
- Ley 1551 de 2012 (Modernización Municipal)

**Sectores Cubiertos (14 sectores):**
- Educación
- Salud
- Agua Potable y Saneamiento
- Vivienda
- Vías y Transporte
- Desarrollo Agropecuario
- Medio Ambiente
- Cultura, Deporte y Recreación
- Desarrollo Económico
- Atención a Grupos Vulnerables
- Justicia y Seguridad
- Ordenamiento Territorial
- Equipamiento Municipal
- Prevención y Atención de Desastres

**Tipos de Competencias:**
- PROPIA: Competencia exclusiva municipal
- CONCURRENTE: Compartida con departamento/nación
- COMPLEMENTARIA: Rol de apoyo

**Uso:**
```python
from competencias_municipales import CATALOGO_COMPETENCIAS

# Validar competencia
validacion = CATALOGO_COMPETENCIAS.validar_competencia_municipal(
    sector="educacion",
    descripcion="construcción de escuelas"
)

# Obtener competencias PDET prioritarias
pdet_competencias = CATALOGO_COMPETENCIAS.get_pdet_prioritarias()
```

### 2. mga_indicadores.py

Catálogo de 51 indicadores MGA (Metodología General Ajustada) del DNP:
- 28 Indicadores de Producto
- 23 Indicadores de Resultado

**Características:**
- Fórmulas de cálculo oficiales
- Fuentes de información verificadas
- Periodicidad de medición
- Alineación con ODS (Objetivos de Desarrollo Sostenible)
- Desagregaciones requeridas

**Sectores Cubiertos:**
- Educación (6 indicadores)
- Salud (6 indicadores)
- Agua Potable y Saneamiento (7 indicadores)
- Vivienda (3 indicadores)
- Vías y Transporte (3 indicadores)
- Desarrollo Agropecuario (3 indicadores)
- Medio Ambiente (3 indicadores)
- Cultura y Deporte (4 indicadores)
- Desarrollo Económico (3 indicadores)
- Atención a Grupos Vulnerables (3 indicadores)
- Justicia y Seguridad (3 indicadores)
- Ordenamiento Territorial (2 indicadores)
- Gestión del Riesgo (3 indicadores)
- Equipamiento Municipal (2 indicadores)

**Uso:**
```python
from mga_indicadores import CATALOGO_MGA

# Obtener indicador específico
indicador = CATALOGO_MGA.get_indicador("EDU-001")

# Buscar indicadores por sector
indicadores_educacion = CATALOGO_MGA.get_by_sector("educacion")

# Obtener solo indicadores de producto
productos = CATALOGO_MGA.get_indicadores_producto()

# Generar reporte de alineación
reporte = CATALOGO_MGA.generar_reporte_alineacion(["EDU-001", "EDU-020"])
```

### 3. pdet_lineamientos.py

Lineamientos especiales para los 170 municipios PDET basado en:
- Decreto 893 de 2017 (Creación PDET)
- Acuerdo Final de Paz (2016)
- Lineamientos DNP para municipios PDET
- Resolución 0464 de 2020 - ART

**8 Pilares PDET:**
1. Ordenamiento Social de la Propiedad y Uso del Suelo
2. Salud Rural
3. Educación Rural
4. Vivienda, Agua y Saneamiento
5. Reactivación Económica y Producción Agropecuaria
6. Reconciliación, Convivencia y Construcción de Paz
7. Sistema de Alimentación y Nutrición
8. Infraestructura y Conectividad Rural

**Requisitos PDET:**
- Participación comunitaria mínima: 70%
- Presupuesto inversión rural: 60%
- Alineación con PATR: 80%
- Participación de víctimas: 40%
- Participación de mujeres: 30%

**Uso:**
```python
from pdet_lineamientos import LINEAMIENTOS_PDET

# Obtener lineamientos por pilar
lineamientos_salud = LINEAMIENTOS_PDET.get_by_pilar(PilarPDET.SALUD_RURAL)

# Validar cumplimiento PDET
validacion = LINEAMIENTOS_PDET.validar_cumplimiento_pdet(
    participacion={"comunitaria": 75, "victimas": 45},
    presupuesto_rural=6_000_000_000,
    presupuesto_total=10_000_000_000,
    alineacion_patr=85.0
)

# Recomendar lineamientos
recomendaciones = LINEAMIENTOS_PDET.recomendar_lineamientos(
    sector="educacion",
    es_rural=True,
    poblacion_victimas=True
)
```

### 4. dnp_integration.py

Módulo integrador que orquesta la validación completa contra estándares DNP.

**Clase Principal: ValidadorDNP**

**Niveles de Cumplimiento:**
- EXCELENTE: >90%
- BUENO: 75-90%
- ACEPTABLE: 60-75%
- INSUFICIENTE: <60%

**Uso:**
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

# Generar reporte
reporte = validador.generar_reporte_cumplimiento(resultado)
print(reporte)
```

## Integración con CDAF Framework (dereck_beach)

El framework CDAF ahora incluye validación DNP automática:

### Uso Básico

```bash
# Procesamiento estándar
python dereck_beach documento.pdf --output-dir resultados/ --policy-code PDM2024

# Procesamiento para municipio PDET
python dereck_beach documento.pdf --output-dir resultados/ --policy-code PDM2024 --pdet
```

### Salidas Generadas

El framework genera automáticamente:

1. **{policy_code}_dnp_compliance_report.txt**: Reporte completo de cumplimiento DNP
   - Score total de cumplimiento
   - Validación de competencias municipales
   - Alineación con indicadores MGA
   - Cumplimiento de lineamientos PDET (si aplica)
   - Alertas críticas
   - Recomendaciones accionables

### Ejemplo de Reporte DNP

```
================================================================================
REPORTE DE CUMPLIMIENTO DE ESTÁNDARES DNP
Código de Política: PDM2024
================================================================================

RESUMEN EJECUTIVO
--------------------------------------------------------------------------------
Total de Proyectos/Metas Analizados: 25
Score Promedio de Cumplimiento: 85.5/100

Distribución por Nivel de Cumplimiento:
  • Excelente (>90%):      12 ( 48.0%)
  • Bueno (75-90%):        10 ( 40.0%)
  • Aceptable (60-75%):     2 (  8.0%)
  • Insuficiente (<60%):    1 (  4.0%)

VALIDACIÓN DETALLADA POR PROYECTO/META
================================================================================

1. META-EDU-001
--------------------------------------------------------------------------------
   Score: 95.0/100 | Nivel: EXCELENTE
   Competencias Municipales: ✓
     - Aplicables: EDU-001, EDU-002
   Indicadores MGA: ✓
     - Usados: EDU-020, EDU-021
   Lineamientos PDET: ✓
     - Cumplidos: 2

[...]
```

## Configuración Avanzada

### Configurar PDET Programáticamente

```python
from dereck_beach import CDAFFramework

framework = CDAFFramework(
    config_path=Path("config.yaml"),
    output_dir=Path("resultados"),
    log_level="INFO"
)

# Activar modo PDET
if framework.dnp_validator:
    framework.dnp_validator.es_municipio_pdet = True

# Procesar documento
framework.process_document(pdf_path, "PDM2024")
```

### Validación de Plan Completo

```python
from dnp_integration import validar_plan_desarrollo_completo

proyectos = [
    {
        "nombre": "Construcción escuelas rurales",
        "sector": "educacion",
        "descripcion": "...",
        "indicadores": ["EDU-020", "EDU-021"],
        "presupuesto": 2_000_000_000,
        "es_rural": True,
        "poblacion_victimas": True
    },
    # ... más proyectos
]

resultado_plan = validar_plan_desarrollo_completo(
    proyectos=proyectos,
    es_municipio_pdet=True,
    presupuesto_total=50_000_000_000,
    presupuesto_rural=30_000_000_000,
    participacion={
        "comunitaria": 75,
        "victimas": 45,
        "mujeres": 35,
        "jovenes": 25
    }
)

print(f"Tasa de cumplimiento: {resultado_plan['tasa_cumplimiento']}%")
```

## Referencias Normativas

### Competencias Municipales
- Constitución Política de Colombia (1991)
- Ley 136 de 1994 - Organización Municipal
- Ley 715 de 2001 - Sistema General de Participaciones
- Ley 1551 de 2012 - Modernización Municipal

### Indicadores MGA
- DNP - Metodología General Ajustada (MGA)
- Sistema de Seguimiento a Proyectos de Inversión (SPI)
- Banco de Proyectos de Inversión Nacional (BPIN)

### PDET
- Decreto 893 de 2017 - Creación de PDET
- Acuerdo Final para la Terminación del Conflicto (2016)
- Resolución 0464 de 2020 - Agencia de Renovación del Territorio
- PATR - Planes de Acción para la Transformación Regional (19 subregiones)

## Contacto y Soporte

Para preguntas sobre la implementación de estándares DNP:
- Departamento Nacional de Planeación: https://www.dnp.gov.co
- Agencia de Renovación del Territorio (ART): https://www.renovacionterritorio.gov.co
- Sistema General de Regalías: https://www.sgr.gov.co

## Changelog

### v2.0.0 (2025-01-20)
- ✅ Implementación inicial de catálogo de competencias municipales (17 competencias)
- ✅ Catálogo completo de indicadores MGA (51 indicadores)
- ✅ Lineamientos PDET (17 lineamientos, 8 pilares)
- ✅ Integración con framework CDAF
- ✅ Generación automática de reportes de cumplimiento DNP
