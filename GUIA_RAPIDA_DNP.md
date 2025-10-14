# Guía Rápida de Referencia - Estándares DNP

## 🎯 Validación Rápida de un Proyecto

```python
from dnp_integration import ValidadorDNP

# Crear validador
validador = ValidadorDNP(es_municipio_pdet=False)  # True para municipios PDET

# Validar proyecto
resultado = validador.validar_proyecto_integral(
    sector="educacion",  # o salud, agua_potable_saneamiento, etc.
    descripcion="Descripción del proyecto",
    indicadores_propuestos=["EDU-020", "EDU-021"],  # Códigos MGA
    presupuesto=1_000_000_000,  # En pesos colombianos
    es_rural=True,  # ¿Es zona rural?
    poblacion_victimas=False  # ¿Atiende víctimas del conflicto?
)

# Ver resultado
print(f"Score: {resultado.score_total}/100")
print(f"Nivel: {resultado.nivel_cumplimiento.value}")
```

## 📊 Indicadores MGA por Sector

### Educación
- **Producto**: EDU-020, EDU-021 (infraestructura, dotación)
- **Resultado**: EDU-001, EDU-002, EDU-003, EDU-010 (cobertura, deserción)

### Salud
- **Producto**: SAL-020, SAL-021 (centros de salud)
- **Resultado**: SAL-001, SAL-002, SAL-010, SAL-015 (vacunación, mortalidad, afiliación)

### Agua Potable y Saneamiento
- **Producto**: APS-020, APS-021, APS-030 (redes, soluciones)
- **Resultado**: APS-001, APS-002, APS-003, APS-010 (cobertura, calidad)

### Vías y Transporte
- **Producto**: VIA-010 (kilómetros mejorados)
- **Resultado**: VIA-001, VIA-002 (estado, accesibilidad)

### Desarrollo Agropecuario
- **Producto**: AGR-002, AGR-010 (asistencia técnica, riego)
- **Resultado**: AGR-001 (productividad)

## ✅ Validación de Competencias

```python
from competencias_municipales import CATALOGO_COMPETENCIAS

validacion = CATALOGO_COMPETENCIAS.validar_competencia_municipal(
    sector="educacion",
    descripcion="proyecto educativo"
)

if validacion['valido']:
    print(f"✓ Competencias aplicables: {validacion['competencias_aplicables']}")
    print(f"Base legal: {validacion['base_legal']}")
else:
    print("✗ Fuera de competencias municipales")
```

## 🏘️ Validación PDET (170 Municipios)

```python
from pdet_lineamientos import LINEAMIENTOS_PDET

# Verificar cumplimiento
validacion = LINEAMIENTOS_PDET.validar_cumplimiento_pdet(
    participacion={
        "comunitaria": 75.0,  # Mínimo 70%
        "victimas": 45.0,     # Mínimo 40%
        "mujeres": 35.0       # Mínimo 30%
    },
    presupuesto_rural=6_000_000_000,
    presupuesto_total=10_000_000_000,  # Mínimo 60% rural
    alineacion_patr=85.0  # Mínimo 80%
)

print(f"Cumple PDET: {validacion['cumple_requisitos']}")
```

## 📋 Uso del Framework Completo

```bash
# Municipio estándar
python dereck_beach plan.pdf --output-dir resultados/ --policy-code PDM2024

# Municipio PDET
python dereck_beach plan.pdf --output-dir resultados/ --policy-code PDM2024 --pdet
```

## 📈 Niveles de Cumplimiento

| Nivel | Score | Descripción |
|-------|-------|-------------|
| 🟢 EXCELENTE | >90% | Cumplimiento sobresaliente de todos los estándares |
| 🟡 BUENO | 75-90% | Cumplimiento adecuado, mejoras menores |
| 🟠 ACEPTABLE | 60-75% | Cumple mínimos, requiere mejoras |
| 🔴 INSUFICIENTE | <60% | No cumple estándares, revisión urgente |

## 🔍 Chequeo Rápido de Indicadores

```python
from mga_indicadores import CATALOGO_MGA

# Verificar si un indicador existe
ind = CATALOGO_MGA.get_indicador("EDU-020")
if ind:
    print(f"✓ {ind.nombre}")
    print(f"  Fórmula: {ind.formula}")
    print(f"  Tipo: {ind.tipo.value}")

# Buscar indicadores por palabra clave
resultados = CATALOGO_MGA.buscar_indicador_por_descripcion("cobertura")
for ind in resultados:
    print(f"- {ind.codigo}: {ind.nombre}")
```

## ⚠️ Alertas Críticas Comunes

1. **Competencias**: "Proyecto fuera de competencias municipales"
   - **Solución**: Verificar base legal (Ley 715/2001, Ley 1551/2012)

2. **MGA**: "Falta alineación con catálogo de indicadores MGA"
   - **Solución**: Agregar al menos 1 indicador de producto + 1 de resultado

3. **PDET**: "Participación comunitaria insuficiente"
   - **Solución**: Aumentar participación a >70%

4. **PDET**: "Presupuesto rural insuficiente"
   - **Solución**: Asignar >60% del presupuesto a zona rural

## 📚 Sectores de Competencias (14)

1. Educación
2. Salud
3. Agua Potable y Saneamiento
4. Vivienda
5. Vías y Transporte
6. Desarrollo Agropecuario
7. Medio Ambiente
8. Cultura, Deporte y Recreación
9. Desarrollo Económico
10. Atención a Grupos Vulnerables
11. Justicia y Seguridad
12. Ordenamiento Territorial
13. Equipamiento Municipal
14. Prevención y Atención de Desastres

## 🎯 8 Pilares PDET

1. Ordenamiento Social de la Propiedad
2. Salud Rural
3. Educación Rural
4. Vivienda, Agua y Saneamiento
5. Reactivación Económica
6. Reconciliación y Convivencia
7. Sistema de Alimentación
8. Infraestructura y Conectividad

## 📞 Recursos Oficiales

- **DNP**: https://www.dnp.gov.co
- **MGA**: https://mgapp.dnp.gov.co
- **PDET/ART**: https://www.renovacionterritorio.gov.co
- **SGR**: https://www.sgr.gov.co

## 🚀 Flujo de Validación Recomendado

1. Validar competencias municipales ✓
2. Asignar indicadores MGA ✓
3. Si es PDET: validar lineamientos ✓
4. Generar reporte de cumplimiento ✓
5. Implementar recomendaciones ✓

---

**Versión**: 2.0.0  
**Última actualización**: 2025-01-20
