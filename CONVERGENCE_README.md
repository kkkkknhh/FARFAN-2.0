# Convergence Verification - FARFAN 2.0

## ✅ Verificación Completa del Sistema - 100% Convergencia

Este sistema verifica la **convergencia perfecta** entre `cuestionario_canonico`, `questions_config.json` y `guia_cuestionario.json` para asegurar que las **300 preguntas** del sistema de evaluación FARFAN 2.0 estén correctamente alineadas.

---

## 🎯 Resultados de Verificación

```json
{
  "percent_questions_converged": 100.0,
  "issues_detected": 0,
  "critical_issues": 0,
  "total_questions_expected": 300
}
```

### ✓ Verificaciones Completadas

1. **Notación Canónica** (P#-D#-Q#)
   - ✅ 300/300 preguntas con formato válido
   - ✅ Sin IDs inválidos
   - ✅ Cobertura completa: 10 políticas × 6 dimensiones × 5 preguntas

2. **Consistencia de Scoring**
   - ✅ Todas las preguntas tienen rúbricas completas
   - ✅ 4 niveles definidos: excelente, bueno, aceptable, insuficiente
   - ✅ Rangos de puntaje correctos (0.0 - 1.0)

3. **Mapeo de Dimensiones**
   - ✅ 10 políticas mapeadas correctamente
   - ✅ Todos los pesos suman 1.0 por política
   - ✅ Dimensiones críticas especificadas

4. **Sin Mapeos Legacy**
   - ✅ Eliminados todos los patrones de "file_contributors"
   - ✅ Sistema usa únicamente notación canónica

5. **Referencias de Módulos**
   - ✅ Solo módulos aprobados: dnp_integration, dereck_beach, competencias_municipales, initial_processor_causal_policy

---

## 🚀 Uso Rápido

### Verificación Básica

```bash
python verify_convergence.py
```

Salida esperada:
```
======================================================================
FARFAN 2.0 - Convergence Verification Report
======================================================================

Convergence: 100.0%
Issues Found: 0
  Critical: 0
  High: 0
  Medium: 0
  Low: 0

✓ VERIFICATION PASSED
```

### Ejecutar Tests

```bash
python -m unittest test_convergence.py -v
```

### Ver Ejemplos

```bash
python example_convergence.py
```

---

## 📋 Notación Canónica - P#-D#-Q#

### Formato

**`P{1-10}-D{1-6}-Q{1-5}`**

Donde:
- **P** = Punto del Decálogo (Policy Area): P1 a P10
- **D** = Dimensión Causal: D1 a D6
- **Q** = Pregunta por Dimensión: Q1 a Q5

### Ejemplos Válidos

```
✓ P1-D1-Q1  → Género - Diagnóstico - Pregunta 1
✓ P4-D3-Q2  → DESC - Productos - Pregunta 2
✓ P10-D6-Q5 → Migración - Teoría de Cambio - Pregunta 5
```

### Ejemplos Inválidos

```
✗ P11-D1-Q1 → P11 no existe (solo P1-P10)
✗ P1-D7-Q1  → D7 no existe (solo D1-D6)
✗ P1-D1-Q0  → Q0 inválido (Q1-Q5 solamente)
✗ D1-Q1     → Falta especificar política
```

---

## 📊 Estructura del Sistema

### 10 Políticas (Puntos del Decálogo)

| ID | Política |
|----|----------|
| P1 | Derechos de las mujeres e igualdad de género |
| P2 | Prevención de la violencia y protección frente al conflicto |
| P3 | Ambiente sano, cambio climático y prevención de desastres |
| P4 | Derechos económicos, sociales y culturales (DESC) |
| P5 | Derechos de las víctimas y construcción de paz |
| P6 | Derecho al buen futuro de la niñez, adolescencia y juventud |
| P7 | Tierras y territorios |
| P8 | Líderes y defensores de derechos humanos |
| P9 | Crisis de derechos de personas privadas de la libertad (PPL) |
| P10 | Migración transfronteriza |

### 6 Dimensiones Causales

| ID | Dimensión | Enfoque |
|----|-----------|---------|
| D1 | Insumos | Diagnóstico y líneas base |
| D2 | Actividades | Diseño de intervención |
| D3 | Productos | Outputs verificables |
| D4 | Resultados | Outcomes medibles |
| D5 | Impactos | Transformaciones de largo plazo |
| D6 | Causalidad | Teoría de cambio explícita |

### Total: 300 Preguntas

```
10 políticas × 6 dimensiones × 5 preguntas = 300 preguntas
```

---

## 🔍 Sistema de Scoring

### Niveles de Cumplimiento

| Nivel | Etiqueta | Rango |
|-------|----------|-------|
| 5 | Excelente | 0.85 - 1.00 |
| 4 | Bueno | 0.70 - 0.84 |
| 3 | Aceptable | 0.55 - 0.69 |
| 2 | Insuficiente | 0.40 - 0.54 |
| 1 | No Cumplimiento | 0.00 - 0.39 |

### Agregación por Dimensión

Cada dimensión tiene:
- 5 preguntas (Q1-Q5)
- Pesos específicos por pregunta
- Umbral mínimo de aprobación (0.50-0.60 según criticidad)

### Agregación por Política

Cada política tiene:
- 6 dimensiones con pesos diferenciados
- Suma de pesos = 1.0
- Dimensiones críticas identificadas
- Umbral mínimo global: 0.60

---

## 📁 Archivos del Sistema

### Configuración Principal

1. **`questions_config.json`** (8,507 líneas)
   - Metadata del sistema (300 preguntas)
   - Definición de dimensiones con pesos
   - Puntos del Decálogo con indicadores DNP
   - 30 preguntas base (6D × 5Q)

2. **`guia_cuestionario`** (JSON, 1,589 líneas)
   - Mapeo dimensión-política
   - Templates de verificación causal
   - Checklist de verificación por pregunta (D#-Q#)
   - Sistema de scoring y agregación
   - Glosario causal con patrones de verificación

3. **`cuestionario_canonico`** (Markdown, 450 líneas)
   - 300 preguntas en formato legible
   - Organizadas por P#, D#, Q#
   - Texto completo de cada pregunta

### Scripts de Verificación

1. **`verify_convergence.py`** - Script principal
   - Carga y valida los 3 archivos
   - Verifica notación canónica
   - Valida scoring y mappings
   - Detecta legacy patterns
   - Genera `convergence_report.json`

2. **`test_convergence.py`** - Suite de tests
   - 12 tests automatizados
   - Verificación completa del sistema
   - Validación de estructura de archivos

3. **`example_convergence.py`** - Ejemplos de uso
   - 8 ejemplos prácticos
   - Casos de uso comunes
   - Integración con canonical_notation.py

### Documentación

- **`CONVERGENCE_VERIFICATION_DOCS.md`** - Documentación completa
- **`CANONICAL_NOTATION_DOCS.md`** - Documentación de notación canónica

---

## 🛠️ Integración con FARFAN 2.0

### Uso en Question Answering Engine

```python
from verify_convergence import ConvergenceVerifier
from canonical_notation import CanonicalID

# 1. Verificar convergencia antes de procesar
verifier = ConvergenceVerifier()
report = verifier.run_full_verification()

if report['verification_summary']['critical_issues'] > 0:
    raise ValueError("Sistema no convergente - resolver issues críticos")

# 2. Procesar preguntas con IDs canónicos
question_id = "P1-D1-Q1"
canonical = CanonicalID.from_string(question_id)

policy = canonical.policy        # "P1"
dimension = canonical.dimension  # "D1"
question_num = canonical.question  # 1

# 3. Obtener metadata de guía
guia = verifier.guia_cuestionario
dimension_weight = guia['decalogo_dimension_mapping'][policy][f'{dimension}_weight']
```

### Pre-commit Hook (Recomendado)

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running convergence verification..."
python verify_convergence.py

if [ $? -ne 0 ]; then
    echo "❌ Convergence verification failed"
    echo "Fix critical issues before committing"
    exit 1
fi

echo "✓ Convergence verified"
```

---

## 📈 Métricas de Calidad

### Cobertura de Preguntas

- ✅ 100% de preguntas documentadas (300/300)
- ✅ 100% con notación canónica válida
- ✅ 100% con scoring completo
- ✅ 0 gaps en la matriz P×D×Q

### Consistencia de Configuración

- ✅ 0 conflictos entre archivos
- ✅ 0 referencias a archivos legacy
- ✅ 0 errores de validación
- ✅ 0 issues críticos

### Calidad de Metadata

- ✅ Todos los pesos de dimensión suman 1.0
- ✅ Todos los niveles de scoring definidos
- ✅ Todas las dimensiones críticas especificadas
- ✅ Todos los módulos referenciados existen

---

## 🎓 Casos de Uso

### 1. Validar antes de deployment

```bash
# CI/CD pipeline
python verify_convergence.py
if [ $? -eq 0 ]; then
    echo "✓ Ready for deployment"
else
    echo "✗ Fix convergence issues first"
    exit 1
fi
```

### 2. Auditar cambios en configuración

```python
# Después de modificar questions_config.json
from verify_convergence import ConvergenceVerifier

verifier = ConvergenceVerifier()
report = verifier.run_full_verification()

for issue in report['convergence_issues']:
    print(f"{issue['severity']}: {issue['description']}")
```

### 3. Generar documentación automática

```python
# Listar todas las preguntas de una política
for d in range(1, 7):
    for q in range(1, 6):
        qid = f"P1-D{d}-Q{q}"
        print(f"- {qid}")
```

---

## 🔧 Troubleshooting

### Error: "JSONDecodeError: Extra data"

**Causa:** Múltiples objetos JSON en un archivo  
**Solución:** El verifier maneja esto automáticamente

### Error: Invalid canonical notation

**Causa:** ID no sigue formato P#-D#-Q#  
**Solución:** Corregir IDs en archivos fuente

### Warning: Dimension weights don't sum to 1.0

**Causa:** Error de redondeo o valores incorrectos  
**Solución:** Ajustar pesos en guia_cuestionario

---

## 📚 Referencias

- [CONVERGENCE_VERIFICATION_DOCS.md](CONVERGENCE_VERIFICATION_DOCS.md) - Documentación detallada
- [CANONICAL_NOTATION_DOCS.md](CANONICAL_NOTATION_DOCS.md) - Sistema de notación
- [canonical_notation.py](canonical_notation.py) - Módulo de validación
- [DNP_INTEGRATION_DOCS.md](DNP_INTEGRATION_DOCS.md) - Integración DNP

---

## 🏆 Estado del Sistema

```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   ✅  SISTEMA COMPLETAMENTE CONVERGENTE                 ║
║                                                          ║
║   • 300/300 preguntas validadas                         ║
║   • 0 issues críticos                                   ║
║   • 100% convergencia                                   ║
║   • Listo para producción                               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

---

**Versión:** 1.0.0  
**Fecha:** 2025-10-14  
**Autor:** AI Systems Architect  
**Licencia:** Proyecto FARFAN 2.0
